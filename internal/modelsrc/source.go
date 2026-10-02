// Package modelsrc fetches a model folder from an object store into a local cache, so the server can be
// pointed at s3://bucket/prefix/ (the three model files) or s3://bucket/key.tar.gz (that folder, packed)
// instead of a local path. Only s3:// is implemented; gs:// and azblob:// are one Store each (see schemes).
package modelsrc

import (
	"archive/tar"
	"compress/gzip"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/url"
	"os"
	"path"
	"path/filepath"
	"strings"
)

// The model folder's files (the same names cmd/openjevx loads). The tokenizer is optional.
const (
	GraphName     = "openjevx.w8.onnx"
	ConfigName    = "config.json"
	TokenizerName = "tokenizer.json"
)

// maxFile caps one extracted file, so a bad archive cannot fill the disk.
const maxFile = 4 << 30

var (
	// ErrNotFound is what a Store returns for a missing object.
	ErrNotFound = errors.New("not found")
	// ErrChecksum means the downloaded model is not the one that was configured.
	ErrChecksum = errors.New("checksum mismatch")
)

// Store reads objects from one bucket. Stat and Fetch return the object's ETag (any opaque version string).
type Store interface {
	Stat(ctx context.Context, key string) (etag string, err error)
	Fetch(ctx context.Context, key string, w io.Writer) (etag string, err error)
}

// Options are the model source settings from openjevx.json.
type Options struct {
	SHA256   string // optional pin: sha256 of the .tar.gz, or of openjevx.w8.onnx for a folder
	CacheDir string // "" = <user cache dir>/openjevx/models
	Endpoint string // "" = AWS; tests point it at a fake S3 (path-style requests)
}

type opener func(ctx context.Context, bucket string, opt Options) (Store, error)

// schemes maps a URL scheme to its Store. Adding gs:// or azblob:// is one opener here.
var schemes = map[string]opener{
	"s3":     openS3,
	"gs":     notYet("gs"),
	"azblob": notYet("azblob"),
}

func notYet(scheme string) opener {
	return func(context.Context, string, Options) (Store, error) {
		return nil, fmt.Errorf("%s:// model sources are not supported yet (only s3://)", scheme)
	}
}

// IsRemote reports whether a "model" setting is an object-store URL rather than a local path.
func IsRemote(s string) bool {
	i := strings.Index(s, "://")
	return i > 0 && schemes[strings.ToLower(s[:i])] != nil
}

// remote is one model location in a store: a folder prefix or a packed .tar.gz.
type remote struct {
	url     string
	store   Store
	key     string // the archive key, or the folder prefix ending in "/" ("" = bucket root)
	archive bool
}

func openRemote(ctx context.Context, raw string, opt Options) (*remote, error) {
	u, err := url.Parse(raw)
	if err != nil {
		return nil, fmt.Errorf("model %s: %w", raw, err)
	}
	open := schemes[strings.ToLower(u.Scheme)]
	if open == nil || u.Host == "" {
		return nil, fmt.Errorf("model %s: want s3://bucket/prefix/ or s3://bucket/model.tar.gz", raw)
	}
	store, err := open(ctx, u.Host, opt)
	if err != nil {
		return nil, err
	}
	r := &remote{url: raw, store: store, key: strings.TrimPrefix(u.Path, "/")}
	if strings.HasSuffix(r.key, ".tar.gz") || strings.HasSuffix(r.key, ".tgz") || strings.HasSuffix(r.key, ".tar") {
		r.archive = true
	} else if r.key != "" && !strings.HasSuffix(r.key, "/") {
		r.key += "/"
	}
	return r, nil
}

func (r *remote) folderKeys() []string {
	return []string{r.key + GraphName, r.key + ConfigName, r.key + TokenizerName}
}

// version is the remote's current version: the archive's ETag, or the three files' ETags joined.
func (r *remote) version(ctx context.Context) (string, error) {
	if r.archive {
		etag, err := r.store.Stat(ctx, r.key)
		if err != nil {
			return "", fmt.Errorf("model %s: %w", r.url, err)
		}
		return etag, nil
	}
	var tags []string
	for i, key := range r.folderKeys() {
		etag, err := r.store.Stat(ctx, key)
		if errors.Is(err, ErrNotFound) && i == 2 {
			etag, err = "-", nil
		}
		if err != nil {
			return "", fmt.Errorf("model %s: %s: %w", r.url, path.Base(key), err)
		}
		tags = append(tags, etag)
	}
	return strings.Join(tags, ","), nil
}

// download fetches the model into dir (which must exist and be empty) and verifies it.
// It returns the version it downloaded and the manifest to store next to it.
func (r *remote) download(ctx context.Context, dir string, opt Options) (*Model, error) {
	m := &Model{URL: r.url, Dir: dir}
	if r.archive {
		tmp := filepath.Join(dir, ".download.tar.gz")
		etag, sum, err := r.fetchFile(ctx, r.key, tmp)
		if err != nil {
			return nil, err
		}
		m.Version, m.SHA256 = etag, sum
		if err := extract(tmp, dir); err != nil {
			return nil, fmt.Errorf("model %s: %w", r.url, err)
		}
		_ = os.Remove(tmp)
	} else {
		var tags []string
		for i, key := range r.folderKeys() {
			etag, sum, err := r.fetchFile(ctx, key, filepath.Join(dir, path.Base(key)))
			if errors.Is(err, ErrNotFound) && i == 2 {
				etag, err = "-", nil
			}
			if err != nil {
				return nil, err
			}
			if i == 0 {
				m.SHA256 = sum
			}
			tags = append(tags, etag)
		}
		m.Version = strings.Join(tags, ",")
	}
	if opt.SHA256 != "" && !strings.EqualFold(opt.SHA256, m.SHA256) {
		return nil, fmt.Errorf("model %s: %w: sha256 %s, configured %s", r.url, ErrChecksum, m.SHA256, opt.SHA256)
	}
	if err := m.verify(); err != nil {
		return nil, err
	}
	return m, nil
}

func (r *remote) fetchFile(ctx context.Context, key, dst string) (etag, sum string, err error) {
	f, err := os.Create(dst)
	if err != nil {
		return "", "", err
	}
	h := sha256.New()
	etag, err = r.store.Fetch(ctx, key, io.MultiWriter(f, h))
	if cerr := f.Close(); err == nil {
		err = cerr
	}
	if err != nil {
		_ = os.Remove(dst)
		if errors.Is(err, ErrNotFound) {
			return "", "", err
		}
		return "", "", fmt.Errorf("model %s: %s: %w", r.url, path.Base(key), err)
	}
	return etag, hex.EncodeToString(h.Sum(nil)), nil
}

// extract unpacks the model files from a .tar.gz (or plain .tar) into dir. The files may sit at the root or in one folder
// (model/); anything else in the archive is ignored, and only base names are used, so no path escapes dir.
func extract(archive, dir string) error {
	f, err := os.Open(archive)
	if err != nil {
		return err
	}
	defer f.Close()
	var in io.Reader = f
	magic := make([]byte, 2)
	if _, err := io.ReadFull(f, magic); err == nil && magic[0] == 0x1f && magic[1] == 0x8b {
		if _, err := f.Seek(0, io.SeekStart); err != nil {
			return err
		}
		if in, err = gzip.NewReader(f); err != nil {
			return fmt.Errorf("bad .tar.gz: %w", err)
		}
	} else if _, err := f.Seek(0, io.SeekStart); err != nil {
		return err
	}
	want := map[string]bool{GraphName: true, ConfigName: true, TokenizerName: true}
	tr := tar.NewReader(in)
	for {
		h, err := tr.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			return fmt.Errorf("bad archive: %w", err)
		}
		name := path.Base(h.Name)
		if h.Typeflag != tar.TypeReg || !want[name] {
			continue
		}
		want[name] = false
		out, err := os.Create(filepath.Join(dir, name))
		if err != nil {
			return err
		}
		n, err := io.Copy(out, io.LimitReader(tr, maxFile+1))
		if cerr := out.Close(); err == nil {
			err = cerr
		}
		if err != nil {
			return err
		}
		if n > maxFile {
			return fmt.Errorf("%s is larger than %d bytes", name, int64(maxFile))
		}
	}
	for _, name := range []string{GraphName, ConfigName} {
		if want[name] {
			return fmt.Errorf("archive has no %s", name)
		}
	}
	return nil
}

// verify checks the folder on disk: the graph's sha256 must match config.json's "sha256" when it has one.
// It fills GraphSHA256 and ModelVersion.
func (m *Model) verify() error {
	sum, err := fileSHA(filepath.Join(m.Dir, GraphName))
	if err != nil {
		return fmt.Errorf("model %s: %w", m.URL, err)
	}
	b, err := os.ReadFile(filepath.Join(m.Dir, ConfigName))
	if err != nil {
		return fmt.Errorf("model %s: %w", m.URL, err)
	}
	var c struct {
		Version string `json:"version"`
		SHA256  string `json:"sha256"`
	}
	if err := json.Unmarshal(b, &c); err != nil {
		return fmt.Errorf("model %s: %s: %w", m.URL, ConfigName, err)
	}
	if c.SHA256 != "" && !strings.EqualFold(c.SHA256, sum) {
		return fmt.Errorf("model %s: %w: %s sha256 %s, config.json says %s", m.URL, ErrChecksum, GraphName, sum, c.SHA256)
	}
	m.GraphSHA256, m.ModelVersion = sum, c.Version
	return nil
}

func fileSHA(p string) (string, error) {
	f, err := os.Open(p)
	if err != nil {
		return "", err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}
