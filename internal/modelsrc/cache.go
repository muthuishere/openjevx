package modelsrc

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

// ErrEmpty means the remote location holds no model yet (a fresh deploy before the first promote).
var ErrEmpty = errors.New("no model there yet")

// Model is one downloaded, verified model folder in the cache. It is also the manifest (source.json)
// written next to the files.
type Model struct {
	URL          string    `json:"url"`
	Version      string    `json:"version"`       // the store's ETag(s) it was downloaded at
	SHA256       string    `json:"sha256"`        // of the .tar.gz, or of openjevx.w8.onnx for a folder
	GraphSHA256  string    `json:"graph_sha256"`  // of openjevx.w8.onnx
	ModelVersion string    `json:"model_version"` // config.json "version"
	FetchedAt    time.Time `json:"fetched_at"`
	Dir          string    `json:"-"`
	Offline      bool      `json:"-"` // served from the cache because the store could not be reached
}

const manifestName = "source.json"

// Source is a remote model and its local cache:
//
//	<cache>/<hash of url>/v-<hash of version>/   one folder per downloaded version
//	<cache>/<hash of url>/current, previous      the folder names in use and kept for rollback
type Source struct {
	remote *remote
	opt    Options
	dir    string

	mu       sync.Mutex
	current  *Model
	previous *Model
}

// New opens a model URL. It does not contact the store yet.
func New(ctx context.Context, rawURL string, opt Options) (*Source, error) {
	r, err := openRemote(ctx, rawURL, opt)
	if err != nil {
		return nil, err
	}
	root := opt.CacheDir
	if root == "" {
		if root, err = DefaultCacheDir(); err != nil {
			return nil, err
		}
	}
	sum := sha256.Sum256([]byte(rawURL))
	s := &Source{remote: r, opt: opt, dir: filepath.Join(root, hex.EncodeToString(sum[:6]))}
	if err := os.MkdirAll(s.dir, 0o755); err != nil {
		return nil, fmt.Errorf("model cache %s: %w (set model_cache / OPENJEVX_MODEL_CACHE to a writable folder)", s.dir, err)
	}
	s.current = s.cached("current")
	s.previous = s.cached("previous")
	return s, nil
}

// DefaultCacheDir is <user cache dir>/openjevx/models, or <tmp>/openjevx-models when there is no home.
func DefaultCacheDir() (string, error) {
	if d, err := os.UserCacheDir(); err == nil {
		return filepath.Join(d, "openjevx", "models"), nil
	}
	return filepath.Join(os.TempDir(), "openjevx-models"), nil
}

// Current is the model in use (nil before the first successful Sync); Previous is the one kept for rollback.
func (s *Source) Current() *Model  { s.mu.Lock(); defer s.mu.Unlock(); return s.current }
func (s *Source) Previous() *Model { s.mu.Lock(); defer s.mu.Unlock(); return s.previous }

// cached reads the folder a pointer file names, if it is still a valid model for this URL and pin.
func (s *Source) cached(pointer string) *Model {
	name, err := os.ReadFile(filepath.Join(s.dir, pointer))
	if err != nil {
		return nil
	}
	dir := filepath.Join(s.dir, strings.TrimSpace(string(name)))
	b, err := os.ReadFile(filepath.Join(dir, manifestName))
	if err != nil {
		return nil
	}
	var m Model
	if json.Unmarshal(b, &m) != nil || m.URL != s.remote.url {
		return nil
	}
	if s.opt.SHA256 != "" && !strings.EqualFold(s.opt.SHA256, m.SHA256) {
		return nil
	}
	m.Dir = dir
	want := m.GraphSHA256
	if m.verify() != nil || m.GraphSHA256 != want {
		return nil
	}
	return &m
}

// Sync makes the cache hold the remote's current model and returns it. If that fails (store unreachable,
// access denied, a bad upload) and the cache holds a valid model, it returns that with Offline set and the
// error in warn, so a bad upload never takes a running deployment down; with no valid cache it fails.
// It returns ErrEmpty when the store is reachable but holds no model yet and nothing is cached.
func (s *Source) Sync(ctx context.Context) (m *Model, warn error, err error) {
	next, err := s.Check(ctx)
	cur := s.Current()
	switch {
	case errors.Is(err, ErrEmpty) && cur != nil:
		return cur, err, nil
	case errors.Is(err, ErrEmpty):
		return nil, nil, err
	case err != nil && cur != nil:
		off := *cur
		off.Offline = true
		return &off, err, nil
	case err != nil:
		return nil, nil, err
	case next == nil:
		return cur, nil, nil
	}
	if err := s.Commit(next); err != nil {
		return nil, nil, err
	}
	return next, nil, nil
}

// Check downloads the remote's model into a new cache folder if its version differs from Current, and
// verifies it. It returns nil when nothing changed. The new folder is not in use until Commit.
func (s *Source) Check(ctx context.Context) (*Model, error) {
	v, err := s.remote.version(ctx)
	if errors.Is(err, ErrNotFound) {
		return nil, fmt.Errorf("model %s: %w", s.remote.url, ErrEmpty)
	}
	if err != nil {
		return nil, err
	}
	if cur := s.Current(); cur != nil && cur.Version == v {
		return nil, nil
	}
	tmp, err := os.MkdirTemp(s.dir, ".download-")
	if err != nil {
		return nil, err
	}
	m, err := s.remote.download(ctx, tmp, s.opt)
	if err != nil {
		_ = os.RemoveAll(tmp)
		if errors.Is(err, ErrNotFound) {
			return nil, fmt.Errorf("model %s: %w", s.remote.url, ErrEmpty)
		}
		return nil, err
	}
	m.FetchedAt = time.Now().UTC()
	vs := sha256.Sum256([]byte(m.Version))
	final := filepath.Join(s.dir, fmt.Sprintf("v-%s-%d", hex.EncodeToString(vs[:6]), m.FetchedAt.UnixNano()))
	b, _ := json.MarshalIndent(m, "", "  ")
	if err := os.WriteFile(filepath.Join(tmp, manifestName), b, 0o644); err != nil {
		_ = os.RemoveAll(tmp)
		return nil, err
	}
	if err := os.Rename(tmp, final); err != nil {
		_ = os.RemoveAll(tmp)
		return nil, err
	}
	m.Dir = final
	return m, nil
}

// Commit makes m current, keeps the old current as previous, and deletes every other cached version.
func (s *Source) Commit(m *Model) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	prev := s.current
	if prev != nil && prev.Dir == m.Dir {
		return nil
	}
	if err := s.point("current", m); err != nil {
		return err
	}
	if prev != nil {
		if err := s.point("previous", prev); err != nil {
			return err
		}
	}
	s.current, s.previous = m, prev
	keep := map[string]bool{"current": true, "previous": true, filepath.Base(m.Dir): true}
	if prev != nil {
		keep[filepath.Base(prev.Dir)] = true
	}
	entries, _ := os.ReadDir(s.dir)
	for _, e := range entries {
		if !keep[e.Name()] && !strings.HasPrefix(e.Name(), ".download-") {
			_ = os.RemoveAll(filepath.Join(s.dir, e.Name()))
		}
	}
	return nil
}

// Discard removes a folder Check downloaded that will not be used (it failed to load).
func (s *Source) Discard(m *Model) { _ = os.RemoveAll(m.Dir) }

// point writes a pointer file atomically (write, then rename).
func (s *Source) point(pointer string, m *Model) error {
	tmp := filepath.Join(s.dir, pointer+".tmp")
	if err := os.WriteFile(tmp, []byte(filepath.Base(m.Dir)+"\n"), 0o644); err != nil {
		return err
	}
	return os.Rename(tmp, filepath.Join(s.dir, pointer))
}

// Reload checks the store once. On a new version it calls load with the verified folder; if load
// succeeds the folder becomes current, otherwise it is discarded and the old model stays.
func (s *Source) Reload(ctx context.Context, load func(*Model) error) error {
	m, err := s.Check(ctx)
	if err != nil || m == nil {
		return err
	}
	if err := load(m); err != nil {
		s.Discard(m)
		return fmt.Errorf("model %s version %s: %w", s.remote.url, m.Version, err)
	}
	return s.Commit(m)
}
