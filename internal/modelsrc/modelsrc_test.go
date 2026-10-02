package modelsrc

import (
	"archive/tar"
	"bytes"
	"compress/gzip"
	"context"
	"crypto/md5"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
)

// fakeS3 serves GET and HEAD /<bucket>/<key> from a map, the way S3 does for path-style requests.
type fakeS3 struct {
	mu      sync.Mutex
	objects map[string][]byte // "bucket/key"
	deny    bool
	calls   []string
}

func (f *fakeS3) put(key string, body []byte) { f.mu.Lock(); f.objects[key] = body; f.mu.Unlock() }

func (f *fakeS3) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	f.mu.Lock()
	defer f.mu.Unlock()
	key := strings.TrimPrefix(r.URL.Path, "/")
	f.calls = append(f.calls, r.Method+" "+key)
	if f.deny {
		w.WriteHeader(http.StatusForbidden)
		if r.Method == http.MethodGet {
			fmt.Fprint(w, `<Error><Code>AccessDenied</Code><Message>Access Denied</Message></Error>`)
		}
		return
	}
	body, ok := f.objects[key]
	if !ok {
		w.WriteHeader(http.StatusNotFound)
		if r.Method == http.MethodGet {
			fmt.Fprint(w, `<Error><Code>NoSuchKey</Code><Message>missing</Message></Error>`)
		}
		return
	}
	sum := md5.Sum(body)
	w.Header().Set("ETag", `"`+hex.EncodeToString(sum[:])+`"`)
	w.Header().Set("Content-Length", fmt.Sprint(len(body)))
	if r.Method == http.MethodGet {
		_, _ = w.Write(body)
	}
}

func newFake(t *testing.T) (*fakeS3, Options) {
	t.Helper()
	t.Setenv("AWS_ACCESS_KEY_ID", "test")
	t.Setenv("AWS_SECRET_ACCESS_KEY", "test")
	t.Setenv("AWS_REGION", "us-east-1")
	t.Setenv("AWS_EC2_METADATA_DISABLED", "true")
	t.Setenv("AWS_CONFIG_FILE", filepath.Join(t.TempDir(), "none"))
	t.Setenv("AWS_SHARED_CREDENTIALS_FILE", filepath.Join(t.TempDir(), "none"))
	f := &fakeS3{objects: map[string][]byte{}}
	srv := httptest.NewServer(f)
	t.Cleanup(srv.Close)
	return f, Options{Endpoint: srv.URL, CacheDir: t.TempDir()}
}

func modelFiles(graph, version string) map[string][]byte {
	sum := sha256.Sum256([]byte(graph))
	return map[string][]byte{
		GraphName:     []byte(graph),
		ConfigName:    []byte(`{"version":"` + version + `","sha256":"` + hex.EncodeToString(sum[:]) + `"}`),
		TokenizerName: []byte(`{}`),
	}
}

func putFolder(f *fakeS3, prefix string, files map[string][]byte) {
	for name, body := range files {
		f.put(prefix+name, body)
	}
}

func tarball(t *testing.T, dir string, files map[string][]byte) []byte {
	t.Helper()
	var buf bytes.Buffer
	gz := gzip.NewWriter(&buf)
	tw := tar.NewWriter(gz)
	for name, body := range files {
		if err := tw.WriteHeader(&tar.Header{Name: dir + name, Mode: 0o644, Size: int64(len(body)), Typeflag: tar.TypeReg}); err != nil {
			t.Fatal(err)
		}
		_, _ = tw.Write(body)
	}
	_ = tw.Close()
	_ = gz.Close()
	return buf.Bytes()
}

func read(t *testing.T, p string) string {
	t.Helper()
	b, err := os.ReadFile(p)
	if err != nil {
		t.Fatal(err)
	}
	return string(b)
}

func TestFolderDownloadsAndThenStartsFromCache(t *testing.T) {
	f, opt := newFake(t)
	putFolder(f, "b/models/current/", modelFiles("graph-1", "1.0.0"))
	ctx := context.Background()
	s, err := New(ctx, "s3://b/models/current/", opt)
	if err != nil {
		t.Fatal(err)
	}
	m, warn, err := s.Sync(ctx)
	if err != nil || warn != nil {
		t.Fatal(err, warn)
	}
	if read(t, filepath.Join(m.Dir, GraphName)) != "graph-1" || m.ModelVersion != "1.0.0" || m.Offline {
		t.Fatalf("got %+v", m)
	}

	// A new process with the store unreachable starts from the cache.
	f.deny = true
	s2, err := New(ctx, "s3://b/models/current/", opt)
	if err != nil {
		t.Fatal(err)
	}
	m2, warn, err := s2.Sync(ctx)
	if err != nil || warn == nil || !m2.Offline || m2.Dir != m.Dir {
		t.Fatalf("offline start: %+v warn %v err %v", m2, warn, err)
	}
}

func TestArchiveWithModelFolderInside(t *testing.T) {
	f, opt := newFake(t)
	archive := tarball(t, "model/", modelFiles("graph-tar", "2.0.0"))
	f.put("b/models/model.tar.gz", archive)
	sum := sha256.Sum256(archive)
	opt.SHA256 = hex.EncodeToString(sum[:])
	ctx := context.Background()
	s, err := New(ctx, "s3://b/models/model.tar.gz", opt)
	if err != nil {
		t.Fatal(err)
	}
	m, _, err := s.Sync(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if read(t, filepath.Join(m.Dir, GraphName)) != "graph-tar" || m.SHA256 != opt.SHA256 {
		t.Fatalf("got %+v", m)
	}
}

func TestChecksumMismatchFailsLoudly(t *testing.T) {
	f, opt := newFake(t)
	ctx := context.Background()

	// The pinned sha256 does not match the archive.
	f.put("b/m.tar.gz", tarball(t, "", modelFiles("graph", "1")))
	opt.SHA256 = strings.Repeat("0", 64)
	s, _ := New(ctx, "s3://b/m.tar.gz", opt)
	if _, _, err := s.Sync(ctx); !errors.Is(err, ErrChecksum) || !strings.Contains(err.Error(), "configured") {
		t.Fatalf("pinned: want checksum error, got %v", err)
	}

	// config.json's sha256 does not match the graph.
	files := modelFiles("graph", "1")
	files[GraphName] = []byte("tampered")
	putFolder(f, "b/p/", files)
	opt.SHA256 = ""
	s, _ = New(ctx, "s3://b/p/", opt)
	if _, _, err := s.Sync(ctx); !errors.Is(err, ErrChecksum) || !strings.Contains(err.Error(), "config.json") {
		t.Fatalf("config sha: want checksum error, got %v", err)
	}
	if s.Current() != nil {
		t.Fatal("a bad download must not become current")
	}

	// With a good model cached, a restart over a bad upload serves the cache and warns.
	putFolder(f, "b/p/", modelFiles("graph", "1"))
	if _, _, err := s.Sync(ctx); err != nil {
		t.Fatal(err)
	}
	f.put("b/p/"+GraphName, []byte("tampered"))
	s, _ = New(ctx, "s3://b/p/", opt)
	m, warn, err := s.Sync(ctx)
	if err != nil || !errors.Is(warn, ErrChecksum) || !m.Offline || read(t, filepath.Join(m.Dir, GraphName)) != "graph" {
		t.Fatalf("restart over a bad upload: %+v warn %v err %v", m, warn, err)
	}
}

func TestEmptyPrefixAndAccessDenied(t *testing.T) {
	f, opt := newFake(t)
	ctx := context.Background()
	s, _ := New(ctx, "s3://b/models/current/", opt)
	if _, _, err := s.Sync(ctx); !errors.Is(err, ErrEmpty) {
		t.Fatalf("want ErrEmpty, got %v", err)
	}
	f.deny = true
	if _, _, err := s.Sync(ctx); err == nil || errors.Is(err, ErrEmpty) || !strings.Contains(err.Error(), "access denied") {
		t.Fatalf("want access denied, got %v", err)
	}
	for _, c := range f.calls {
		if !strings.HasPrefix(c, "HEAD b/models/current/") && !strings.HasPrefix(c, "GET b/models/current/") {
			t.Fatalf("call outside the object API on the prefix: %s", c)
		}
	}
}

func TestReloadSwapsAndKeepsPrevious(t *testing.T) {
	f, opt := newFake(t)
	putFolder(f, "b/models/current/", modelFiles("graph-1", "1.0.0"))
	ctx := context.Background()
	s, _ := New(ctx, "s3://b/models/current/", opt)
	first, _, err := s.Sync(ctx)
	if err != nil {
		t.Fatal(err)
	}

	var served *Model
	load := func(m *Model) error { served = m; return nil }
	if err := s.Reload(ctx, load); err != nil || served != nil {
		t.Fatalf("no change must not reload: %v %+v", err, served)
	}

	// Promote 2.0.0: it is downloaded to a new folder, loaded, and becomes current; 1.0.0 is kept.
	putFolder(f, "b/models/current/", modelFiles("graph-2", "2.0.0"))
	if err := s.Reload(ctx, load); err != nil {
		t.Fatal(err)
	}
	if served == nil || served.ModelVersion != "2.0.0" || s.Current() != served || s.Previous().Dir != first.Dir {
		t.Fatalf("swap: served %+v current %+v previous %+v", served, s.Current(), s.Previous())
	}
	if read(t, filepath.Join(first.Dir, GraphName)) != "graph-1" {
		t.Fatal("previous folder must stay for rollback")
	}

	// 3.0.0 fails to load: it is discarded and 2.0.0 stays current.
	putFolder(f, "b/models/current/", modelFiles("graph-3", "3.0.0"))
	var tried *Model
	err = s.Reload(ctx, func(m *Model) error { tried = m; return errors.New("bad graph") })
	if err == nil || s.Current() != served {
		t.Fatalf("failed load must keep the old model: %v", err)
	}
	if _, err := os.Stat(tried.Dir); !os.IsNotExist(err) {
		t.Fatal("the failed folder must be removed")
	}

	// A restart picks up the committed pointers.
	s2, _ := New(ctx, "s3://b/models/current/", opt)
	if s2.Current() == nil || s2.Current().Dir != served.Dir || s2.Previous().Dir != first.Dir {
		t.Fatalf("restart: %+v %+v", s2.Current(), s2.Previous())
	}
}

func TestSchemes(t *testing.T) {
	if !IsRemote("s3://b/k") || !IsRemote("gs://b/k") || IsRemote("/srv/model") || IsRemote("model") {
		t.Fatal("IsRemote")
	}
	if _, err := New(context.Background(), "gs://b/models/", Options{CacheDir: t.TempDir()}); err == nil || !strings.Contains(err.Error(), "not supported yet") {
		t.Fatalf("gs: %v", err)
	}
}

// A bucket outside the configured region: S3 answers 301 with x-amz-bucket-region, and the store
// retries there instead of needing s3:GetBucketLocation.
func TestFollowsBucketRegion(t *testing.T) {
	f, opt := newFake(t)
	putFolder(f, "b/models/current/", modelFiles("graph", "1"))
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !strings.Contains(r.Header.Get("Authorization"), "/eu-west-1/s3/") {
			w.Header().Set("X-Amz-Bucket-Region", "eu-west-1")
			w.WriteHeader(http.StatusMovedPermanently)
			return
		}
		f.ServeHTTP(w, r)
	}))
	t.Cleanup(srv.Close)
	opt.Endpoint = srv.URL
	ctx := context.Background()
	s, _ := New(ctx, "s3://b/models/current/", opt)
	if m, _, err := s.Sync(ctx); err != nil || m.ModelVersion != "1" {
		t.Fatalf("got %+v %v", m, err)
	}
}
