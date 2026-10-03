package main

import (
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestLoopback(t *testing.T) {
	for listen, want := range map[string]bool{
		"127.0.0.1:21118": true, "127.0.0.2:1": true, "[::1]:21118": true, "localhost:21118": true,
		"0.0.0.0:21118": false, ":21118": false, "[::]:21118": false, "10.0.0.5:21118": false, "myhost:21118": false,
	} {
		if got := loopback(listen); got != want {
			t.Errorf("loopback(%q) = %v, want %v", listen, got, want)
		}
	}
}

func clearAuthEnv(t *testing.T) {
	for _, k := range []string{"OPENJEVX_API_KEY", "OPENJEVX_PASSWORD", "OPENJEVX_ALLOW_NO_API_KEY", "OPENJEVX_ALLOW_NO_PASSWORD"} {
		t.Setenv(k, "")
	}
}

// Beyond loopback with no key: one is generated beside openjevx.json, mode 0600, and the same one comes back next start.
func TestAPIKeyGeneratedWhenPublic(t *testing.T) {
	clearAuthEnv(t)
	dir := t.TempDir()
	cfg := config{Listen: "0.0.0.0:21118", configDir: dir}
	key, pw, err := resolveAuth(&cfg)
	if err != nil {
		t.Fatal(err)
	}
	if len(cfg.APIKey) < minAPIKey || len(cfg.Password) < 12 || key.value != cfg.APIKey || pw.value != cfg.Password {
		t.Fatalf("key %q password %q", cfg.APIKey, cfg.Password)
	}
	path := filepath.Join(dir, "openjevx.api-key")
	if !strings.HasPrefix(key.from, path) || strings.Contains(key.from, cfg.APIKey) {
		t.Fatalf("from %q: want the file, never the value", key.from)
	}
	st, err := os.Stat(path)
	if err != nil || st.Mode().Perm() != 0o600 {
		t.Fatalf("%s: %v %v", path, st, err)
	}
	again := config{Listen: "0.0.0.0:21118", configDir: dir}
	if _, _, err := resolveAuth(&again); err != nil || again.APIKey != cfg.APIKey || again.Password != cfg.Password {
		t.Fatalf("second start: key %q password %q err %v; want the kept ones", again.APIKey, again.Password, err)
	}
}

// On loopback the key is off unless set; the dashboard password is still required.
func TestAPIKeyOffOnLoopback(t *testing.T) {
	clearAuthEnv(t)
	dir := t.TempDir()
	cfg := config{Listen: "127.0.0.1:21118", configDir: dir}
	if _, _, err := resolveAuth(&cfg); err != nil {
		t.Fatal(err)
	}
	if cfg.APIKey != "" || cfg.Password == "" {
		t.Fatalf("key %q password %q", cfg.APIKey, cfg.Password)
	}
	if _, err := os.Stat(filepath.Join(dir, "openjevx.api-key")); err == nil {
		t.Fatal("no key file on loopback")
	}
}

func TestAuthSettings(t *testing.T) {
	clearAuthEnv(t)
	dir := t.TempDir()
	// The environment beats openjevx.json.
	t.Setenv("OPENJEVX_API_KEY", "from-the-environment-0123")
	cfg := config{Listen: "127.0.0.1:21118", APIKey: "from-the-config-file-0123", Password: "a-real-password", configDir: dir}
	key, _, err := resolveAuth(&cfg)
	if err != nil || cfg.APIKey != "from-the-environment-0123" || key.from != "OPENJEVX_API_KEY" {
		t.Fatalf("key %q from %q err %v", cfg.APIKey, key.from, err)
	}
	// A short key is refused.
	t.Setenv("OPENJEVX_API_KEY", "short")
	if _, _, err := resolveAuth(&config{Listen: "127.0.0.1:1", configDir: dir}); err == nil {
		t.Fatal("a 5-character key was accepted")
	}
	// Turned off on purpose, even when public.
	t.Setenv("OPENJEVX_API_KEY", "")
	t.Setenv("OPENJEVX_ALLOW_NO_API_KEY", "1")
	t.Setenv("OPENJEVX_ALLOW_NO_PASSWORD", "1")
	cfg = config{Listen: "0.0.0.0:21118", configDir: dir}
	if _, _, err := resolveAuth(&cfg); err != nil || cfg.APIKey != "" || cfg.Password != "" {
		t.Fatalf("allow_no_*: key %q password %q err %v", cfg.APIKey, cfg.Password, err)
	}
	// The old published default counts as no password.
	clearAuthEnv(t)
	cfg = config{Listen: "127.0.0.1:21118", Password: oldDefaultPassword, configDir: dir}
	if _, _, err := resolveAuth(&cfg); err != nil || cfg.Password == oldDefaultPassword || cfg.Password == "" {
		t.Fatalf("password %q err %v", cfg.Password, err)
	}
}

// Public with no key and nowhere to keep one: refuse to start rather than serve an open API.
func TestAPIKeyUnwritable(t *testing.T) {
	clearAuthEnv(t)
	cfg := config{Listen: "0.0.0.0:21118", configDir: filepath.Join(t.TempDir(), "missing")}
	if _, _, err := resolveAuth(&cfg); err == nil || !strings.Contains(err.Error(), "OPENJEVX_API_KEY") {
		t.Fatalf("err %v", err)
	}
}

func TestRequireKey(t *testing.T) {
	ok := func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(http.StatusOK) }
	h := requireKey("0123456789abcdef-key", ok)
	for header, want := range map[string]int{
		"":                            http.StatusUnauthorized,
		"Bearer wrong":                http.StatusUnauthorized,
		"Basic 0123456789abcdef-key":  http.StatusUnauthorized,
		"0123456789abcdef-key":        http.StatusUnauthorized,
		"Bearer 0123456789abcdef-key": http.StatusOK,
		"bearer 0123456789abcdef-key": http.StatusOK,
	} {
		r := httptest.NewRequest(http.MethodPost, "/v1/systemone", nil)
		if header != "" {
			r.Header.Set("Authorization", header)
		}
		rec := httptest.NewRecorder()
		h(rec, r)
		if rec.Code != want {
			t.Errorf("Authorization %q: %d, want %d", header, rec.Code, want)
		}
		if want == http.StatusUnauthorized && (rec.Header().Get("WWW-Authenticate") != "Bearer" || !strings.Contains(rec.Body.String(), "missing or wrong API key")) {
			t.Errorf("Authorization %q: 401 should match the jev-cloud gate: %v %q", header, rec.Header(), rec.Body)
		}
	}
	rec := httptest.NewRecorder()
	requireKey("", ok)(rec, httptest.NewRequest(http.MethodPost, "/v1/systemone", nil))
	if rec.Code != http.StatusOK {
		t.Fatalf("no key set: %d", rec.Code)
	}
}

func TestDashboardPassword(t *testing.T) {
	guard := protected("a-real-password")
	for _, c := range []struct {
		user, pass, query string
		want              int
	}{
		{"", "", "", http.StatusUnauthorized},
		{"admin", "wrong", "", http.StatusUnauthorized},
		{"anyone", "a-real-password", "", http.StatusOK},
		{"", "", "?password=a-real-password", http.StatusOK},
	} {
		r := httptest.NewRequest(http.MethodGet, "/stats"+c.query, nil)
		if c.user != "" {
			r.SetBasicAuth(c.user, c.pass)
		}
		rec := httptest.NewRecorder()
		guard(rec, r, func() { rec.WriteHeader(http.StatusOK) })
		if rec.Code != c.want {
			t.Errorf("%+v: %d", c, rec.Code)
		}
	}
}
