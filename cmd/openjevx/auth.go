package main

import (
	"crypto/rand"
	"crypto/subtle"
	"errors"
	"fmt"
	"io/fs"
	"log"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"strings"
)

// Two credentials guard the server:
//   - api_key: "Authorization: Bearer <key>" on POST /v1/systemone, the header jevx and the jev-cloud gate send.
//     On whenever the server listens beyond loopback; off on 127.0.0.1, ::1 and localhost unless one is set.
//   - password: HTTP Basic (any user name) on the dashboard, /stats, /metrics and /recipes. Always on.
// Each comes from the environment (OPENJEVX_API_KEY, OPENJEVX_PASSWORD), then openjevx.json. One that is needed
// but unset is generated once and kept beside openjevx.json (openjevx.api-key, openjevx.password; mode 0600), so
// every install has its own. The log names the file, never the value. allow_no_api_key / allow_no_password
// (OPENJEVX_ALLOW_NO_API_KEY=1, OPENJEVX_ALLOW_NO_PASSWORD=1) turn one off, e.g. behind a proxy that checks it.

const (
	minAPIKey = 16
	// oldDefaultPassword shipped in openjevx.json up to v0.5.6; it is public, so it counts as no password.
	oldDefaultPassword = "adminadmin"
)

// loopback reports whether a listen address only accepts connections from this machine.
func loopback(listen string) bool {
	host, _, err := net.SplitHostPort(listen)
	if err != nil {
		host = listen
	}
	if strings.EqualFold(host, "localhost") {
		return true
	}
	ip := net.ParseIP(strings.Trim(host, "[]"))
	return ip != nil && ip.IsLoopback()
}

// secret is one resolved credential: its value ("" = off) and where it came from, for the log.
type secret struct {
	value, from string
}

// resolveAuth fills in cfg.APIKey and cfg.Password following the rules above and says where each came from.
func resolveAuth(cfg *config) (key, password secret, err error) {
	dir := cfg.configDir
	if dir == "" {
		dir = "."
	}
	if v := os.Getenv("OPENJEVX_ALLOW_NO_API_KEY"); v != "" {
		cfg.AllowNoAPIKey = v == "1" || strings.EqualFold(v, "true")
	}
	if v := os.Getenv("OPENJEVX_ALLOW_NO_PASSWORD"); v != "" {
		cfg.AllowNoPassword = v == "1" || strings.EqualFold(v, "true")
	}

	key = secret{cfg.APIKey, "openjevx.json"}
	if v := os.Getenv("OPENJEVX_API_KEY"); v != "" {
		key = secret{v, "OPENJEVX_API_KEY"}
	}
	switch {
	case key.value != "":
		if len(key.value) < minAPIKey {
			return key, password, fmt.Errorf("api key from %s is %d characters; use at least %d", key.from, len(key.value), minAPIKey)
		}
	case cfg.AllowNoAPIKey:
		key.from = "allow_no_api_key"
	case loopback(cfg.Listen):
		key.from = "loopback " + cfg.Listen
	default:
		key, err = keptSecret(filepath.Join(dir, "openjevx.api-key"))
		if err != nil {
			return key, password, fmt.Errorf("listening on %s needs an API key: %w (or set OPENJEVX_API_KEY)", cfg.Listen, err)
		}
	}

	password = secret{cfg.Password, "openjevx.json"}
	if v := os.Getenv("OPENJEVX_PASSWORD"); v != "" {
		password = secret{v, "OPENJEVX_PASSWORD"}
	}
	if password.value == oldDefaultPassword {
		log.Printf("password: %s has the old published default %q; ignoring it", password.from, oldDefaultPassword)
		password.value = ""
	}
	switch {
	case password.value != "":
	case cfg.AllowNoPassword:
		password.from = "allow_no_password"
	default:
		password, err = keptSecret(filepath.Join(dir, "openjevx.password"))
		if err != nil {
			return key, password, fmt.Errorf("the dashboard needs a password: %w (or set OPENJEVX_PASSWORD)", err)
		}
	}
	cfg.APIKey, cfg.Password = key.value, password.value
	return key, password, nil
}

// keptSecret reads the secret in path, or creates it (mode 0600): 26 random letters and digits, 128 bits.
func keptSecret(path string) (secret, error) {
	b, err := os.ReadFile(path)
	if err == nil {
		if v := strings.TrimSpace(string(b)); v != "" {
			return secret{v, path}, nil
		}
		return secret{}, fmt.Errorf("%s is empty", path)
	}
	if !errors.Is(err, fs.ErrNotExist) {
		return secret{}, err
	}
	v := rand.Text()
	f, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o600)
	if err != nil {
		return secret{}, fmt.Errorf("cannot create %s: %w", path, err)
	}
	if _, err := f.WriteString(v + "\n"); err != nil {
		f.Close()
		return secret{}, err
	}
	if err := f.Close(); err != nil {
		return secret{}, err
	}
	return secret{v, path + " (generated)"}, nil
}

// requireKey lets a request through only with "Authorization: Bearer <key>"; an empty key lets everything through.
// The 401 matches the jev-cloud gate's, so clients see one answer whichever of the two checks them.
func requireKey(key string, next http.HandlerFunc) http.HandlerFunc {
	if key == "" {
		return next
	}
	return func(w http.ResponseWriter, r *http.Request) {
		scheme, got, _ := strings.Cut(r.Header.Get("Authorization"), " ")
		if !strings.EqualFold(scheme, "Bearer") || subtle.ConstantTimeCompare([]byte(strings.TrimSpace(got)), []byte(key)) != 1 {
			w.Header().Set("WWW-Authenticate", "Bearer")
			http.Error(w, `{"error":"missing or wrong API key"}`, http.StatusUnauthorized)
			return
		}
		next(w, r)
	}
}
