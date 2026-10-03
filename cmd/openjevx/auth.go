package main

import (
	"crypto/rand"
	"crypto/sha256"
	"crypto/subtle"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"log"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"slices"
	"strings"
)

// Two credentials guard the server:
//   - api_key: "Authorization: Bearer <key>" on POST /v1/systemone, the header jevx and the jev-cloud gate send.
//     On whenever the server listens beyond loopback; off on 127.0.0.1, ::1 and localhost unless one is set.
//   - password: HTTP Basic (any user name) on the dashboard, /stats, /metrics and /recipes. Always on.
// Each comes from the environment (OPENJEVX_API_KEY, OPENJEVX_PASSWORD), then openjevx.json. One that is needed
// but unset is generated once and kept in a file (openjevx.api-key, openjevx.password; mode 0600), so every install
// has its own: beside openjevx.json (or the executable when there is none), else in $OPENJEVX_DATA, else the working
// folder, whichever is writable (a config mounted read-only still starts). The value is printed once, and only to a
// terminal: logs (Docker, systemd, ECS) get the file and a fingerprint, never the secret. allow_no_api_key / allow_no_password
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

// secret is one resolved credential: its value ("" = off), where it came from, and whether this start created it.
type secret struct {
	value, from string
	generated   bool
}

// resolveAuth fills in cfg.APIKey and cfg.Password following the rules above and says where each came from.
func resolveAuth(cfg *config) (key, password secret, err error) {
	dirs, err := secretDirs(cfg.configDir)
	if err != nil {
		return key, password, err
	}
	if v := os.Getenv("OPENJEVX_ALLOW_NO_API_KEY"); v != "" {
		cfg.AllowNoAPIKey = v == "1" || strings.EqualFold(v, "true")
	}
	if v := os.Getenv("OPENJEVX_ALLOW_NO_PASSWORD"); v != "" {
		cfg.AllowNoPassword = v == "1" || strings.EqualFold(v, "true")
	}

	key = secret{value: cfg.APIKey, from: "openjevx.json"}
	if v := os.Getenv("OPENJEVX_API_KEY"); v != "" {
		key = secret{value: v, from: "OPENJEVX_API_KEY"}
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
		key, err = keptSecret("openjevx.api-key", dirs)
		if err != nil {
			return key, password, fmt.Errorf("listening on %s needs an API key: %w (or set OPENJEVX_API_KEY)", cfg.Listen, err)
		}
	}

	password = secret{value: cfg.Password, from: "openjevx.json"}
	if v := os.Getenv("OPENJEVX_PASSWORD"); v != "" {
		password = secret{value: v, from: "OPENJEVX_PASSWORD"}
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
		password, err = keptSecret("openjevx.password", dirs)
		if err != nil {
			return key, password, fmt.Errorf("the dashboard needs a password: %w (or set OPENJEVX_PASSWORD)", err)
		}
	}
	cfg.APIKey, cfg.Password = key.value, password.value
	return key, password, nil
}

// secretDirs is where credential files may live, in order: the config's folder (or the executable's), $OPENJEVX_DATA,
// then the working folder.
func secretDirs(configDir string) ([]string, error) {
	first := configDir
	if first == "" {
		exe, err := os.Executable()
		if err != nil {
			return nil, err
		}
		first = filepath.Dir(exe)
	}
	var dirs []string
	for _, d := range []string{first, os.Getenv("OPENJEVX_DATA"), "."} {
		if d == "" {
			continue
		}
		if abs, err := filepath.Abs(d); err == nil {
			d = abs
		}
		if !slices.Contains(dirs, d) {
			dirs = append(dirs, d)
		}
	}
	return dirs, nil
}

// keptSecret reads the secret file name from the first of dirs that has it, or creates it (mode 0600, 26 random
// letters and digits, 128 bits) in the first of dirs that is writable.
func keptSecret(name string, dirs []string) (secret, error) {
	for _, d := range dirs {
		path := filepath.Join(d, name)
		b, err := os.ReadFile(path)
		if err == nil {
			if v := strings.TrimSpace(string(b)); v != "" {
				return secret{value: v, from: path}, nil
			}
			return secret{}, fmt.Errorf("%s is empty", path)
		}
		if !errors.Is(err, fs.ErrNotExist) {
			return secret{}, err
		}
	}
	v := rand.Text()
	var tried []string
	for _, d := range dirs {
		path := filepath.Join(d, name)
		f, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o600)
		if err != nil {
			tried = append(tried, fmt.Sprintf("%s (%v)", d, errors.Unwrap(err)))
			continue
		}
		if _, err := f.WriteString(v + "\n"); err != nil {
			f.Close()
			return secret{}, err
		}
		if err := f.Close(); err != nil {
			return secret{}, err
		}
		return secret{value: v, from: path, generated: true}, nil
	}
	return secret{}, fmt.Errorf("cannot create %s in any of: %s; set OPENJEVX_DATA to a writable folder", name, strings.Join(tried, ", "))
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

// printNew reports a credential this start created. Only a terminal sees the value, once; anything else (a log
// driver, journald, CloudWatch) gets the file and a fingerprint, so the secret never lands in logs.
func printNew(w io.Writer, terminal bool, key, password secret) {
	for _, c := range []struct {
		s          secret
		what, hint string
	}{
		{password, "dashboard password", ""},
		{key, "API key for /v1/systemone", "Send it as: Authorization: Bearer <key>\n"},
	} {
		if !c.s.generated {
			continue
		}
		if terminal {
			fmt.Fprintf(w, "\nNew %s (shown once; kept in %s):\n  %s\n%s\n", c.what, c.s.from, c.s.value, c.hint)
		} else {
			fmt.Fprintf(w, "new %s generated into %s (sha256 ...%s); not printed: stderr is not a terminal\n", c.what, c.s.from, fingerprint(c.s.value))
		}
	}
}

// fingerprint is the last 4 hex digits of the value's sha256: enough to tell two secrets apart, nothing to guess from.
func fingerprint(v string) string {
	sum := sha256.Sum256([]byte(v))
	h := hex.EncodeToString(sum[:])
	return h[len(h)-4:]
}

// isTerminal reports whether f is a character device (a terminal), not a pipe or a file.
func isTerminal(f *os.File) bool {
	st, err := f.Stat()
	return err == nil && st.Mode()&os.ModeCharDevice != 0
}
