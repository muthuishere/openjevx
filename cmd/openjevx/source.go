package main

import (
	"context"
	"errors"
	"fmt"
	"log"
	"strings"
	"time"

	ort "github.com/yalue/onnxruntime_go"

	"github.com/muthuishere/openjevx/internal/modelsrc"
)

// modelInfo is what /health reports about a model.
type modelInfo struct {
	Version       string    `json:"version"`
	SHA256        string    `json:"sha256"`
	Source        string    `json:"source"`                   // the s3:// URL, or the local path
	Fallback      bool      `json:"fallback"`                 // the local fallback, served while the remote is empty
	Offline       bool      `json:"offline,omitempty"`        // from the cache; the store could not be reached
	RemoteVersion string    `json:"remote_version,omitempty"` // the store's ETag(s)
	Dir           string    `json:"dir"`
	LoadedAt      time.Time `json:"loaded_at"`
}

// served is one loaded model and its session; the server swaps it whole on a reload.
type served struct {
	m       *loadedModel
	session *ort.DynamicAdvancedSession
	info    modelInfo
}

// openServed loads a model folder and opens its session. The first call picks the device (cfg.Device
// "auto" becomes "cpu" or "gpu"); a reload passes that choice back in.
func openServed(cfg *config, path string) (*served, error) {
	m, err := loadModel(path)
	if err != nil {
		return nil, err
	}
	tokSrc := m.TokFile
	if tokSrc == "" {
		tokSrc = "embedded"
	}
	log.Printf("model %s version %s sha256 %s temperatures choice=%g score=%g noul=%g max_len %d head_max %d tokenizer %s",
		m.Path, m.Version, m.SHA256[:12], m.Params.Temperature[0], m.Params.Temperature[1], m.Params.Temperature[2],
		m.Params.MaxLen, m.Params.HeadMax, tokSrc)
	probe, err := probeItems(m.Tokenizer, m.Params)
	if err != nil {
		return nil, err
	}
	session, used, err := openSession(*cfg, m.Graph, probe)
	m.Graph = nil // the session holds its own copy
	if err != nil {
		return nil, err
	}
	cfg.Device = used
	return &served{m: m, session: session, info: modelInfo{
		Version: m.Version, SHA256: m.SHA256, Source: path, Dir: path, LoadedAt: time.Now().UTC(),
	}}, nil
}

// reloadEvery is the "model_reload" interval: a Go duration ("5m"); "" or "0" = off.
func reloadEvery(cfg config) (time.Duration, error) {
	if cfg.ModelReload == "" || cfg.ModelReload == "0" {
		return 0, nil
	}
	d, err := time.ParseDuration(cfg.ModelReload)
	if err != nil || d < 0 {
		return 0, fmt.Errorf("model_reload %q: want a duration such as 5m, or 0 for off", cfg.ModelReload)
	}
	if d < 10*time.Second {
		return 0, fmt.Errorf("model_reload %q: at least 10s", cfg.ModelReload)
	}
	return d, nil
}

// fallbackPoll is how often an empty remote is checked when model_reload is off.
const fallbackPoll = time.Minute

// startModel loads the configured model. A local path is loaded as before. An object-store URL is synced
// into the cache first; if it holds no model yet, the local fallback (model_fallback, else the model folder
// next to the executable) is served instead and src is returned so the caller polls for the real one.
func startModel(ctx context.Context, cfg *config, exeDir string) (*served, *modelsrc.Source, error) {
	if !modelsrc.IsRemote(cfg.Model) {
		if strings.Contains(cfg.Model, "://") {
			return nil, nil, fmt.Errorf("model %s: only s3:// URLs and local paths are supported", cfg.Model)
		}
		path, err := resolveModel(cfg.Model, exeDir)
		if err != nil {
			return nil, nil, err
		}
		s, err := openServed(cfg, path)
		return s, nil, err
	}
	src, err := modelsrc.New(ctx, cfg.Model, modelsrc.Options{SHA256: cfg.ModelSHA256, CacheDir: cfg.ModelCache})
	if err != nil {
		return nil, nil, err
	}
	m, warn, err := src.Sync(ctx)
	if errors.Is(err, modelsrc.ErrEmpty) {
		path, ferr := resolveModel(cfg.ModelFallback, exeDir)
		if ferr != nil {
			return nil, nil, fmt.Errorf("%v, and no fallback model: %v", err, ferr)
		}
		log.Printf("%v; serving the fallback model %s until one appears", err, path)
		s, err := openServed(cfg, path)
		if err != nil {
			return nil, nil, err
		}
		s.info.Fallback = true
		return s, src, nil
	}
	if err != nil {
		return nil, nil, err
	}
	if warn != nil {
		log.Printf("WARNING: %v; starting from the cached model %s", warn, m.Dir)
	}
	s, err := openServed(cfg, m.Dir)
	if err != nil {
		return nil, nil, err
	}
	s.info.Source, s.info.RemoteVersion, s.info.Offline = cfg.Model, m.Version, m.Offline
	return s, src, nil
}
