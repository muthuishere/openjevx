package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/muthuishere/openjevx/internal/bpe"
	"github.com/muthuishere/openjevx/internal/decide"
)

// A model is a folder: the graph, its config (calibration temperatures, lengths, ids) and its tokenizer.
//
//	<dir>/openjevx.w8.onnx
//	<dir>/config.json
//	<dir>/tokenizer.json   (optional; the embedded tokenizer otherwise)
//
// A bare .onnx file still works and gets decide.Defaults().
const (
	graphName     = "openjevx.w8.onnx"
	modelConfName = "config.json"
	tokenizerName = "tokenizer.json"
)

type modelConfig struct {
	Name        string `json:"name"`
	Version     string `json:"version"`
	Temperature *struct {
		Choice float64 `json:"choice"`
		Score  float64 `json:"score"`
		Noul   float64 `json:"noul"`
	} `json:"temperature"`
	MaxLen     int            `json:"max_len"`
	HeadMax    int            `json:"head_max"`
	SpecialIDs map[string]int `json:"special_ids"`
	SHA256     string         `json:"sha256"`
}

type loadedModel struct {
	Path      string // the folder, or the .onnx file
	Graph     []byte
	Name      string
	Version   string
	SHA256    string
	Params    decide.Params
	Tokenizer *bpe.Tokenizer
	TokFile   string // "" = embedded
}

// resolveModel picks the model path: "model" from openjevx.json, else a folder next to the executable
// (model/, models/openjevx/), else openjevx.w8.onnx next to the executable.
func resolveModel(configured, exeDir string) (string, error) {
	if configured != "" {
		if _, err := os.Stat(configured); err != nil {
			return "", fmt.Errorf("model %s: not found (openjevx.json \"model\" must be a model folder or a .onnx file)", configured)
		}
		return configured, nil
	}
	tried := []string{filepath.Join(exeDir, "model"), filepath.Join(exeDir, "models", "openjevx"), filepath.Join(exeDir, graphName)}
	for _, p := range tried {
		if _, err := os.Stat(p); err == nil {
			return p, nil
		}
	}
	return "", fmt.Errorf("no model found: set \"model\" in openjevx.json, or put a model folder at %s", strings.Join(tried, ", "))
}

func loadModel(path string) (*loadedModel, error) {
	info, err := os.Stat(path)
	if err != nil {
		return nil, fmt.Errorf("model %s: not found", path)
	}
	m := &loadedModel{Path: path, Name: "openjevx", Version: "unknown", Params: decide.Defaults()}
	graph := path
	if info.IsDir() {
		graph = filepath.Join(path, graphName)
		b, err := os.ReadFile(filepath.Join(path, modelConfName))
		if err != nil {
			return nil, fmt.Errorf("model folder %s: missing %s", path, modelConfName)
		}
		var c modelConfig
		if err := json.Unmarshal(b, &c); err != nil {
			return nil, fmt.Errorf("model folder %s: %s: %w", path, modelConfName, err)
		}
		if c.Temperature == nil {
			return nil, fmt.Errorf("model folder %s: %s has no \"temperature\"", path, modelConfName)
		}
		t := [3]float64{c.Temperature.Choice, c.Temperature.Score, c.Temperature.Noul}
		for _, v := range t {
			if v <= 0 {
				return nil, fmt.Errorf("model folder %s: temperatures must be > 0, got %v", path, t)
			}
		}
		m.Params.Temperature = t
		if c.MaxLen > 0 {
			m.Params.MaxLen = c.MaxLen
		}
		if c.HeadMax > 0 {
			m.Params.HeadMax = c.HeadMax
		}
		for key, dst := range map[string]*int{"cls": &m.Params.CLS, "sep": &m.Params.SEP, "pad": &m.Params.PAD, "mask": &m.Params.MASK} {
			if id, ok := c.SpecialIDs[key]; ok {
				*dst = id
			}
		}
		if c.Name != "" {
			m.Name = c.Name
		}
		if c.Version != "" {
			m.Version = c.Version
		}
		m.SHA256 = strings.ToLower(c.SHA256)
		if tb, err := os.ReadFile(filepath.Join(path, tokenizerName)); err == nil {
			if m.Tokenizer, err = bpe.Parse(tb); err != nil {
				return nil, fmt.Errorf("model folder %s: %s: %w", path, tokenizerName, err)
			}
			m.TokFile = filepath.Join(path, tokenizerName)
		} else if !errors.Is(err, os.ErrNotExist) {
			return nil, err
		}
	}
	if m.Graph, err = os.ReadFile(graph); err != nil {
		return nil, fmt.Errorf("model %s: cannot read graph %s: %w", path, graph, err)
	}
	sum := sha256.Sum256(m.Graph)
	got := hex.EncodeToString(sum[:])
	if m.SHA256 != "" && m.SHA256 != got {
		return nil, fmt.Errorf("model %s: %s sha256 %s does not match config.json %s", path, graphName, got, m.SHA256)
	}
	m.SHA256 = got
	if m.Tokenizer == nil {
		if m.Tokenizer, err = bpe.Load(); err != nil {
			return nil, err
		}
	}
	return m, nil
}
