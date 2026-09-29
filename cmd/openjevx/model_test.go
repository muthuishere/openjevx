package main

import (
	"crypto/sha256"
	"encoding/hex"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/muthuishere/openjevx/internal/decide"
)

func writeFile(t *testing.T, path, body string) {
	t.Helper()
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
}

func TestModelFolderUsesItsConfigTemperatures(t *testing.T) {
	dir := t.TempDir()
	writeFile(t, filepath.Join(dir, graphName), "graph")
	sum := sha256.Sum256([]byte("graph"))
	writeFile(t, filepath.Join(dir, modelConfName), `{"name":"openjevx","version":"9.9.9",
		"temperature":{"choice":1.5,"score":2.5,"noul":3.5},"max_len":512,"head_max":192,
		"special_ids":{"cls":1,"sep":2,"pad":3,"mask":4},"sha256":"`+hex.EncodeToString(sum[:])+`"}`)
	m, err := loadModel(dir)
	if err != nil {
		t.Fatal(err)
	}
	want := decide.Params{Temperature: [3]float64{1.5, 2.5, 3.5}, MaxLen: 512, HeadMax: 192, CLS: 1, SEP: 2, PAD: 3, MASK: 4}
	if m.Params != want || m.Version != "9.9.9" || m.TokFile != "" || m.Tokenizer == nil {
		t.Fatalf("got %+v version %s tok %q", m.Params, m.Version, m.TokFile)
	}
}

func TestModelFolderShaMismatchFails(t *testing.T) {
	dir := t.TempDir()
	writeFile(t, filepath.Join(dir, graphName), "graph")
	writeFile(t, filepath.Join(dir, modelConfName), `{"temperature":{"choice":1,"score":1,"noul":1},"sha256":"00"}`)
	if _, err := loadModel(dir); err == nil || !strings.Contains(err.Error(), "does not match") {
		t.Fatalf("want sha mismatch, got %v", err)
	}
}

func TestOnnxFileUsesDefaults(t *testing.T) {
	file := filepath.Join(t.TempDir(), "old.onnx")
	writeFile(t, file, "graph")
	m, err := loadModel(file)
	if err != nil {
		t.Fatal(err)
	}
	if m.Params != decide.Defaults() {
		t.Fatalf("got %+v", m.Params)
	}
}

func TestMissingModelErrorsClearly(t *testing.T) {
	missing := filepath.Join(t.TempDir(), "nope")
	if _, err := resolveModel(missing, t.TempDir()); err == nil || !strings.Contains(err.Error(), "not found") {
		t.Fatalf("got %v", err)
	}
	if _, err := resolveModel("", t.TempDir()); err == nil || !strings.Contains(err.Error(), "no model found") {
		t.Fatalf("got %v", err)
	}
	empty := t.TempDir()
	if _, err := loadModel(empty); err == nil || !strings.Contains(err.Error(), "missing config.json") {
		t.Fatalf("got %v", err)
	}
}

func TestDefaultPrefersFolderNextToExecutable(t *testing.T) {
	exeDir := t.TempDir()
	writeFile(t, filepath.Join(exeDir, graphName), "graph")
	if p, _ := resolveModel("", exeDir); p != filepath.Join(exeDir, graphName) {
		t.Fatalf("file fallback: %s", p)
	}
	if err := os.Mkdir(filepath.Join(exeDir, "model"), 0o755); err != nil {
		t.Fatal(err)
	}
	if p, _ := resolveModel("", exeDir); p != filepath.Join(exeDir, "model") {
		t.Fatalf("folder first: %s", p)
	}
}
