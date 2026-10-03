package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/muthuishere/openjevx/internal/bpe"
	"github.com/muthuishere/openjevx/internal/decide"
)

// The handler with a fake session: the answer keeps its fields and gains Server-Timing and usage.server_ms.
func TestDecisionServerTiming(t *testing.T) {
	tok, err := bpe.Load()
	if err != nil {
		t.Fatal(err)
	}
	var live atomic.Pointer[served]
	live.Store(&served{m: &loadedModel{Tokenizer: tok, Params: decide.Defaults()}})
	var mu sync.Mutex
	infer := func(_ *served, items []decide.Item) ([]float32, int, []float32, error) {
		time.Sleep(3 * time.Millisecond)
		markers := 0
		for _, it := range items {
			markers = max(markers, len(it.Markers))
		}
		return make([]float32, len(items)*markers), markers, make([]float32, len(items)*2), nil
	}
	h := decisionHandler(&config{Device: "cpu"}, &live, &mu, infer)
	rec := httptest.NewRecorder()
	h(rec, httptest.NewRequest(http.MethodPost, "/v1/systemone", strings.NewReader(
		`{"state":"The parcel arrived crushed.","questions":{"a":{"type":"noul","instructions":"Is the customer angry?"}}}`)))
	if rec.Code != http.StatusOK {
		t.Fatalf("status %d: %s", rec.Code, rec.Body)
	}
	st := rec.Header().Get("Server-Timing")
	m := regexp.MustCompile(`^encode;dur=[0-9.]+, wait;dur=[0-9.]+, run;dur=([0-9.]+), total;dur=([0-9.]+)$`).FindStringSubmatch(st)
	if m == nil {
		t.Fatalf("Server-Timing %q", st)
	}
	var body struct {
		Model   string                     `json:"model"`
		Answers map[string]json.RawMessage `json:"answers"`
		Usage   struct {
			InputTokens  *int     `json:"input_tokens"`
			OutputTokens *int     `json:"output_tokens"`
			ServerMS     *float64 `json:"server_ms"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
		t.Fatal(err)
	}
	u := body.Usage
	if body.Model != "openjevx" || body.Answers["a"] == nil || u.InputTokens == nil || *u.InputTokens == 0 || u.OutputTokens == nil {
		t.Fatalf("existing fields: %s", rec.Body)
	}
	run, _ := strconv.ParseFloat(m[1], 64)
	total, _ := strconv.ParseFloat(m[2], 64)
	if run < 3 || total < run || u.ServerMS == nil || *u.ServerMS < run {
		t.Fatalf("run %v total %v server_ms %v: want run >= the 3 ms the fake session took, total and server_ms >= run (%s)",
			run, total, u.ServerMS, rec.Body)
	}
}
