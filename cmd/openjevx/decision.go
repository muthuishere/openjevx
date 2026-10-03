package main

import (
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"sync"
	"sync/atomic"
	"time"

	"github.com/muthuishere/openjevx/internal/decide"
	"github.com/muthuishere/openjevx/internal/stats"
)

// inferFunc runs one batch through the live session (run in production; a fake in tests).
type inferFunc func(cur *served, items []decide.Item) (logits []float32, width int, act []float32, err error)

// decisionHandler serves POST /v1/systemone. Every answer carries a Server-Timing header
// (encode, wait for the session, run, total; in ms) and usage.server_ms, the same total.
func decisionHandler(cfg *config, live *atomic.Pointer[served], inferMu *sync.Mutex, infer inferFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "POST", http.StatusMethodNotAllowed)
			return
		}
		started := time.Now()
		var req struct {
			State     json.RawMessage            `json:"state"`
			Questions map[string]json.RawMessage `json:"questions"`
		}
		answer := func(errText string, status int) {
			if errText != "" {
				http.Error(w, errText, status)
			}
			var types []string
			for id := range req.Questions {
				if raw, ok := req.Questions[id]; ok {
					var q struct {
						Type string `json:"type"`
					}
					if json.Unmarshal(raw, &q) == nil {
						types = append(types, q.Type)
					}
				}
			}
			stats.Record(stats.Request{
				Time: time.Now(), Duration: float64(time.Since(started).Microseconds()) / 1000,
				Questions: len(req.Questions), Device: cfg.Device, Error: errText,
			}, types)
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			answer(err.Error(), http.StatusBadRequest)
			return
		}
		if len(req.Questions) == 0 {
			answer("questions missing", http.StatusBadRequest)
			return
		}
		order := orderedKeys(req.Questions)
		ids, qs, err := decide.ParseQuestions(req.Questions, order)
		if err != nil {
			answer(err.Error(), http.StatusBadRequest)
			return
		}
		var encode, wait, runDur time.Duration
		t := time.Now()
		cur := live.Load()
		items := decide.Encode(cur.m.Tokenizer, cur.m.Params, stateText(req.State), qs)
		encode = time.Since(t)
		t = time.Now()
		inferMu.Lock()
		wait = time.Since(t)
		if now := live.Load(); now != cur { // swapped by a reload while encoding
			cur = now
			t = time.Now()
			items = decide.Encode(cur.m.Tokenizer, cur.m.Params, stateText(req.State), qs)
			encode += time.Since(t)
		}
		t = time.Now()
		logits, width, act, err := infer(cur, items)
		runDur = time.Since(t)
		inferMu.Unlock()
		if err != nil {
			answer(err.Error(), http.StatusInternalServerError)
			return
		}
		var types []string
		for _, q := range qs {
			types = append(types, q.Type)
		}
		tokens := tokenCount(items)
		answers := decide.Decode(cur.m.Params, ids, qs, items, logits, width, act)
		total := time.Since(started)
		w.Header().Set("Server-Timing", serverTiming(encode, wait, runDur, total))
		writeJSON(w, map[string]any{
			"model":   "openjevx",
			"answers": answers,
			"usage":   map[string]any{"input_tokens": tokens, "output_tokens": 0, "server_ms": math.Round(ms(total)*100) / 100},
		})
		// Recorded after the answer is written so stats never delay the client.
		stats.Record(stats.Request{
			Time:      time.Now(),
			Duration:  float64(time.Since(started).Microseconds()) / 1000,
			Questions: len(ids), Tokens: tokens, Device: cfg.Device,
		}, types)
	}
}

func ms(d time.Duration) float64 { return float64(d.Microseconds()) / 1000 }

// serverTiming is the W3C Server-Timing value: total runs from reading the request to the decoded answer.
func serverTiming(encode, wait, run, total time.Duration) string {
	return fmt.Sprintf("encode;dur=%.1f, wait;dur=%.1f, run;dur=%.1f, total;dur=%.1f", ms(encode), ms(wait), ms(run), ms(total))
}
