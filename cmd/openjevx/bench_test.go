package main

// The latency benchmark. It needs a real model folder and ONNX Runtime, so it only runs when asked:
//
//	OPENJEVX_BENCH=.local/model OPENJEVX_ORT=.local/libonnxruntime.dylib go test ./cmd/openjevx -run TestBench -v -timeout 30m
//
// It prints cold start (read+verify, session creation, first run) and p50/p95/p99 per stage for a short,
// a typical and a long request, sent alone and as a batch of questions. OPENJEVX_BENCH_N sets the rounds.

import (
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"strconv"
	"strings"
	"testing"
	"time"

	ort "github.com/yalue/onnxruntime_go"

	"github.com/muthuishere/openjevx/internal/decide"
)

type benchCase struct {
	name      string
	state     string
	questions map[string]json.RawMessage
}

func benchCases() []benchCase {
	para := "The customer wrote in because the parcel arrived crushed, the blender inside is cracked and they want a refund, not a replacement. They have ordered from us for six years and say this is the third damaged delivery this year. "
	one := map[string]json.RawMessage{"angry": json.RawMessage(`{"type":"noul","instructions":"Is the customer angry?"}`)}
	many := map[string]json.RawMessage{
		"angry":  json.RawMessage(`{"type":"noul","instructions":"Is the customer angry?"}`),
		"team":   json.RawMessage(`{"type":"choice","instructions":"Which team should handle this?","criteria":{"refunds":"Refunds and billing","damage":"Damaged parcels","login":"Account and login","other":"Anything else"}}`),
		"urgent": json.RawMessage(`{"type":"score","instructions":"How urgent is this?","criteria":["not urgent","can wait a day","today","right now"]}`),
		"churn":  json.RawMessage(`{"type":"noul","instructions":"Is the customer likely to leave?"}`),
		"refund": json.RawMessage(`{"type":"noul","instructions":"Does the customer ask for money back?"}`),
		"lang":   json.RawMessage(`{"type":"choice","instructions":"Which language is the message in?","criteria":["English","Spanish","German","Other"]}`),
		"polite": json.RawMessage(`{"type":"score","instructions":"How polite is the message?","criteria":["rude","neutral","polite"]}`),
		"repeat": json.RawMessage(`{"type":"noul","instructions":"Has this happened before?"}`),
	}
	return []benchCase{
		{"short x1", "Where is my parcel?", one},
		{"typical x1", strings.Repeat(para, 3), one},
		{"long x1", strings.Repeat(para, 20), one},
		{"typical x8", strings.Repeat(para, 3), many},
		{"long x8", strings.Repeat(para, 20), many},
	}
}

func pct(d []time.Duration, p float64) float64 {
	s := append([]time.Duration(nil), d...)
	sort.Slice(s, func(i, j int) bool { return s[i] < s[j] })
	i := int(p * float64(len(s)-1))
	return float64(s[i].Microseconds()) / 1000
}

func TestBench(t *testing.T) {
	dir := os.Getenv("OPENJEVX_BENCH")
	if dir == "" {
		t.Skip("set OPENJEVX_BENCH to a model folder")
	}
	rounds := 30
	if v, err := strconv.Atoi(os.Getenv("OPENJEVX_BENCH_N")); err == nil && v > 0 {
		rounds = v
	}
	t0 := time.Now()
	m, err := loadModel(dir)
	if err != nil {
		t.Fatal(err)
	}
	load := time.Since(t0)
	if err := startRuntime(loadConfig()); err != nil {
		t.Fatal(err)
	}
	defer ort.DestroyEnvironment()
	t1 := time.Now()
	session, err := sessionWith(m.Graph, inputNames, outputNames, cpuOptions(loadConfig()))
	if err != nil {
		t.Fatal(err)
	}
	defer session.Destroy()
	create := time.Since(t1)
	probe, _ := probeItems(m.Tokenizer, m.Params)
	t2 := time.Now()
	if _, _, _, err := run(session, m.Params.PAD, probe); err != nil {
		t.Fatal(err)
	}
	first := time.Since(t2)
	fmt.Printf("\ncold start: read+sha256 %.0f ms · session %.0f ms · first run %.0f ms · total %.0f ms\n\n",
		ms(load), ms(create), ms(first), ms(load+create+first))
	fmt.Printf("| case | tokens | encode p50 | run p50 | run p95 | run p99 | decode p50 | total p50 | total p95 | total p99 |\n|---|---|---|---|---|---|---|---|---|---|\n")
	for _, c := range benchCases() {
		order := orderedKeys(c.questions)
		ids, qs, err := decide.ParseQuestions(c.questions, order)
		if err != nil {
			t.Fatal(err)
		}
		var enc, inf, dec, total []time.Duration
		var tokens int
		for r := 0; r < rounds+2; r++ {
			a := time.Now()
			items := decide.Encode(m.Tokenizer, m.Params, c.state, qs)
			b := time.Now()
			logits, width, act, err := run(session, m.Params.PAD, items)
			if err != nil {
				t.Fatal(err)
			}
			d := time.Now()
			_ = decide.Decode(m.Params, ids, qs, items, logits, width, act)
			e := time.Now()
			if r < 2 { // warm-up
				continue
			}
			tokens = tokenCount(items)
			enc, inf, dec, total = append(enc, b.Sub(a)), append(inf, d.Sub(b)), append(dec, e.Sub(d)), append(total, e.Sub(a))
		}
		fmt.Printf("| %s | %d | %.2f | %.1f | %.1f | %.1f | %.3f | %.1f | %.1f | %.1f |\n", c.name, tokens,
			pct(enc, .5), pct(inf, .5), pct(inf, .95), pct(inf, .99), pct(dec, .5), pct(total, .5), pct(total, .95), pct(total, .99))
	}
}

func ms(d time.Duration) float64 { return float64(d.Microseconds()) / 1000 }
