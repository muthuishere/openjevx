package stats

import (
	"fmt"
	"sort"
	"strings"
	"sync"
	"time"
)

const maxRecent = 200
const maxLatencies = 20000

var started = time.Now()

type Request struct {
	Time      time.Time `json:"time"`
	Duration  float64   `json:"ms"`
	Questions int       `json:"questions"`
	Tokens    int       `json:"input_tokens"`
	Device    string    `json:"device"`
	Error     string    `json:"error,omitempty"`
}

type counter struct {
	Total     int64   `json:"total"`
	Errors    int64   `json:"errors"`
	Questions int64   `json:"questions"`
	Tokens    int64   `json:"input_tokens"`
	Device    string  `json:"device"`
	P50       float64 `json:"p50_ms"`
	P95       float64 `json:"p95_ms"`
	P99       float64 `json:"p99_ms"`
	Max       float64 `json:"max_ms"`
	Avg       float64 `json:"avg_ms"`
}

type store struct {
	mu        sync.Mutex
	total     int64
	errors    int64
	questions int64
	tokens    int64
	latSum    float64
	latencies []float64 // ring buffer of the last maxLatencies durations
	latNext   int
	recent    []Request // ring buffer of the last maxRecent requests
	recNext   int
	perType   map[string]int64
}

var Store = &store{perType: map[string]int64{}}

func Record(r Request, types []string) {
	Store.mu.Lock()
	defer Store.mu.Unlock()
	Store.total++
	if r.Error != "" {
		Store.errors++
	}
	Store.questions += int64(r.Questions)
	Store.tokens += int64(r.Tokens)
	Store.latSum += r.Duration
	for _, t := range types {
		Store.perType[t]++
	}
	if len(Store.latencies) < maxLatencies {
		Store.latencies = append(Store.latencies, r.Duration)
	} else {
		Store.latencies[Store.latNext] = r.Duration
		Store.latNext = (Store.latNext + 1) % maxLatencies
	}
	if len(Store.recent) < maxRecent {
		Store.recent = append(Store.recent, r)
	} else {
		Store.recent[Store.recNext] = r
		Store.recNext = (Store.recNext + 1) % maxRecent
	}
}

func pct(sorted []float64, p float64) float64 {
	if len(sorted) == 0 {
		return 0
	}
	i := int(p * float64(len(sorted)-1))
	return sorted[i]
}

func Snapshot(device string) map[string]any {
	// Copy under the lock, sort outside it, so a dashboard poll never blocks Record.
	Store.mu.Lock()
	sorted := append([]float64(nil), Store.latencies...)
	recent := append(append([]Request(nil), Store.recent[Store.recNext:]...), Store.recent[:Store.recNext]...)
	total, errs, questions, tokens := Store.total, Store.errors, Store.questions, Store.tokens
	var avg float64
	if total > 0 {
		avg = Store.latSum / float64(total)
	}
	types := map[string]any{}
	for k, v := range Store.perType {
		types[k] = v
	}
	Store.mu.Unlock()
	sort.Float64s(sorted)
	return map[string]any{
		"uptime_seconds": int(time.Since(started).Seconds()),
		"device":         device,
		"requests": counter{
			Total: total, Errors: errs, Questions: questions,
			Tokens: tokens, Device: device,
			Avg: round(avg), P50: round(pct(sorted, .5)), P95: round(pct(sorted, .95)),
			P99: round(pct(sorted, .99)), Max: round(pct(sorted, 1)),
		},
		"per_question_type": types,
		"recent":            recent,
	}
}

var device = "cpu"

func SetDevice(d string) { device = d }

func round(v float64) float64 {
	m := float64(100)
	return float64(int(v*m+.5)) / m
}

func Prometheus() string {
	s := Snapshot(device)
	c := s["requests"].(counter)
	var b strings.Builder
	line := func(name, typ, help string, v float64) {
		fmt.Fprintf(&b, "# HELP %s %s\n# TYPE %s %s\n%s %g\n", name, help, name, typ, name, v)
	}
	line("openjevx_requests_total", "counter", "OpenJevX decisions served", float64(c.Total))
	line("openjevx_request_errors_total", "counter", "OpenJevX failed requests", float64(c.Errors))
	line("openjevx_questions_total", "counter", "OpenJevX questions answered", float64(c.Questions))
	line("openjevx_input_tokens_total", "counter", "OpenJevX encoder tokens", float64(c.Tokens))
	line("openjevx_latency_ms_avg", "gauge", "OpenJevX mean latency in milliseconds", c.Avg)
	line("openjevx_latency_ms_p50", "gauge", "OpenJevX median latency in milliseconds", c.P50)
	line("openjevx_latency_ms_p95", "gauge", "OpenJevX p95 latency in milliseconds", c.P95)
	line("openjevx_latency_ms_p99", "gauge", "OpenJevX p99 latency in milliseconds", c.P99)
	line("openjevx_latency_ms_max", "gauge", "OpenJevX max latency in milliseconds", c.Max)
	return b.String()
}
