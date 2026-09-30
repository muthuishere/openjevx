package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"time"

	ort "github.com/yalue/onnxruntime_go"

	"github.com/muthuishere/openjevx/internal/assets"
	"github.com/muthuishere/openjevx/internal/bpe"
	"github.com/muthuishere/openjevx/internal/decide"
	"github.com/muthuishere/openjevx/internal/stats"
	"github.com/muthuishere/openjevx/recipes"
)

type config struct {
	Listen   string `json:"listen"`
	Device   string `json:"device"`
	Password string `json:"password,omitempty"`
	Model    string `json:"model,omitempty"`
	Runtime  string `json:"runtime,omitempty"`
}

func main() {
	help := flag.Bool("help", false, "show help")
	modelFlag := flag.String("model", "", "ONNX model file to serve instead of the embedded one")
	deviceFlag := flag.String("device", "", "auto, cpu or gpu")
	flag.Parse()
	if *help || (len(os.Args) > 1 && (os.Args[1] == "-h" || os.Args[1] == "--help")) {
		fmt.Print(usage)
		return
	}
	cfg := loadConfig()
	// Precedence: flag, then environment, then openjevx.json, then the default model location.
	for _, v := range []string{os.Getenv("OPENJEVX_MODEL"), *modelFlag} {
		if v != "" {
			cfg.Model = v
		}
	}
	for _, v := range []string{os.Getenv("OPENJEVX_DEVICE"), *deviceFlag} {
		if v != "" {
			cfg.Device = normalDevice(v)
		}
	}
	exe, _ := os.Executable()
	path, err := resolveModel(cfg.Model, filepath.Dir(exe))
	if err != nil {
		log.Fatal(err)
	}
	m, err := loadModel(path)
	if err != nil {
		log.Fatal(err)
	}
	tokSrc := m.TokFile
	if tokSrc == "" {
		tokSrc = "embedded"
	}
	log.Printf("model %s version %s sha256 %s temperatures choice=%g score=%g noul=%g max_len %d head_max %d tokenizer %s",
		m.Path, m.Version, m.SHA256[:12], m.Params.Temperature[0], m.Params.Temperature[1], m.Params.Temperature[2],
		m.Params.MaxLen, m.Params.HeadMax, tokSrc)
	if err := startRuntime(cfg); err != nil {
		log.Fatal(err)
	}
	defer ort.DestroyEnvironment()
	probe, err := probeItems(m.Tokenizer, m.Params)
	if err != nil {
		log.Fatal(err)
	}
	session, used, err := openSession(cfg, m.Graph, probe)
	m.Graph = nil // the session holds its own copy
	if err != nil {
		log.Fatal(err)
	}
	cfg.Device = used
	log.Printf("device %s", used)
	defer session.Destroy()
	stats.SetDevice(cfg.Device)
	guard := protected(cfg.Password)
	var inferMu sync.Mutex

	http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		guard(w, r, func() {
			w.Header().Set("Content-Type", "text/html; charset=utf-8")
			_, _ = w.Write(assets.Dashboard)
		})
	})
	serveRecipes := recipesHandler(recipes.Files)
	http.HandleFunc("/recipes", func(w http.ResponseWriter, r *http.Request) {
		guard(w, r, func() { serveRecipes(w, r) })
	})
	http.HandleFunc("/recipes/", func(w http.ResponseWriter, r *http.Request) {
		guard(w, r, func() { serveRecipes(w, r) })
	})
	http.HandleFunc("/stats", func(w http.ResponseWriter, r *http.Request) {
		guard(w, r, func() { writeJSON(w, stats.Snapshot(cfg.Device)) })
	})
	http.HandleFunc("/metrics", func(w http.ResponseWriter, r *http.Request) {
		guard(w, r, func() {
			w.Header().Set("Content-Type", "text/plain; version=0.0.4")
			_, _ = w.Write([]byte(stats.Prometheus()))
		})
	})

	http.HandleFunc("/health", func(w http.ResponseWriter, _ *http.Request) {
		writeJSON(w, map[string]any{"status": "ok", "device": cfg.Device, "model": m.Name, "version": m.Version, "sha256": m.SHA256})
	})
	http.HandleFunc("/v1/systemone", func(w http.ResponseWriter, r *http.Request) {
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
		items := decide.Encode(m.Tokenizer, m.Params, stateText(req.State), qs)
		inferMu.Lock()
		logits, width, act, err := run(session, m.Params.PAD, items)
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
		writeJSON(w, map[string]any{
			"model":   "openjevx",
			"answers": decide.Decode(m.Params, ids, qs, items, logits, width, act),
			"usage":   map[string]int{"input_tokens": tokens, "output_tokens": 0},
		})
		// Recorded after the answer is written so stats never delay the client.
		stats.Record(stats.Request{
			Time:      time.Now(),
			Duration:  float64(time.Since(started).Microseconds()) / 1000,
			Questions: len(ids), Tokens: tokens, Device: cfg.Device,
		}, types)
	})
	log.Printf("OpenJevX %s at http://%s (dashboard on /, metrics on /metrics)", cfg.Device, cfg.Listen)
	log.Fatal(http.ListenAndServe(cfg.Listen, nil))
}

func run(session *ort.DynamicAdvancedSession, pad int, items []decide.Item) ([]float32, int, []float32, error) {
	n := len(items)
	length, markers := 0, 0
	for _, item := range items {
		length = max(length, len(item.IDs))
		markers = max(markers, len(item.Markers))
	}
	ids := make([]int64, n*length)
	attn := make([]int64, n*length)
	mpos := make([]int64, n*markers)
	mmask := make([]bool, n*markers)
	qtype := make([]int64, n)
	for i, item := range items {
		for j, id := range item.IDs {
			ids[i*length+j] = int64(id)
			attn[i*length+j] = 1
		}
		for j := len(item.IDs); j < length; j++ {
			ids[i*length+j] = int64(pad)
		}
		for j, m := range item.Markers {
			mpos[i*markers+j] = int64(m)
			mmask[i*markers+j] = true
		}
		qtype[i] = int64(item.QType)
	}
	inIDs, err := ort.NewTensor(ort.NewShape(int64(n), int64(length)), ids)
	if err != nil {
		return nil, 0, nil, err
	}
	defer inIDs.Destroy()
	inAttn, err := ort.NewTensor(ort.NewShape(int64(n), int64(length)), attn)
	if err != nil {
		return nil, 0, nil, err
	}
	defer inAttn.Destroy()
	inPos, err := ort.NewTensor(ort.NewShape(int64(n), int64(markers)), mpos)
	if err != nil {
		return nil, 0, nil, err
	}
	defer inPos.Destroy()
	inMask, err := ort.NewTensor(ort.NewShape(int64(n), int64(markers)), mmask)
	if err != nil {
		return nil, 0, nil, err
	}
	defer inMask.Destroy()
	inType, err := ort.NewTensor(ort.NewShape(int64(n)), qtype)
	if err != nil {
		return nil, 0, nil, err
	}
	defer inType.Destroy()
	outLogits, err := ort.NewEmptyTensor[float32](ort.NewShape(int64(n), int64(markers)))
	if err != nil {
		return nil, 0, nil, err
	}
	defer outLogits.Destroy()
	outAct, err := ort.NewEmptyTensor[float32](ort.NewShape(int64(n), 2))
	if err != nil {
		return nil, 0, nil, err
	}
	defer outAct.Destroy()
	err = session.Run([]ort.Value{inIDs, inAttn, inPos, inMask, inType}, []ort.Value{outLogits, outAct})
	if err != nil {
		return nil, 0, nil, err
	}
	return outLogits.GetData(), markers, outAct.GetData(), nil
}

func startRuntime(cfg config) error {
	path := cfg.Runtime
	if path == "" {
		path = os.Getenv("OPENJEVX_ORT")
	}
	if path == "" {
		exe, _ := os.Executable()
		dir := filepath.Dir(exe)
		for _, name := range []string{"libonnxruntime.so", "libonnxruntime.dylib", "onnxruntime.dll"} {
			candidate := filepath.Join(dir, name)
			if _, err := os.Stat(candidate); err == nil {
				path = candidate
				break
			}
		}
	}
	if path != "" {
		ort.SetSharedLibraryPath(path)
	}
	return ort.InitializeEnvironment()
}

// probeItems is one fixed request used to check that a GPU provider really runs the model.
func probeItems(tok *bpe.Tokenizer, p decide.Params) ([]decide.Item, error) {
	raw := map[string]json.RawMessage{
		"a": json.RawMessage(`{"type":"noul","instructions":"Is the customer angry?"}`),
		"b": json.RawMessage(`{"type":"choice","instructions":"Which team?","criteria":{"1":"Refunds","2":"Damaged parcels","3":"Login"}}`),
	}
	_, qs, err := decide.ParseQuestions(raw, []string{"a", "b"})
	if err != nil {
		return nil, err
	}
	return decide.Encode(tok, p, "Refund request: the parcel arrived crushed and the customer is furious.", qs), nil
}

// agree runs the probe on both sessions and fails unless the GPU logits match the CPU ones.
func agree(gpu, cpu *ort.DynamicAdvancedSession, probe []decide.Item) error {
	want, _, _, err := run(cpu, decide.Defaults().PAD, probe)
	if err != nil {
		return fmt.Errorf("cpu probe: %w", err)
	}
	got, _, _, err := run(gpu, decide.Defaults().PAD, probe)
	if err != nil {
		return fmt.Errorf("probe run: %w", err)
	}
	if len(got) != len(want) {
		return fmt.Errorf("probe gave %d logits, cpu %d", len(got), len(want))
	}
	worst := 0.0
	for i := range got {
		d := math.Abs(float64(got[i] - want[i]))
		if math.IsNaN(d) {
			return fmt.Errorf("probe gave NaN")
		}
		worst = math.Max(worst, d)
	}
	if worst > 0.25 {
		return fmt.Errorf("probe logits differ from cpu by %.3f", worst)
	}
	return nil
}

func openSession(cfg config, model []byte, probe []decide.Item) (*ort.DynamicAdvancedSession, string, error) {
	names, outs := inputNames, outputNames
	cpu, err := sessionWith(model, names, outs, cpuOptions(cfg))
	if err != nil {
		return nil, "", err
	}
	if cfg.Device == "cpu" {
		return cpu, "cpu", nil
	}
	var failed []string
	for _, p := range gpuProviders {
		session, err := sessionWith(model, names, outs, p.add)
		if err == nil {
			if err = agree(session, cpu, probe); err == nil {
				log.Printf("gpu provider %s", p.name)
				cpu.Destroy()
				return session, "gpu", nil
			}
			session.Destroy()
		}
		failed = append(failed, fmt.Sprintf("%s: %v", p.name, err))
	}
	if cfg.Device == "gpu" {
		cpu.Destroy()
		return nil, "", fmt.Errorf("device gpu was set and no GPU provider ran the model: %s", strings.Join(failed, "; "))
	}
	log.Printf("no GPU provider ran the model, using CPU: %s", strings.Join(failed, "; "))
	return cpu, "cpu", nil
}

var (
	inputNames  = []string{"input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"}
	outputNames = []string{"logits", "act_logits"}
)

// cpuOptions are the CPU session settings.
func cpuOptions(cfg config) func(*ort.SessionOptions) error {
	return func(*ort.SessionOptions) error { return nil }
}

// gpuProviders are tried in order; the first one this ONNX Runtime build can load wins.
var gpuProviders = []struct {
	name string
	add  func(*ort.SessionOptions) error
}{
	{"cuda", func(o *ort.SessionOptions) error {
		cuda, err := ort.NewCUDAProviderOptions()
		if err != nil {
			return err
		}
		defer cuda.Destroy()
		return o.AppendExecutionProviderCUDA(cuda)
	}},
	{"coreml", func(o *ort.SessionOptions) error {
		if runtime.GOOS != "darwin" {
			return fmt.Errorf("macOS only")
		}
		return o.AppendExecutionProviderCoreMLV2(map[string]string{"ModelFormat": "MLProgram", "MLComputeUnits": "ALL"})
	}},
	{"coreml-nn", func(o *ort.SessionOptions) error {
		if runtime.GOOS != "darwin" {
			return fmt.Errorf("macOS only")
		}
		return o.AppendExecutionProviderCoreMLV2(map[string]string{"ModelFormat": "NeuralNetwork", "MLComputeUnits": "ALL"})
	}},
	{"directml", func(o *ort.SessionOptions) error {
		if runtime.GOOS != "windows" {
			return fmt.Errorf("Windows only")
		}
		return o.AppendExecutionProviderDirectML(0)
	}},
}

func sessionWith(model []byte, names, outs []string, add func(*ort.SessionOptions) error) (*ort.DynamicAdvancedSession, error) {
	opts, err := ort.NewSessionOptions()
	if err != nil {
		return nil, err
	}
	defer opts.Destroy()
	if err := add(opts); err != nil {
		return nil, err
	}
	return ort.NewDynamicAdvancedSessionWithONNXData(model, names, outs, opts)
}

func loadConfig() config {
	cfg := config{Listen: "127.0.0.1:21118", Device: "auto"}
	for _, path := range configPaths() {
		b, err := os.ReadFile(path)
		if err != nil {
			continue
		}
		if json.Unmarshal(b, &cfg) == nil {
			break
		}
	}
	cfg.Device = normalDevice(cfg.Device)
	if cfg.Listen == "" {
		cfg.Listen = "127.0.0.1:21118"
	}
	return cfg
}

func normalDevice(d string) string {
	d = strings.ToLower(strings.TrimSpace(d))
	if d != "cpu" && d != "gpu" {
		return "auto"
	}
	return d
}

func configPaths() []string {
	exe, _ := os.Executable()
	return []string{"openjevx.json", filepath.Join(filepath.Dir(exe), "openjevx.json")}
}

func stateText(raw json.RawMessage) string {
	raw = bytesTrim(raw)
	if len(raw) == 0 {
		return ""
	}
	if raw[0] == '"' {
		var s string
		_ = json.Unmarshal(raw, &s)
		return s
	}
	return string(raw)
}

func orderedKeys(raw map[string]json.RawMessage) []string {
	ids := make([]string, 0, len(raw))
	for id := range raw {
		ids = append(ids, id)
	}
	return ids
}

func tokenCount(items []decide.Item) int {
	n := 0
	for _, item := range items {
		n += len(item.IDs)
	}
	return n
}

func protected(password string) func(http.ResponseWriter, *http.Request, func()) {
	return func(w http.ResponseWriter, r *http.Request, fn func()) {
		if password == "" {
			fn()
			return
		}
		_, pass, ok := r.BasicAuth()
		if !ok {
			if pass = r.URL.Query().Get("password"); pass == "" {
				w.Header().Set("WWW-Authenticate", `Basic realm="openjevx"`)
				http.Error(w, "unauthorized", http.StatusUnauthorized)
				return
			}
		}
		if pass != password {
			w.Header().Set("WWW-Authenticate", `Basic realm="openjevx"`)
			http.Error(w, "unauthorized", http.StatusUnauthorized)
			return
		}
		fn()
	}
}

func writeJSON(w http.ResponseWriter, v any) {
	w.Header().Set("Content-Type", "application/json")
	enc := json.NewEncoder(w)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(v); err != nil {
		log.Printf("encode response: %v", err)
	}
}

func bytesTrim(b []byte) []byte {
	return []byte(strings.TrimSpace(string(b)))
}

func bytesHasPrefix(b, prefix []byte) bool {
	return len(b) >= len(prefix) && string(b[:len(prefix)]) == string(prefix)
}

const usage = `OpenJevX

One executable. No Python. The model is a folder next to it (model/): the 8-bit ONNX graph,
config.json (its calibration temperatures) and tokenizer.json.

Linux:    ./README
Windows:  README.cmd

Config file openjevx.json, next to the executable:

  { "listen": "127.0.0.1:8000", "device": "cpu", "model": "model" }

model is a model folder (or, for old models, a .onnx file). Unset: model/ or models/openjevx/
next to the executable, then openjevx.w8.onnx next to it.

device is auto (default), cpu or gpu. auto uses the first GPU provider that loads (CUDA, CoreML on
macOS, DirectML on Windows) and whose answers match the CPU on a probe, otherwise CPU. gpu refuses to
start unless one of them loads.

model is a model folder (openjevx.w8.onnx + config.json + tokenizer.json) or a plain .onnx file.
Unset: model/ or models/openjevx/ next to the executable, then openjevx.w8.onnx next to it.
Flags and environment override the file: -model / OPENJEVX_MODEL, -device / OPENJEVX_DEVICE.

jevx:
  jevx profile add openjevx http://127.0.0.1:8000/v1/systemone --model openjevx
  jevx profile use openjevx
`
