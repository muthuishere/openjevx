package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"log"
	"math"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	ort "github.com/yalue/onnxruntime_go"

	"github.com/muthuishere/openjevx/internal/assets"
	"github.com/muthuishere/openjevx/internal/bpe"
	"github.com/muthuishere/openjevx/internal/decide"
	"github.com/muthuishere/openjevx/internal/modelsrc"
	"github.com/muthuishere/openjevx/internal/stats"
	"github.com/muthuishere/openjevx/recipes"
)

type config struct {
	Listen   string `json:"listen"`
	Device   string `json:"device"`
	Password string `json:"password,omitempty"`
	Model    string `json:"model,omitempty"`
	// Remote models (model = s3://...): pinned sha256, cache folder, reload interval, fallback folder.
	ModelSHA256   string `json:"model_sha256,omitempty"`
	ModelCache    string `json:"model_cache,omitempty"`
	ModelReload   string `json:"model_reload,omitempty"`
	ModelFallback string `json:"model_fallback,omitempty"`
	ModelS3Path   bool   `json:"model_s3_path_style,omitempty"` // or AWS_S3_USE_PATH_STYLE=true
	Runtime       string `json:"runtime,omitempty"`
	Threads       int    `json:"threads,omitempty"`
	threadsFrom   string // where Threads came from: config, env, cgroup, ecs or GOMAXPROCS (threads.go)
}

func main() {
	help := flag.Bool("help", false, "show help")
	modelFlag := flag.String("model", "", "model folder, .onnx file, or s3://bucket/prefix/ (or .tar.gz) to serve")
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
	var threads string
	for env, dst := range map[string]*string{
		"OPENJEVX_MODEL_SHA256": &cfg.ModelSHA256, "OPENJEVX_MODEL_CACHE": &cfg.ModelCache,
		"OPENJEVX_MODEL_RELOAD": &cfg.ModelReload, "OPENJEVX_MODEL_FALLBACK": &cfg.ModelFallback,
		"OPENJEVX_THREADS": &threads,
	} {
		if v := os.Getenv(env); v != "" {
			*dst = v
		}
	}
	if threads != "" {
		n, err := strconv.Atoi(threads)
		if err != nil || n < 1 {
			log.Fatalf("OPENJEVX_THREADS=%q: want a whole number of threads, 1 or more", threads)
		}
		cfg.Threads, cfg.threadsFrom = n, "env"
	}
	t := hostCPU().resolve(cfg)
	cfg.Threads, cfg.threadsFrom = t.n, t.from
	log.Printf("threads: intra-op %d (from %s%s), GOMAXPROCS %d, NumCPU %d", t.n, t.from, t.note, runtime.GOMAXPROCS(0), runtime.NumCPU())
	if line := cpuLine("/proc/cpuinfo"); line != "" {
		log.Print(line)
	}
	every, err := reloadEvery(cfg)
	if err != nil {
		log.Fatal(err)
	}
	if err := startRuntime(cfg); err != nil {
		log.Fatal(err)
	}
	defer ort.DestroyEnvironment()
	exe, _ := os.Executable()
	ctx := context.Background()
	first, src, err := startModel(ctx, &cfg, filepath.Dir(exe))
	if err != nil {
		log.Fatal(err)
	}
	log.Printf("device %s", cfg.Device)
	var live atomic.Pointer[served]
	var previous atomic.Pointer[modelInfo]
	live.Store(first)
	defer func() { live.Load().session.Destroy() }()
	var inferMu sync.Mutex
	if src != nil && (every > 0 || first.info.Fallback) {
		go func() {
			for {
				wait := every
				if wait == 0 {
					wait = fallbackPoll
				}
				time.Sleep(wait)
				err := src.Reload(ctx, func(m *modelsrc.Model) error {
					c := cfg // the device is already chosen; don't write the shared config
					next, err := openServed(&c, m.Dir)
					if err != nil {
						return err
					}
					next.info.Source, next.info.RemoteVersion = cfg.Model, m.Version
					inferMu.Lock()
					old := live.Swap(next)
					inferMu.Unlock()
					old.session.Destroy()
					previous.Store(&old.info)
					log.Printf("model reloaded: %s version %s (was %s version %s)", m.Dir, next.info.Version, old.info.Dir, old.info.Version)
					return nil
				})
				if err != nil && !errors.Is(err, modelsrc.ErrEmpty) {
					log.Printf("model reload: %v (still serving %s)", err, live.Load().info.Dir)
				}
				if every == 0 && !live.Load().info.Fallback {
					return // reload is off: the fallback poll ends once the real model is in
				}
			}
		}()
	}
	stats.SetDevice(cfg.Device)
	guard := protected(cfg.Password)

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
		cur := live.Load()
		health := map[string]any{"status": "ok", "device": cfg.Device, "model": cur.m.Name, "version": cur.info.Version,
			"sha256": cur.info.SHA256, "source": cur.info.Source, "fallback": cur.info.Fallback, "loaded_at": cur.info.LoadedAt,
			"dir": cur.info.Dir}
		if cur.info.RemoteVersion != "" {
			health["remote_version"] = cur.info.RemoteVersion
		}
		if cur.info.Offline {
			health["offline"] = true
		}
		if p := previous.Load(); p != nil {
			health["previous"] = p
		}
		writeJSON(w, health)
	})
	http.HandleFunc("/v1/systemone", decisionHandler(&cfg, &live, &inferMu, func(cur *served, items []decide.Item) ([]float32, int, []float32, error) {
		return run(cur.session, cur.m.Params.PAD, items)
	}))
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
	// ONNX Runtime 1.29's Linux build has Microsoft telemetry on by default. Creating the environment then
	// reads /etc/machine-id, else runs popen("echo `blkid; hostname`"), and dereferences a NULL FILE* where
	// there is no /bin/sh (distroless): SIGSEGV in CreateOrtEnv. A self-hosted server sends nothing anyway,
	// so it is off unless ORT_DISABLE_TELEMETRY is set explicitly.
	telemetry := os.Getenv("ORT_DISABLE_TELEMETRY")
	if telemetry == "" {
		if err := os.Setenv("ORT_DISABLE_TELEMETRY", "1"); err != nil {
			return err
		}
	}
	if err := ort.InitializeEnvironment(); err != nil {
		return err
	}
	if telemetry == "" {
		return ort.DisableTelemetry()
	}
	return nil
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

// cpuOptions are the CPU session settings: cfg.Threads intra-op threads, resolved once at startup
// (threads.go), because ONNX Runtime alone sizes its pool from the host's cores, not the container's limit.
func cpuOptions(cfg config) func(*ort.SessionOptions) error {
	return func(o *ort.SessionOptions) error {
		return o.SetIntraOpNumThreads(cfg.Threads)
	}
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
Flags and environment override the file: -model / OPENJEVX_MODEL, -device / OPENJEVX_DEVICE,
threads / OPENJEVX_THREADS (CPU threads per request; default: the cgroup CPU quota, else the ECS/Fargate task's
Limits.CPU, else GOMAXPROCS; never more than the CPUs).

model can also be s3://bucket/prefix/ (holding the three files) or s3://bucket/model.tar.gz. Credentials
come from the AWS default chain (env, profile, instance role, IRSA). The folder is downloaded into a
cache, verified, and served from there; a valid cache starts offline. More settings (env in brackets):

  "model_sha256":   pin the .tar.gz (or the folder's openjevx.w8.onnx) sha256  [OPENJEVX_MODEL_SHA256]
  "model_cache":    cache folder (default: user cache dir/openjevx/models)      [OPENJEVX_MODEL_CACHE]
  "model_reload":   check the ETag this often, e.g. "5m" (default off)          [OPENJEVX_MODEL_RELOAD]
  "model_fallback": served while the bucket is empty (default: model/ next to  [OPENJEVX_MODEL_FALLBACK]
                    the executable)
  "model_s3_path_style": true for MinIO, Ceph or R2 at AWS_ENDPOINT_URL      [AWS_S3_USE_PATH_STYLE]
                    without bucket DNS (bucket in the path, not the host)

jevx:
  jevx profile add openjevx http://127.0.0.1:8000/v1/systemone --model openjevx
  jevx profile use openjevx
`
