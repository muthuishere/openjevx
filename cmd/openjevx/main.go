package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strings"

	ort "github.com/yalue/onnxruntime_go"

	"github.com/muthuishere/openjevx/internal/assets"
	"github.com/muthuishere/openjevx/internal/bpe"
	"github.com/muthuishere/openjevx/internal/decide"
)

type config struct {
	Listen  string `json:"listen"`
	Device  string `json:"device"`
	Model   string `json:"model,omitempty"`
	Runtime string `json:"runtime,omitempty"`
}

func main() {
	help := flag.Bool("help", false, "show help")
	flag.Parse()
	if *help || (len(os.Args) > 1 && (os.Args[1] == "-h" || os.Args[1] == "--help")) {
		fmt.Print(usage)
		return
	}
	cfg := loadConfig()
	tok, err := bpe.Load()
	if err != nil {
		log.Fatal(err)
	}
	model, err := modelBytes(cfg)
	if err != nil {
		log.Fatal(err)
	}
	if err := startRuntime(cfg); err != nil {
		log.Fatal(err)
	}
	defer ort.DestroyEnvironment()
	session, used, err := openSession(cfg, model)
	if err != nil {
		log.Fatal(err)
	}
	cfg.Device = used
	log.Printf("device %s", used)
	defer session.Destroy()
	http.HandleFunc("/health", func(w http.ResponseWriter, _ *http.Request) {
		writeJSON(w, map[string]any{"status": "ok", "device": cfg.Device, "model": "openjevx"})
	})
	http.HandleFunc("/v1/systemone", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "POST", http.StatusMethodNotAllowed)
			return
		}
		var req struct {
			State     json.RawMessage            `json:"state"`
			Questions map[string]json.RawMessage `json:"questions"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		order := orderedKeys(req.Questions)
		ids, qs, err := decide.ParseQuestions(req.Questions, order)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		items := decide.Encode(tok, stateText(req.State), qs)
		logits, width, act, err := run(session, items)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		writeJSON(w, map[string]any{
			"model":   "openjevx",
			"answers": decide.Decode(ids, qs, items, logits, width, act),
			"usage":   map[string]int{"input_tokens": tokenCount(items), "output_tokens": 0},
		})
	})
	log.Printf("OpenJevX %s at http://%s/v1/systemone", cfg.Device, cfg.Listen)
	log.Fatal(http.ListenAndServe(cfg.Listen, nil))
}

func run(session *ort.DynamicAdvancedSession, items []decide.Item) ([]float32, int, []float32, error) {
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
			ids[i*length+j] = 50283
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

func modelBytes(cfg config) ([]byte, error) {
	if cfg.Model != "" {
		return os.ReadFile(cfg.Model)
	}
	if len(assets.Model) > 1024 {
		return assets.Model, nil
	}
	path := cfg.Model
	if path == "" {
		exe, _ := os.Executable()
		path = filepath.Join(filepath.Dir(exe), "openjevx.int8.onnx")
	}
	return os.ReadFile(path)
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

func openSession(cfg config, model []byte) (*ort.DynamicAdvancedSession, string, error) {
	names := []string{"input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"}
	outs := []string{"logits", "act_logits"}
	if cfg.Device != "cpu" {
		if session, err := sessionWithCUDA(model, names, outs); err == nil {
			return session, "gpu", nil
		} else if cfg.Device == "gpu" {
			return nil, "", fmt.Errorf("device gpu was set and CUDA did not load: %w", err)
		}
	}
	opts, err := ort.NewSessionOptions()
	if err != nil {
		return nil, "", err
	}
	defer opts.Destroy()
	session, err := ort.NewDynamicAdvancedSessionWithONNXData(model, names, outs, opts)
	if err != nil {
		return nil, "", err
	}
	return session, "cpu", nil
}

func sessionWithCUDA(model []byte, names, outs []string) (*ort.DynamicAdvancedSession, error) {
	opts, err := ort.NewSessionOptions()
	if err != nil {
		return nil, err
	}
	defer opts.Destroy()
	cuda, err := ort.NewCUDAProviderOptions()
	if err != nil {
		return nil, err
	}
	defer cuda.Destroy()
	if err := opts.AppendExecutionProviderCUDA(cuda); err != nil {
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
	cfg.Device = strings.ToLower(strings.TrimSpace(cfg.Device))
	if cfg.Device != "cpu" && cfg.Device != "gpu" {
		cfg.Device = "auto"
	}
	if cfg.Listen == "" {
		cfg.Listen = "127.0.0.1:21118"
	}
	return cfg
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

One executable. No Python. The small int8 ONNX model is inside the release binary.

Linux:    ./README
Windows:  README.cmd

Config file openjevx.json, next to the executable:

  { "listen": "127.0.0.1:8000", "device": "cpu" }

device is cpu or gpu. gpu refuses to start unless the CUDA ONNX Runtime provider loads.
The embedded model is int8 and is the small CPU build. A gpu config can name a separate
CUDA graph with "model".

jevx:
  jevx profile add openjevx http://127.0.0.1:8000/v1/systemone --model openjevx
  jevx profile use openjevx
`
