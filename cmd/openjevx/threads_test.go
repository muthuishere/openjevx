package main

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func fakeHost(t *testing.T, files map[string]string) cpuHost {
	t.Helper()
	root := t.TempDir()
	for name, body := range files {
		p := filepath.Join(root, name)
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			t.Fatal(err)
		}
		writeFile(t, p, body)
	}
	return cpuHost{cgroupRoot: root, client: &http.Client{Timeout: time.Second}, numCPU: 8, gomaxprocs: 8}
}

// fakeECS answers /task and the container document the way the ECS task metadata endpoint v4 does.
func fakeECS(t *testing.T, task, container string, delay time.Duration) string {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(delay)
		body := container
		if r.URL.Path == "/v4/abc/task" {
			body = task
		}
		if body == "" {
			http.NotFound(w, r)
			return
		}
		fmt.Fprint(w, body)
	}))
	t.Cleanup(srv.Close)
	return srv.URL + "/v4/abc"
}

func TestThreadsOrder(t *testing.T) {
	cases := []struct {
		name  string
		files map[string]string
		task  string // "" = no ECS
		cont  string
		cfg   config
		n     int
		from  string
	}{
		{"config wins over a quota", map[string]string{"cpu.max": "100000 100000"}, "", "", config{Threads: 3}, 3, "config"},
		{"env wins", nil, "", "", config{Threads: 2, threadsFrom: "env"}, 2, "env"},
		{"cgroup v2 quota rounds up", map[string]string{"cpu.max": "150000 100000\n"}, `{"Limits":{"CPU":4}}`, "", config{}, 2, "cgroup"},
		{"cgroup v2 max falls through to ECS", map[string]string{"cpu.max": "max 100000\n"}, `{"Limits":{"CPU":1}}`, "", config{}, 1, "ecs"},
		{"cgroup v1 quota", map[string]string{"cpu,cpuacct/cpu.cfs_quota_us": "50000\n", "cpu,cpuacct/cpu.cfs_period_us": "100000\n"}, "", "", config{}, 1, "cgroup"},
		{"cgroup v1 unlimited", map[string]string{"cpu/cpu.cfs_quota_us": "-1\n", "cpu/cpu.cfs_period_us": "100000\n"}, "", "", config{}, 8, "GOMAXPROCS"},
		{"Fargate 1 vCPU task", nil, `{"Cluster":"c","Limits":{"CPU":1,"Memory":2048}}`, "", config{}, 1, "ecs"},
		{"task 0.25 vCPU", nil, `{"Limits":{"CPU":0.25}}`, "", config{}, 1, "ecs"},
		{"task in CPU units", nil, `{"Limits":{"CPU":2048}}`, "", config{}, 2, "ecs"},
		{"no task limit: the container's units", nil, `{"Limits":{"Memory":2048}}`, `{"Limits":{"CPU":1536}}`, config{}, 2, "ecs"},
		{"no limits anywhere", nil, `{}`, `{"Limits":{"CPU":0}}`, config{}, 8, "GOMAXPROCS"},
		{"never above NumCPU", map[string]string{"cpu.max": "1600000 100000"}, "", "", config{}, 8, "cgroup"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			h := fakeHost(t, c.files)
			if c.task != "" || c.cont != "" {
				h.ecsURI = fakeECS(t, c.task, c.cont, 0)
			}
			got := h.resolve(c.cfg)
			if got.n != c.n || got.from != c.from {
				t.Fatalf("got %d from %s%s, want %d from %s", got.n, got.from, got.note, c.n, c.from)
			}
		})
	}
}

func TestThreadsECSTimeoutFallsBack(t *testing.T) {
	h := fakeHost(t, nil)
	h.ecsURI = fakeECS(t, `{"Limits":{"CPU":1}}`, "", 1500*time.Millisecond)
	started := time.Now()
	got := h.resolve(config{})
	if got.n != 8 || got.from != "GOMAXPROCS" || time.Since(started) > 1400*time.Millisecond {
		t.Fatalf("got %+v after %s, want GOMAXPROCS within the 1 s timeout", got, time.Since(started))
	}
	h.ecsURI = "http://127.0.0.1:1/v4/none" // nothing listening
	if got := h.resolve(config{}); got.from != "GOMAXPROCS" {
		t.Fatalf("unreachable endpoint: %+v", got)
	}
}

func TestThreadsCapNote(t *testing.T) {
	h := fakeHost(t, nil)
	h.numCPU = 2
	if got := h.resolve(config{Threads: 6, threadsFrom: "env"}); got.n != 2 || got.from != "env" || got.note == "" {
		t.Fatalf("got %+v", got)
	}
}

func TestCPULine(t *testing.T) {
	dir := t.TempDir()
	cases := map[string]struct{ cpuinfo, want string }{
		"amd avx2": {"processor\t: 0\nmodel name\t: AMD EPYC 7763 64-Core Processor\nflags\t\t: fpu sse avx avx2 fma\n\nprocessor\t: 1\nmodel name\t: AMD EPYC 7763 64-Core Processor\nflags\t\t: fpu sse avx avx2 fma\n",
			"cpu: AMD EPYC 7763 64-Core Processor; has avx2; lacks avx512f avx512_vnni avx_vnni amx_int8"},
		"intel vnni": {"model name\t: INTEL(R) XEON(R) PLATINUM 8573C\nflags\t\t: avx avx2 avx512f avx512bw avx512_vnni amx_int8\n",
			"cpu: INTEL(R) XEON(R) PLATINUM 8573C; has avx2 avx512f avx512_vnni amx_int8; lacks avx_vnni"},
		"graviton": {"processor\t: 0\nBogoMIPS\t: 2100.00\nFeatures\t: fp asimd asimddp i8mm sve bf16\nCPU implementer\t: 0x41\nCPU part\t: 0xd40\n",
			"cpu: implementer 0x41 part 0xd40; has asimd asimddp i8mm sve bf16; lacks sve2 sme"},
		"old x86": {"model name\t: Pentium\nflags\t\t: fpu sse\n", "cpu: Pentium; has none of avx2 avx512f avx512_vnni avx_vnni amx_int8"},
	}
	for name, c := range cases {
		p := filepath.Join(dir, name)
		writeFile(t, p, c.cpuinfo)
		if got := cpuLine(p); got != c.want {
			t.Errorf("%s:\n got %q\nwant %q", name, got, c.want)
		}
	}
	if got := cpuLine(filepath.Join(dir, "missing")); got != "" {
		t.Errorf("missing file: %q", got)
	}
}
