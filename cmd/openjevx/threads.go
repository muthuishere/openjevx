package main

import (
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"time"
)

// cpuHost is where the CPU limit is read from; tests point it at fake files and a fake metadata server.
type cpuHost struct {
	cgroupRoot string // /sys/fs/cgroup
	ecsURI     string // ECS_CONTAINER_METADATA_URI_V4 ("" = not on ECS)
	client     *http.Client
	numCPU     int
	gomaxprocs int
}

func hostCPU() cpuHost {
	return cpuHost{
		cgroupRoot: "/sys/fs/cgroup",
		ecsURI:     os.Getenv("ECS_CONTAINER_METADATA_URI_V4"),
		client:     &http.Client{Timeout: time.Second},
		numCPU:     runtime.NumCPU(),
		gomaxprocs: runtime.GOMAXPROCS(0),
	}
}

type threadChoice struct {
	n    int
	from string // config, env, cgroup, ecs or GOMAXPROCS
	note string // e.g. "; capped at NumCPU 2"
}

// resolve picks the intra-op thread count: "threads" / OPENJEVX_THREADS; else the cgroup CPU quota
// (v2 cpu.max, v1 cpu.cfs_quota_us / cpu.cfs_period_us), rounded up; else the ECS task's (then the
// container's) Limits.CPU from the task metadata endpoint, for Fargate, which limits CPU with shares
// that neither the cgroup quota nor Go can see; else GOMAXPROCS. Never more than NumCPU, never below 1.
func (h cpuHost) resolve(cfg config) threadChoice {
	c := threadChoice{n: h.gomaxprocs, from: "GOMAXPROCS"}
	switch {
	case cfg.Threads > 0:
		c.n, c.from = cfg.Threads, cfg.threadsFrom
		if c.from == "" {
			c.from = "config"
		}
	default:
		if n, ok := h.cgroupQuota(); ok {
			c.n, c.from = n, "cgroup"
		} else if n, ok := h.ecsLimit(); ok {
			c.n, c.from = n, "ecs"
		}
	}
	if h.numCPU > 0 && c.n > h.numCPU {
		c.n, c.note = h.numCPU, fmt.Sprintf("; capped at NumCPU %d", h.numCPU)
	}
	c.n = max(c.n, 1)
	return c
}

// cgroupQuota is the CPU quota in whole CPUs, rounded up; ok is false when there is none ("max", -1, no files).
func (h cpuHost) cgroupQuota() (int, bool) {
	if b, err := os.ReadFile(filepath.Join(h.cgroupRoot, "cpu.max")); err == nil {
		f := strings.Fields(string(b))
		if len(f) == 2 && f[0] != "max" {
			return ceilRatio(f[0], f[1])
		}
		return 0, false
	}
	for _, dir := range []string{"cpu", "cpu,cpuacct", "cpuacct,cpu"} {
		q, err1 := os.ReadFile(filepath.Join(h.cgroupRoot, dir, "cpu.cfs_quota_us"))
		p, err2 := os.ReadFile(filepath.Join(h.cgroupRoot, dir, "cpu.cfs_period_us"))
		if err1 == nil && err2 == nil {
			return ceilRatio(strings.TrimSpace(string(q)), strings.TrimSpace(string(p)))
		}
	}
	return 0, false
}

func ceilRatio(quota, period string) (int, bool) {
	q, err1 := strconv.ParseFloat(quota, 64)
	p, err2 := strconv.ParseFloat(period, 64)
	if err1 != nil || err2 != nil || q <= 0 || p <= 0 {
		return 0, false
	}
	return max(int(math.Ceil(q/p)), 1), true
}

// ecsLimit reads Limits.CPU from the ECS task metadata endpoint (v4): the task's, else the container's.
// AWS documents the task value in vCPUs (0.25, 1, 2) and the container value in CPU units (1024 = 1 vCPU),
// so a value of 64 or more is read as CPU units and a smaller one as vCPUs. Rounded up; 0 means unset.
func (h cpuHost) ecsLimit() (int, bool) {
	if h.ecsURI == "" {
		return 0, false
	}
	for _, url := range []string{strings.TrimRight(h.ecsURI, "/") + "/task", h.ecsURI} {
		var doc struct {
			Limits struct {
				CPU float64 `json:"CPU"`
			} `json:"Limits"`
		}
		resp, err := h.client.Get(url)
		if err != nil {
			return 0, false // unreachable: the metadata endpoint is one server, so stop here
		}
		err = json.NewDecoder(resp.Body).Decode(&doc)
		resp.Body.Close()
		if err != nil || resp.StatusCode != http.StatusOK || doc.Limits.CPU <= 0 {
			continue
		}
		v := doc.Limits.CPU
		if v >= 64 {
			v /= 1024
		}
		return max(int(math.Ceil(v)), 1), true
	}
	return 0, false
}

// cpuFlags are the SIMD features that decide CPU inference speed (llmresults/14): on x86 AVX2 vs
// AVX-512 VNNI hosts differ about 2x, and Fargate hands out both.
var cpuFlags = map[string][]string{
	"x86":   {"avx2", "avx512f", "avx512_vnni", "avx_vnni", "amx_int8"},
	"arm64": {"asimd", "asimddp", "i8mm", "sve", "sve2", "bf16", "sme"},
}

// cpuLine describes the CPU from /proc/cpuinfo: its model (x86 "model name"; arm64 has none, so the
// implementer and part codes, e.g. 0x41/0xd40 is a Neoverse V1, Graviton3), the flags above it has and
// the ones it lacks. "" when the file is missing (macOS, Windows).
func cpuLine(path string) string {
	b, err := os.ReadFile(path)
	if err != nil {
		return ""
	}
	fields := map[string]string{}
	for _, line := range strings.Split(string(b), "\n") {
		k, v, ok := strings.Cut(line, ":")
		k = strings.TrimSpace(k)
		if _, seen := fields[k]; ok && !seen {
			fields[k] = strings.TrimSpace(v)
		}
	}
	arch, flags := "x86", fields["flags"]
	model := fields["model name"]
	if flags == "" {
		arch, flags = "arm64", fields["Features"]
		if model == "" && fields["CPU implementer"] != "" {
			model = "implementer " + fields["CPU implementer"] + " part " + fields["CPU part"]
		}
	}
	if model == "" {
		model = "unknown model"
	}
	have := map[string]bool{}
	for _, f := range strings.Fields(flags) {
		have[f] = true
	}
	var yes, no []string
	for _, f := range cpuFlags[arch] {
		if have[f] {
			yes = append(yes, f)
		} else {
			no = append(no, f)
		}
	}
	line := "cpu: " + model + "; has " + strings.Join(yes, " ")
	if len(yes) == 0 {
		line = "cpu: " + model + "; has none of " + strings.Join(cpuFlags[arch], " ")
	} else if len(no) > 0 {
		line += "; lacks " + strings.Join(no, " ")
	}
	return line
}
