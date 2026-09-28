#!/usr/bin/env node

import { existsSync, mkdirSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { spawn, spawnSync } from "node:child_process";

const args = process.argv.slice(2);

function value(name, fallback) {
  const index = args.indexOf(name);
  return index >= 0 && args[index + 1] ? args[index + 1] : fallback;
}

if (args.includes("--help") || args.includes("-h")) {
  console.log(`openjevx [options]

Options:
  --port PORT       HTTP port (default: 8000)
  --host HOST       Bind address (default: 127.0.0.1)
  --device DEVICE   cpu, cuda, or mps (default: auto)
  --model MODEL     Hugging Face model (default: muthuishere/openjevx)
  --help             Show this help
`);
  process.exit(0);
}

const root = dirname(dirname(fileURLToPath(import.meta.url)));
const home = process.env.OPENJEVX_HOME || join(homedir(), ".cache", "openjevx");
const venv = join(home, "venv");
const windows = process.platform === "win32";
const python = join(venv, windows ? "Scripts/python.exe" : "bin/python");
const systemPython = process.env.OPENJEVX_PYTHON || (windows ? "python" : "python3");

mkdirSync(home, { recursive: true });

if (!existsSync(python)) {
  console.log("Creating the OpenJevX runtime...");
  const created = spawnSync(systemPython, ["-m", "venv", venv], { stdio: "inherit" });
  if (created.status !== 0) {
    console.error("Python 3.10 or newer with venv support is required.");
    process.exit(created.status || 1);
  }
}

const marker = join(home, "laya-0.3.21-installed");
if (!existsSync(marker)) {
  console.log("Installing the pinned OpenJevX runtime...");
  const installed = spawnSync(
    python,
    ["-m", "pip", "install", "--disable-pip-version-check", "laya[serve]==0.3.21"],
    { stdio: "inherit" },
  );
  if (installed.status !== 0) process.exit(installed.status || 1);
  spawnSync(python, ["-c", `from pathlib import Path; Path(${JSON.stringify(marker)}).touch()`]);
}

const serverArgs = [
  join(root, "server.py"),
  "--host", value("--host", "127.0.0.1"),
  "--port", value("--port", "8000"),
  "--model", value("--model", "muthuishere/openjevx"),
];
const device = value("--device", "");
if (device) serverArgs.push("--device", device);

const child = spawn(python, serverArgs, { stdio: "inherit" });
for (const signal of ["SIGINT", "SIGTERM"]) {
  process.on(signal, () => child.kill(signal));
}
child.on("exit", (code, signal) => {
  process.exitCode = signal ? 1 : (code ?? 1);
});
