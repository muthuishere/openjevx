#!/usr/bin/env node
import { chmodSync, createWriteStream, existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { homedir, platform, arch } from "node:os";
import { join } from "node:path";
import { spawnSync } from "node:child_process";
import { get } from "node:https";

const version = "v0.5.6";
// The server and the model are versioned separately (deploy/VERSION, deploy/MODEL_VERSION): v0.5.6 ships the unchanged 0.5.2 model.
const modelVersion = "0.5.2";
const modelAsset = `openjevx-model-${modelVersion}.tar.gz`;
const base = `https://github.com/muthuishere/openjevx/releases/download/${version}`;
const home = process.env.OPENJEVX_HOME || join(homedir(), ".local", "share", "openjevx");
const binDir = process.env.OPENJEVX_BIN || join(homedir(), ".local", "bin");
const windows = platform() === "win32";
const asset = windows
  ? "openjevx-windows-amd64.zip"
  : platform() === "darwin"
    ? "openjevx-darwin-arm64.tar"
    : arch() === "arm64"
      ? "openjevx-linux-amd64.tar"
      : "openjevx-linux-amd64.tar";
const exe = join(home, windows ? "openjevx.exe" : "openjevx");

function download(url, dest) {
  return new Promise((resolve, reject) => {
    const file = createWriteStream(dest);
    const req = get(url, (res) => {
      if (res.statusCode >= 300 && res.statusCode < 400 && res.headers.location) {
        file.close();
        download(res.headers.location, dest).then(resolve, reject);
        return;
      }
      if (res.statusCode !== 200) {
        reject(new Error(`HTTP ${res.statusCode} for ${url}`));
        return;
      }
      res.pipe(file);
      file.on("finish", () => file.close(resolve));
    });
    req.on("error", reject);
  });
}

// A release is a small binary per platform plus one model folder (model/: graph, config.json, tokenizer.json).
const stamp = join(home, "VERSION");
let installed = "";
try { installed = readFileSync(stamp, "utf8").trim(); } catch {}
async function fetchAndUnpack(name) {
  const archive = join(home, name);
  console.log(`Downloading ${name} over HTTPS...`);
  await download(`${base}/${name}`, archive);
  const extracted = spawnSync("tar", ["-xf", archive, "-C", home], { stdio: "inherit" });
  if (extracted.status !== 0) process.exit(extracted.status || 1);
  rmSync(archive, { force: true });
}
if (!existsSync(exe) || installed !== version) {
  mkdirSync(home, { recursive: true });
  await fetchAndUnpack(asset);
  if (!windows) chmodSync(exe, 0o755);
  rmSync(join(home, "model"), { recursive: true, force: true });
  await fetchAndUnpack(modelAsset);
  if (!existsSync(join(home, "openjevx.json"))) {
    writeFileSync(join(home, "openjevx.json"), JSON.stringify({ listen: "127.0.0.1:21118", device: "auto" }, null, 2) + "\n");
  }
  writeFileSync(stamp, version + "\n");
}

mkdirSync(binDir, { recursive: true });
const cmd = windows ? join(binDir, "openjevx.cmd") : join(binDir, "openjevx");
if (windows) {
  writeFileSync(cmd, `@echo off\r\ncd /d "${home}"\r\n"${exe}" %*\r\n`);
} else {
  writeFileSync(cmd, `#!/bin/sh\ncd "${home}" || exit 1\nexec "${exe}" "$@"\n`);
  chmodSync(cmd, 0o755);
}
console.log("OpenJevX is ready.");
console.log("Default port: 21118");
console.log("Server:  http://127.0.0.1:21118/v1/systemone");
console.log("Command: openjevx");
if (!windows) console.log(`If needed: export PATH="${binDir}:$PATH"`);
