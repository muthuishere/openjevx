#!/usr/bin/env node
import { chmodSync, createWriteStream, existsSync, mkdirSync, writeFileSync } from "node:fs";
import { homedir, platform, arch } from "node:os";
import { join } from "node:path";
import { spawnSync } from "node:child_process";
import { get } from "node:https";

const version = "v0.4.0";
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

if (!existsSync(exe)) {
  mkdirSync(home, { recursive: true });
  const archive = join(home, asset);
  console.log(`Downloading ${asset} over HTTPS...`);
  await download(`${base}/${asset}`, archive);
  const extracted = spawnSync("tar", ["-xf", archive, "-C", home], { stdio: "inherit" });
  if (extracted.status !== 0) process.exit(extracted.status || 1);
  if (!windows) chmodSync(exe, 0o755);
  writeFileSync(join(home, "openjevx.json"), JSON.stringify({ listen: "127.0.0.1:21118", device: "auto" }, null, 2) + "\n");
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
