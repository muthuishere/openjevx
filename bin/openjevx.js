#!/usr/bin/env node
import { chmodSync, createWriteStream, existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { homedir, platform, arch } from "node:os";
import { join } from "node:path";
import { spawnSync } from "node:child_process";
import { get } from "node:https";
import { assetFor } from "./platform.js";
import { mergeConfig } from "./config.js";

const version = "v0.5.8";
// The server and the model are versioned separately (deploy/VERSION, deploy/MODEL_VERSION): v0.5.8 ships the unchanged 0.5.2 model.
const modelVersion = "0.5.2";
const modelAsset = `openjevx-model-${modelVersion}.tar.gz`;
const base = `https://github.com/muthuishere/openjevx/releases/download/${version}`;
const home = process.env.OPENJEVX_HOME || join(homedir(), ".local", "share", "openjevx");
const binDir = process.env.OPENJEVX_BIN || join(homedir(), ".local", "bin");
const windows = platform() === "win32";
let asset;
try { asset = assetFor(platform(), arch()); } catch (e) { console.error(e.message); process.exit(1); }
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
// Archives up to v0.5.6 carried an openjevx.json (with the password "adminadmin") that replaced the user's on every
// upgrade. Releases no longer ship one; the user's file is still put back after unpacking, whatever the archive holds.
const configPath = join(home, "openjevx.json");
if (!existsSync(exe) || installed !== version) {
  mkdirSync(home, { recursive: true });
  const kept = existsSync(configPath) ? readFileSync(configPath) : null;
  await fetchAndUnpack(asset);
  if (kept) writeFileSync(configPath, kept);
  if (!windows) chmodSync(exe, 0o755);
  rmSync(join(home, "model"), { recursive: true, force: true });
  await fetchAndUnpack(modelAsset);
  const merged = mergeConfig(existsSync(configPath) ? readFileSync(configPath, "utf8") : null);
  if (merged) writeFileSync(configPath, merged);
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
console.log(`Dashboard password: printed once on the first start and kept in ${join(home, "openjevx.password")}`);
console.log(`  (or set "password" in ${configPath}; an old "adminadmin" there is ignored)`);
if (!windows) console.log(`If needed: export PATH="${binDir}:$PATH"`);
