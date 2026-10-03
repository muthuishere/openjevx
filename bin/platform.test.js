import { test } from "node:test";
import assert from "node:assert/strict";
import { assetFor } from "./platform.js";

test("each platform gets its own archive", () => {
  assert.equal(assetFor("linux", "x64"), "openjevx-linux-amd64.tar");
  assert.equal(assetFor("linux", "arm64"), "openjevx-linux-arm64.tar");
  assert.equal(assetFor("darwin", "arm64"), "openjevx-darwin-arm64.tar");
  assert.equal(assetFor("win32", "x64"), "openjevx-windows-amd64.zip");
});

test("unsupported platforms fail with a clear message", () => {
  assert.throws(() => assetFor("darwin", "x64"), /Apple Silicon only/);
  assert.throws(() => assetFor("linux", "ia32"), /no server build for linux\/ia32/);
  assert.throws(() => assetFor("freebsd", "x64"), /no server build/);
});
