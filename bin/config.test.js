import { test } from "node:test";
import assert from "node:assert/strict";
import { mergeConfig, defaults } from "./config.js";

test("a fresh install gets the defaults and no password", () => {
  const cfg = JSON.parse(mergeConfig(null));
  assert.deepEqual(cfg, defaults);
  assert.equal("password" in cfg, false);
});

test("an upgrade never changes the user's values", () => {
  const mine = '{\n  "listen": "0.0.0.0:9999",\n  "device": "cpu",\n  "password": "my-own-secret-pw"\n}\n';
  assert.equal(mergeConfig(mine), null, "nothing missing: the file is not rewritten");
  const merged = JSON.parse(mergeConfig('{"password":"my-own-secret-pw","threads":2}'));
  assert.deepEqual(merged, { password: "my-own-secret-pw", threads: 2, listen: "127.0.0.1:21118", device: "auto" });
});

test("a config that is not a JSON object is left alone", () => {
  assert.equal(mergeConfig("{ not json"), null);
  assert.equal(mergeConfig("[]"), null);
});
