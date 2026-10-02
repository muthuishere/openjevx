# ADR 0012 — Serve the model folder from an object store (S3 first)

Status: accepted
Date: 2026-10-02

## Context

On AWS Marketplace the customer trains their own model with SageMaker in their own account, and the server must
serve it from their bucket: SageMaker → `s3://<bucket>/training/<job>/output/model.tar.gz` → promote into
`s3://<bucket>/models/current/` → server. Until now `"model"` was only a local path, so a new model meant baking an image
or copying files onto the box. The consumer's contract (jev-cloud `docs/model-s3-contract.md`) fixes the layout,
a read-only IAM role, an empty bucket on a fresh deploy, a cache under `/tmp` (uid 10001, distroless), and
`/health` fields.

## Decision

1. `"model"` / `-model` / `OPENJEVX_MODEL` takes a local path (unchanged) **or** `s3://bucket/prefix/` (the three model
   files) or `s3://bucket/key.tar.gz|.tgz|.tar` (files at the root or under one `model/` folder). The model is still one
   folder; nothing is embedded in the binary.
2. `internal/modelsrc` owns it. A `Store` is two calls, `Stat` and `Fetch`, each returning an ETag; a URL scheme maps to
   one `Store`. s3 uses the AWS SDK for Go v2 default credential chain only. gs:// and azblob:// are registered
   and say "not supported yet"; adding one is a new `Store`.
3. Only `HeadObject` and `GetObject` (both `s3:GetObject`). No `HeadBucket` or `GetBucketLocation`: the region is
   `AWS_REGION` / the profile's, else us-east-1, following the `x-amz-bucket-region` header S3 returns on a redirect.
4. Download into `<cache>/<hash of url>/v-<hash of etags>-<time>/` with a `source.json` manifest, verify (pinned
   `model_sha256`, then `config.json`'s `sha256` of the graph), then rename into place. Pointer files `current` and
   `previous` are written with rename; other versions are deleted.
5. Start: same ETags as the cache → no download. Store unreachable, denied or holding a bad upload → serve a valid
   cache with a warning; no valid cache → refuse to start. Empty prefix → serve `model_fallback` (default: the
   existing `model/` lookup next to the executable, `/app/model` in the image) and poll until a model appears.
6. Reload (`model_reload`, off by default): on new ETags, download, verify, open a new ORT session, swap a pointer
   under the inference lock, destroy the old session. A failed load discards the new folder and keeps serving.
   `/health` reports version, sha256, source, fallback, ETags, `loaded_at` and the previous model.

## Consequences

- A promote that changes only `config.json` still downloads the graph again (about 600 MB). Reusing unchanged files by
  ETag is a later step if promotes get frequent.
- The pin (`model_sha256`) and reload are at odds: a pinned server rejects every new upload. Pin for a fixed version,
  reload for a moving `current/`.
- The store's ETag is the version identity. It is opaque: a multipart re-upload of the same bytes looks new and is
  downloaded once more, which is harmless.
- Tests use an `httptest` fake S3 (path-style); no AWS account is needed in CI.
