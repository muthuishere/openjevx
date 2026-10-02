# ADRs

The step-by-step release checklist is [`docs/RELEASE_PROCESS.md`](../RELEASE_PROCESS.md).

Read these before the next OpenJevX training run. Do not revive the Qwen path in `modeltraining`.

| ADR | Use it when |
|---|---|
| [0001](0001-base-model.md) | Choosing or replacing the base model |
| [0002](0002-training-run.md) | Renting a GPU and fine-tuning |
| [0003](0003-export-and-release.md) | Exporting ONNX and publishing |
| [0004](0004-training-data-for-v0.5.md) | Gathering training data for the next model |
| [0005](0005-report-output-folder.md) | Writing agent reports or result files |
| [0006](0006-finetune-on-your-own-data.md) | Writing the fine-tune-on-your-own-data guide, toolkit or skill |
| [0007](0007-managed-finetune-service.md) | Building the managed service: customers bring data, we return their ONNX |
| [0008](0008-self-hosted-appliance.md) | Packaging the self-hosted appliance (DigitalOcean/AWS marketplace) and its training UI |
| [0009](0009-one-job-gpu-run-via-r2.md) | Running a GPU training job (R2 inputs and runtime bundle, self-destroying box) |
| [0010](0010-v0.5-release-and-v0.6-scope.md) | v0.5.0 release shape (CPU-first, container); GPU runner deferred to v0.6 |
| [0011](0011-inference-runtime.md) | Serving runtime and inference speed (ORT 1.29; smaller model is the next step) |
| [0012](0012-model-from-object-store.md) | Serving a model from S3 (cache, verify, fallback, ETag reload) |
