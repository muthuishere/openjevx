# OpenJevX server image: Go binary, ONNX Runtime beside it, and the model folder at /app/model.
FROM golang:1.26-bookworm AS build
ARG TARGETARCH
ARG ORT_VERSION=1.29.0
# Empty means the version in deploy/MODEL_VERSION.
ARG MODEL_VERSION=
WORKDIR /src
COPY go.mod go.sum* ./
RUN go mod download
COPY . .
# The model folder from the release: model/{openjevx.w8.onnx,config.json,tokenizer.json}.
RUN v="${MODEL_VERSION:-$(cat deploy/MODEL_VERSION)}" && mkdir -p /out && curl -fsSL https://github.com/muthuishere/openjevx/releases/download/v$v/openjevx-model-$v.tar.gz | tar -xz -C /out \
 && test -f /out/model/config.json
RUN case "$TARGETARCH" in arm64) ort=onnxruntime-linux-aarch64 ;; *) ort=onnxruntime-linux-x64 ;; esac \
 && curl -fsSL https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/${ort}-${ORT_VERSION}.tgz | tar -xz \
 && mkdir -p /out && cp -L ${ort}-${ORT_VERSION}/lib/libonnxruntime.so /out/
RUN CGO_ENABLED=1 go build -o /out/openjevx ./cmd/openjevx

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl && rm -rf /var/lib/apt/lists/*
# Runs as uid 10001 (the uid the AWS Marketplace image uses), never root. /app (binary, runtime, model) is read-only to
# it; /data is its working folder: the generated openjevx.json, any generated credential files, and the model cache.
RUN groupadd --system --gid 10001 openjevx && useradd --system --uid 10001 --gid 10001 --home-dir /data --shell /usr/sbin/nologin openjevx \
 && mkdir -p /data && chown openjevx:openjevx /data && chmod 700 /data
# OPENJEVX_DATA: where generated credential files go when the config's folder is read-only (a config mounted at /app).
ENV HOME=/data OPENJEVX_DATA=/data
WORKDIR /app
COPY --from=build /out/ /app/
# Apache-2.0: our LICENSE and NOTICE, and the third-party licenses (ONNX Runtime, Laya, ModernBERT, Go modules).
COPY --from=build /src/LICENSE /src/NOTICE /src/CREDITS /app/
COPY --from=build /src/licenses /app/licenses
# /app/model is found next to /app/openjevx; mount another model folder there to swap models.
# The recipes are also served on /recipes from inside the binary; these are the same pages as files.
COPY --from=build /src/recipes/*.md /usr/share/openjevx/recipes/
# No credentials are baked in: the entrypoint takes OPENJEVX_PASSWORD and OPENJEVX_API_KEY, or a mounted openjevx.json.
COPY --from=build /src/deploy/docker-entrypoint.sh /app/docker-entrypoint.sh
WORKDIR /data
USER 10001:10001
EXPOSE 21160
HEALTHCHECK --interval=30s --timeout=3s --start-period=60s CMD curl -fsS http://127.0.0.1:21160/health || exit 1
ENTRYPOINT ["/app/docker-entrypoint.sh"]
