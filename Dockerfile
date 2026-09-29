# OpenJevX server image: Go binary, ONNX Runtime beside it, and the model folder at /app/model.
FROM golang:1.26-bookworm AS build
ARG TARGETARCH
ARG ORT_VERSION=1.22.0
ARG MODEL_VERSION=0.5.0
WORKDIR /src
COPY go.mod go.sum* ./
RUN go mod download
COPY . .
# The model folder from the release: model/{openjevx.w8.onnx,config.json,tokenizer.json}.
RUN mkdir -p /out && curl -fsSL https://github.com/muthuishere/openjevx/releases/download/v${MODEL_VERSION}/openjevx-model-${MODEL_VERSION}.tar.gz | tar -xz -C /out \
 && test -f /out/model/config.json
RUN case "$TARGETARCH" in arm64) ort=onnxruntime-linux-aarch64 ;; *) ort=onnxruntime-linux-x64 ;; esac \
 && curl -fsSL https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/${ort}-${ORT_VERSION}.tgz | tar -xz \
 && mkdir -p /out && cp -L ${ort}-${ORT_VERSION}/lib/libonnxruntime.so /out/
RUN CGO_ENABLED=1 go build -o /out/openjevx ./cmd/openjevx

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=build /out/ /app/
# /app/model is found next to /app/openjevx; mount another model folder there to swap models.
# The recipes are also served on /recipes from inside the binary; these are the same pages as files.
COPY --from=build /src/recipes/*.md /usr/share/openjevx/recipes/
RUN printf '{\n  "listen": "0.0.0.0:21118",\n  "device": "cpu",\n  "model": "/app/model",\n  "password": "adminadmin"\n}\n' > /app/openjevx.json
EXPOSE 21118
HEALTHCHECK --interval=30s --timeout=3s --start-period=60s CMD curl -fsS http://127.0.0.1:21118/health || exit 1
CMD ["/app/openjevx"]
