# OpenJevX server image: Go binary with the 8-bit model embedded, ONNX Runtime beside it.
FROM golang:1.26-bookworm AS build
ARG TARGETARCH
ARG ORT_VERSION=1.22.0
WORKDIR /src
COPY go.mod go.sum* ./
RUN go mod download
COPY . .
RUN apt-get update && apt-get install -y --no-install-recommends unzip && rm -rf /var/lib/apt/lists/* \
 && curl -fsSL -o /tmp/w8.zip https://github.com/muthuishere/openjevx/releases/download/v0.4.0/openjevx.w8.onnx.zip && unzip -p /tmp/w8.zip openjevx.w8.onnx > internal/assets/model.onnx && rm /tmp/w8.zip
RUN case "$TARGETARCH" in arm64) ort=onnxruntime-linux-aarch64 ;; *) ort=onnxruntime-linux-x64 ;; esac \
 && curl -fsSL https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/${ort}-${ORT_VERSION}.tgz | tar -xz \
 && mkdir -p /out && cp -L ${ort}-${ORT_VERSION}/lib/libonnxruntime.so /out/
RUN CGO_ENABLED=1 go build -o /out/openjevx ./cmd/openjevx

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=build /out/ /app/
RUN printf '{\n  "listen": "0.0.0.0:21118",\n  "device": "cpu",\n  "password": "adminadmin"\n}\n' > /app/openjevx.json
EXPOSE 21118
HEALTHCHECK --interval=30s --timeout=3s --start-period=60s CMD curl -fsS http://127.0.0.1:21118/health || exit 1
CMD ["/app/openjevx"]
