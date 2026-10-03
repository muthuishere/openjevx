// The server archive for this machine, or an error naming what is missing. Released today: macOS on Apple Silicon,
// Linux amd64 and arm64 (glibc 2.28+), Windows amd64 (Windows on Arm runs it under x64 emulation).
export function assetFor(os, cpu) {
  if (os === "win32") return "openjevx-windows-amd64.zip";
  if (os === "darwin" && cpu === "arm64") return "openjevx-darwin-arm64.tar";
  if (os === "linux" && cpu === "x64") return "openjevx-linux-amd64.tar";
  if (os === "linux" && cpu === "arm64") return "openjevx-linux-arm64.tar";
  const why = os === "darwin"
    ? "ONNX Runtime 1.29 has no Intel macOS build, so OpenJevX runs on Apple Silicon only (or use Docker)"
    : "there is no OpenJevX release for it (use Docker, or build from source)";
  throw new Error(`OpenJevX: no server build for ${os}/${cpu}: ${why}`);
}
