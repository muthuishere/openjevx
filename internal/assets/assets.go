package assets

import _ "embed"

//go:embed model.onnx
var Model []byte

//go:embed dashboard.html
var Dashboard []byte
