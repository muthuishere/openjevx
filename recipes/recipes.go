// Package recipes embeds the recipe pages so the server can serve them on /recipes.
package recipes

import "embed"

//go:embed *.md
var Files embed.FS
