package main

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/muthuishere/openjevx/recipes"
)

func getRecipe(t *testing.T, path string) *httptest.ResponseRecorder {
	t.Helper()
	w := httptest.NewRecorder()
	recipesHandler(recipes.Files)(w, httptest.NewRequest(http.MethodGet, path, nil))
	return w
}

func TestRecipesIndexListsFiles(t *testing.T) {
	w := getRecipe(t, "/recipes")
	if w.Code != http.StatusOK {
		t.Fatalf("status %d", w.Code)
	}
	for _, want := range []string{`href="/recipes/what-you-get"`, `href="/recipes/01-find-failures-in-a-log"`, `href="/recipes/18-would-the-user-accept"`} {
		if !strings.Contains(w.Body.String(), want) {
			t.Errorf("index is missing %s", want)
		}
	}
	if w := getRecipe(t, "/recipes/what-you-get"); w.Code != http.StatusOK || !strings.Contains(w.Body.String(), "<h1>What you get</h1>") {
		t.Errorf("what-you-get: status %d", w.Code)
	}
}

func TestRecipesUnknownAndTraversal(t *testing.T) {
	for _, path := range []string{"/recipes/nope", "/recipes/../main.go", "/recipes/..%2Frecipes", "/recipes/%2e%2e/go.mod", "/recipes/recipes.go", "/recipes/a/b"} {
		if w := getRecipe(t, path); w.Code != http.StatusNotFound {
			t.Errorf("%s: status %d, want 404", path, w.Code)
		}
	}
}

func TestRenderMarkdownEscapesHTML(t *testing.T) {
	out := renderMarkdown("<script>alert(1)</script> [x](javascript:alert(1)) `<b>`\n\n```\n<i>\n```")
	if strings.Contains(out, "<script>") || strings.Contains(out, "javascript:") || strings.Contains(out, "<i>") || strings.Contains(out, "<b>") {
		t.Fatalf("unescaped html: %s", out)
	}
}
