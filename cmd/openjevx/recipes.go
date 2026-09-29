package main

import (
	"html"
	"io/fs"
	"net/http"
	"regexp"
	"sort"
	"strconv"
	"strings"
)

// recipesHandler serves the embedded markdown recipes: /recipes is the index, /recipes/<name> one page.
func recipesHandler(files fs.FS) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		name := strings.TrimPrefix(strings.TrimPrefix(r.URL.Path, "/recipes"), "/")
		if name == "" {
			name = "README"
		}
		name = strings.TrimSuffix(name, ".md")
		if !validRecipe.MatchString(name) {
			http.NotFound(w, r)
			return
		}
		body, err := fs.ReadFile(files, name+".md")
		if err != nil {
			http.NotFound(w, r)
			return
		}
		if r.URL.Query().Get("raw") != "" {
			w.Header().Set("Content-Type", "text/markdown; charset=utf-8")
			_, _ = w.Write(body)
			return
		}
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		page := renderMarkdown(string(body))
		if name == "README" {
			page += recipeList(files)
		}
		_, _ = w.Write([]byte(recipePage(name, page)))
	}
}

var validRecipe = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9_-]*$`)

func recipeList(files fs.FS) string {
	names, _ := fs.Glob(files, "*.md")
	sort.Strings(names)
	var b strings.Builder
	b.WriteString("<h2>All files</h2><ul>")
	for _, n := range names {
		n = strings.TrimSuffix(n, ".md")
		if n == "README" {
			continue
		}
		b.WriteString(`<li><a href="/recipes/` + n + `">` + html.EscapeString(n) + "</a></li>")
	}
	b.WriteString("</ul>")
	return b.String()
}

func recipePage(name, body string) string {
	return `<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1"><title>OpenJevX recipes</title><style>
:root{--bg:#f6f7f9;--panel:#fff;--line:#e4e7ec;--text:#101828;--muted:#667085;--accent:#4f46e5}
@media (prefers-color-scheme:dark){:root{--bg:#0b0d12;--panel:#12151c;--line:#232733;--text:#e6e8ee;--muted:#8a91a3;--accent:#818cf8}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:15px/1.6 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
.wrap{max-width:860px;margin:0 auto;padding:24px 16px 48px}header{display:flex;gap:14px;align-items:center;margin-bottom:20px;font-size:13px}
.logo{font-weight:700;font-size:18px}a{color:var(--accent)}main{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:8px 24px 20px}
pre{background:var(--bg);border:1px solid var(--line);border-radius:8px;padding:12px;overflow-x:auto;font-size:13px;line-height:1.45}
code{font:13px ui-monospace,SFMono-Regular,Menlo,monospace}p code,li code,td code{background:var(--bg);padding:1px 4px;border-radius:4px}
table{border-collapse:collapse;width:100%;display:block;overflow-x:auto}th,td{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left;font-size:13px}
h1{font-size:24px;letter-spacing:-.01em}h2{font-size:18px;margin-top:28px}h3{font-size:15px}
</style></head><body><div class=wrap><header><span class=logo>OpenJevX</span><a href="/">Dashboard</a><a href="/recipes">Recipes</a><a href="/recipes/what-you-get">What you get</a><a href="/recipes/` + name + `?raw=1">Markdown</a></header><main>` + body + `</main></div></body></html>`
}

// renderMarkdown renders the small markdown subset the recipes use. Every piece of text is HTML-escaped.
func renderMarkdown(src string) string {
	var b strings.Builder
	lines := strings.Split(strings.ReplaceAll(src, "\r\n", "\n"), "\n")
	var para []string
	list := ""
	flush := func() {
		if len(para) > 0 {
			b.WriteString("<p>" + inline(strings.Join(para, " ")) + "</p>\n")
			para = nil
		}
		if list != "" {
			b.WriteString("</" + list + ">\n")
			list = ""
		}
	}
	for i := 0; i < len(lines); i++ {
		line := lines[i]
		trimmed := strings.TrimSpace(line)
		switch {
		case strings.HasPrefix(trimmed, "```"):
			flush()
			var code []string
			for i++; i < len(lines) && !strings.HasPrefix(strings.TrimSpace(lines[i]), "```"); i++ {
				code = append(code, lines[i])
			}
			b.WriteString("<pre><code>" + html.EscapeString(strings.Join(code, "\n")) + "</code></pre>\n")
		case trimmed == "":
			flush()
		case strings.HasPrefix(trimmed, "#"):
			flush()
			level := len(trimmed) - len(strings.TrimLeft(trimmed, "#"))
			if level > 6 {
				level = 6
			}
			tag := string(rune('0' + level))
			b.WriteString("<h" + tag + ">" + inline(strings.TrimSpace(trimmed[level:])) + "</h" + tag + ">\n")
		case strings.HasPrefix(trimmed, "|"):
			flush()
			b.WriteString("<table>")
			row := 0
			for ; i < len(lines) && strings.HasPrefix(strings.TrimSpace(lines[i]), "|"); i++ {
				cells := strings.Split(strings.Trim(strings.TrimSpace(lines[i]), "|"), "|")
				if strings.Trim(strings.Join(cells, ""), "-: ") == "" {
					continue
				}
				cell := "td"
				if row == 0 {
					cell = "th"
				}
				b.WriteString("<tr>")
				for _, c := range cells {
					b.WriteString("<" + cell + ">" + inline(strings.TrimSpace(c)) + "</" + cell + ">")
				}
				b.WriteString("</tr>")
				row++
			}
			i--
			b.WriteString("</table>\n")
		case strings.HasPrefix(trimmed, "- ") || strings.HasPrefix(trimmed, "* ") || orderedItem.MatchString(trimmed):
			kind := "ul"
			item := trimmed[2:]
			if m := orderedItem.FindString(trimmed); m != "" {
				kind, item = "ol", trimmed[len(m):]
			}
			if len(para) > 0 || (list != "" && list != kind) {
				flush()
			}
			if list == "" {
				list = kind
				b.WriteString("<" + kind + ">")
			}
			b.WriteString("<li>" + inline(item) + "</li>")
		default:
			if list != "" {
				flush()
			}
			para = append(para, trimmed)
		}
	}
	flush()
	return b.String()
}

var (
	orderedItem = regexp.MustCompile(`^\d+\. `)
	inlineCode  = regexp.MustCompile("`([^`]+)`")
	boldText    = regexp.MustCompile(`\*\*([^*]+)\*\*`)
	linkText    = regexp.MustCompile(`\[([^\]]+)\]\(([^)\s]+)\)`)
)

// inline escapes the text first, then turns code spans, bold and links into tags.
func inline(text string) string {
	var codes []string
	text = inlineCode.ReplaceAllStringFunc(text, func(m string) string {
		codes = append(codes, "<code>"+html.EscapeString(m[1:len(m)-1])+"</code>")
		return "\x00" + strconv.Itoa(len(codes)-1) + "\x00"
	})
	text = html.EscapeString(text)
	text = boldText.ReplaceAllString(text, "<strong>$1</strong>")
	text = linkText.ReplaceAllStringFunc(text, func(m string) string {
		parts := linkText.FindStringSubmatch(m)
		return `<a href="` + safeHref(html.UnescapeString(parts[2])) + `">` + parts[1] + "</a>"
	})
	for i, c := range codes {
		text = strings.Replace(text, "\x00"+strconv.Itoa(i)+"\x00", c, 1)
	}
	return text
}

func safeHref(href string) string {
	switch {
	case strings.HasPrefix(href, "https://"), strings.HasPrefix(href, "http://"):
		return html.EscapeString(href)
	case strings.HasSuffix(href, ".md") && validRecipe.MatchString(strings.TrimSuffix(strings.TrimPrefix(href, "./"), ".md")):
		return "/recipes/" + strings.TrimSuffix(strings.TrimPrefix(href, "./"), ".md")
	}
	return "#"
}
