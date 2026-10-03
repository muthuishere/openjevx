#!/bin/sh
# Copy the license and notice files of every Go module linked into the server into licenses/go/,
# plus licenses/go/MODULES.txt (module, version, files). Re-run after changing go.mod.
#   scripts/go_licenses.sh           write licenses/go/
#   scripts/go_licenses.sh --check   fail if licenses/go/ is out of date (CI)
set -eu
cd "$(dirname "$0")/.."
out=licenses/go
[ "${1:-}" = "--check" ] && out=$(mktemp -d)/go
rm -rf "$out" && mkdir -p "$out"
go list -deps -f '{{with .Module}}{{if not .Main}}{{.Path}} {{.Version}} {{.Dir}}{{end}}{{end}}' ./cmd/openjevx | sort -u |
while read -r path version dir; do
  [ -n "$dir" ] || { echo "go_licenses: $path has no source dir (run go mod download)" >&2; exit 1; }
  files=$(cd "$dir" && ls | grep -i -E '^(licen[cs]e|notice|copying|patents)' || true)
  [ -n "$files" ] || { echo "go_licenses: $path@$version has no LICENSE file; check it by hand" >&2; exit 1; }
  mkdir -p "$out/$path"
  for f in $files; do cp "$dir/$f" "$out/$path/$f" && chmod 644 "$out/$path/$f"; done
  echo "$path $version $(echo $files | tr '\n' ' ')" >> "$out/MODULES.txt"
done
if [ "${1:-}" = "--check" ]; then
  diff -r licenses/go "$out" >/dev/null || { echo "licenses/go is out of date: run scripts/go_licenses.sh" >&2; diff -r licenses/go "$out" | head -20 >&2; exit 1; }
  echo "licenses/go is up to date"
fi
