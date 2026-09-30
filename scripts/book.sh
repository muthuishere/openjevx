#!/bin/sh
# Build the training book ("How to train your own Jev") as markdown + PDF into dist/book/.
# Source: docs/book/train-your-own-jev.md. {{SOURCE_URL}} is filled from $BOOK_SOURCE_URL
# (default: the origin remote of this checkout; the public URL is https://jev.deemwar.com/source).
# PUBLIC=1 builds the copy for deemwar surfaces (jev.deemwar.com/book): it refuses any output
# that names 'muthuishere' or links github.com; its source URL defaults to https://jev.deemwar.com/source.
# Needs pandoc, typst and python3.
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
SRC=$ROOT/docs/book/train-your-own-jev.md
OUT=$ROOT/dist/book
command -v pandoc >/dev/null || { echo "book: pandoc not found (brew install pandoc)" >&2; exit 1; }
command -v typst >/dev/null || { echo "book: typst not found (brew install typst)" >&2; exit 1; }
[ "${PUBLIC:-0}" = 1 ] && : "${BOOK_SOURCE_URL:=https://jev.deemwar.com/source}"
URL=${BOOK_SOURCE_URL:-$(git -C "$ROOT" remote get-url origin 2>/dev/null || true)}
[ -n "$URL" ] || { echo "book: no source URL: set BOOK_SOURCE_URL (e.g. https://jev.deemwar.com/source) or add an origin remote" >&2; exit 1; }
mkdir -p "$OUT"
# git@host:path -> https://host/path; drop any user:token@ so a credential never lands in the book.
# The URL is passed through the environment, never into a sed replacement, so # & \ are safe.
URL="$URL" python3 - "$SRC" "$OUT/train-your-own-jev.md" <<'PY'
import os, re, sys
url = os.environ["URL"]
url = re.sub(r"^git@([^:]+):", r"https://\1/", url)
url = re.sub(r"^([a-z+]+://)[^/@]*@", r"\1", url)
src, dst = sys.argv[1:]
open(dst, "w").write(open(src).read().replace("{{SOURCE_URL}}", url))
PY
if [ "${PUBLIC:-0}" = 1 ] && grep -n -i -e muthuishere -e github.com "$OUT/train-your-own-jev.md" >&2; then
  echo "book: the lines above break the public-text rule (no 'muthuishere', no github.com links); set BOOK_SOURCE_URL" >&2
  rm -f "$OUT/train-your-own-jev.md"
  exit 1
fi
pandoc "$OUT/train-your-own-jev.md" --pdf-engine=typst -V papersize=a4 -V margin.x=2cm -V margin.y=2cm \
  -o "$OUT/train-your-own-jev.pdf"
echo "book: $OUT/train-your-own-jev.md"
echo "book: $OUT/train-your-own-jev.pdf"
