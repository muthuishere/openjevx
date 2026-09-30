#!/bin/sh
# Build the training book ("How to train your own Jev") as markdown + PDF into dist/book/.
# Source: docs/book/train-your-own-jev.md. {{SOURCE_URL}} is filled from $BOOK_SOURCE_URL
# (default: the git remote of this checkout).
# PUBLIC=1 builds the copy for deemwar surfaces (jev.deemwar.com/book): it refuses any output
# that names 'muthuishere' or links github.com, so set BOOK_SOURCE_URL to the public source first.
# Needs pandoc and typst.
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
SRC=$ROOT/docs/book/train-your-own-jev.md
OUT=$ROOT/dist/book
command -v pandoc >/dev/null || { echo "book: pandoc not found (brew install pandoc)" >&2; exit 1; }
command -v typst >/dev/null || { echo "book: typst not found (brew install typst)" >&2; exit 1; }
URL=${BOOK_SOURCE_URL:-$(git -C "$ROOT" remote get-url origin | sed "s#^git@\([^:]*\):#https://\1/#")}
mkdir -p "$OUT"
sed "s#{{SOURCE_URL}}#$URL#g" "$SRC" > "$OUT/train-your-own-jev.md"
if [ "${PUBLIC:-0}" = 1 ] && grep -n -i -e muthuishere -e github.com "$OUT/train-your-own-jev.md" >&2; then
  echo "book: the lines above break the public-text rule (no 'muthuishere', no github.com links); set BOOK_SOURCE_URL" >&2
  rm -f "$OUT/train-your-own-jev.md"
  exit 1
fi
pandoc "$OUT/train-your-own-jev.md" --pdf-engine=typst -V papersize=a4 -V margin.x=2cm -V margin.y=2cm \
  -o "$OUT/train-your-own-jev.pdf"
echo "book: $OUT/train-your-own-jev.md"
echo "book: $OUT/train-your-own-jev.pdf"
