// Build-output checks for the GitHub Pages site. Run after `astro build`.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, existsSync, readdirSync, statSync } from 'node:fs';
import { join } from 'node:path';

const dist = new URL('../dist/', import.meta.url).pathname;
const BASE = '/openjevx';
const pages = ['', 'journey', 'results', 'use', 'finetune'];

function html(page) {
  return readFileSync(join(dist, page, 'index.html'), 'utf8');
}

// The page plus every stylesheet it links, so CSS extracted to /_astro/*.css is checked too.
function css(page) {
  const h = html(page);
  const linked = [...h.matchAll(/<link rel="stylesheet" href="([^"]+)"/g)].map(([, href]) =>
    readFileSync(join(dist, href.slice(BASE.length)), 'utf8'));
  return [h, ...linked].join('\n');
}

function walk(dir) {
  return readdirSync(dir).flatMap((f) => {
    const p = join(dir, f);
    return statSync(p).isDirectory() ? walk(p) : [p];
  });
}

test('every page is built', () => {
  for (const p of pages) assert.ok(existsSync(join(dist, p, 'index.html')), `missing ${p || 'home'}`);
});

test('page content is styled (global styles, not scoped to the layout)', () => {
  for (const p of pages) {
    const h = css(p);
    // Scoped styles would rewrite selectors to pre[data-astro-cid-...]; content in <slot> would stay unstyled.
    assert.match(h, /(^|[\s,}])pre\s*\{[^}]*background/, `${p}: pre has no global rule`);
    assert.match(h, /\.btn\s*\{/, `${p}: .btn has no global rule`);
    assert.match(h, /main\.wrap\s*\{[^}]*padding-top/, `${p}: main top padding is overridden by .wrap`);
    assert.doesNotMatch(h, /data-astro-cid/, `${p}: layout styles are scoped`);
  }
});

test('every internal link resolves to a built file', () => {
  for (const p of pages) {
    for (const [, href] of html(p).matchAll(/href="([^"#]+)/g)) {
      if (!href.startsWith(BASE)) continue;
      const rel = href.slice(BASE.length).replace(/^\//, '');
      const target = join(dist, rel);
      const ok = existsSync(target) && (statSync(target).isFile() || existsSync(join(target, 'index.html')));
      assert.ok(ok, `${p || 'home'}: broken link ${href}`);
    }
  }
});

test('in-page anchors point at real ids', () => {
  for (const p of pages) {
    const h = html(p);
    for (const [, id] of h.matchAll(/href="#([^"]+)"/g)) assert.match(h, new RegExp(`id="${id}"`), `${p}: #${id}`);
  }
});

test('no page mentions the old clauderesults folder', () => {
  for (const f of walk(dist).filter((f) => f.endsWith('.html'))) {
    assert.doesNotMatch(readFileSync(f, 'utf8'), /clauderesults/, f);
  }
});

test('every page has a title and a description', () => {
  for (const p of pages) {
    const h = html(p);
    assert.match(h, /<title>[^<]*OpenJevX[^<]*<\/title>/);
    assert.match(h, /<meta name="description" content="[^"]+"/);
  }
});

