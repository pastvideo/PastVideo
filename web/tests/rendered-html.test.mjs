import assert from "node:assert/strict";
import { access, readFile } from "node:fs/promises";
import test from "node:test";

process.env.NEXT_PUBLIC_PASTVIDEO_BASE = "/pastvideo_demo";

async function render() {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);

  return worker.fetch(
    new Request("http://localhost/pastvideo_demo/", {
      headers: { accept: "text/html" },
    }),
    {
      ASSETS: {
        fetch: async () => new Response("Not found", { status: 404 }),
      },
    },
    {
      waitUntil() {},
      passThroughOnException() {},
    },
  );
}

test("server-renders the PastVideo search shell and social metadata", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);

  const html = await response.text();
  assert.match(html, /<title>PastVideo — Local semantic video search<\/title>/i);
  assert.match(html, /data-testid="search-input"/);
  assert.match(html, /data-testid="search-button"/);
  assert.match(html, /All videos/);
  assert.match(html, /data-testid="video-list"/);
  assert.doesNotMatch(html, /Find the moment\.|Not the filename\.|RETRIEVAL ENGINE/);
  assert.match(html, /an archer shooting an arrow/);
  assert.match(html, /a person bowling/);
  assert.match(html, /people flying a kite/);
  assert.match(html, /an athlete doing the high jump/);
  assert.match(html, /a marching band/);
  assert.match(html, /property="og:image" content="\/pastvideo_demo\/pastvideo-social\.png"/);
  assert.match(html, /\/pastvideo_demo\/assets\//);
  assert.doesNotMatch(html, /codex-preview|Your site is taking shape/);
});

test("the client is wired to the real local API", async () => {
  const [page, packageJson] = await Promise.all([
    readFile(new URL("../app/page.tsx", import.meta.url), "utf8"),
    readFile(new URL("../package.json", import.meta.url), "utf8"),
    access(new URL("../public/pastvideo-social.png", import.meta.url)),
  ]);

  assert.match(page, /NEXT_PUBLIC_PASTVIDEO_BASE/);
  assert.match(page, /NEXT_PUBLIC_PASTVIDEO_API/);
  assert.match(page, /\/api\/status/);
  assert.match(page, /\/api\/videos/);
  assert.match(page, /\/api\/search/);
  assert.match(page, /\/api\/clip/);
  assert.match(page, /selected\.media_url/);
  assert.match(page, /catalogSelection\.media_url/);
  assert.match(page, /selectCatalogVideo/);
  assert.match(page, /currentTime = selected\.start_time/);
  assert.match(page, /Clip ready — download MP4/);
  assert.doesNotMatch(packageJson, /react-loading-skeleton/);
});
