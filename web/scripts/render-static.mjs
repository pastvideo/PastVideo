import { mkdir, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { prefixStaticAssets } from "./static-html.mjs";

const basePath = (process.env.NEXT_PUBLIC_PASTVIDEO_BASE ?? "").replace(/\/$/, "");
const workerPath = resolve("dist/server/index.js");
const { default: worker } = await import(pathToFileURL(workerPath).href);
const response = await worker.fetch(
  // The built worker receives a basePath-stripped pathname. The environment
  // value still prefixes links and assets in the rendered HTML.
  new Request("http://localhost/", {
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

if (!response.ok) {
  throw new Error(`Static render failed with HTTP ${response.status}`);
}

const output = resolve("dist/client/index.html");
await mkdir(resolve("dist/client"), { recursive: true });
await writeFile(output, prefixStaticAssets(await response.text(), basePath), "utf8");
console.log(`Rendered ${output}`);
