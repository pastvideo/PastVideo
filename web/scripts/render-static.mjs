import { mkdir, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";

const basePath = (process.env.NEXT_PUBLIC_PASTVIDEO_BASE ?? "").replace(/\/$/, "");
const workerPath = resolve("dist/server/index.js");
const { default: worker } = await import(pathToFileURL(workerPath).href);
const response = await worker.fetch(
  new Request(`http://localhost${basePath || "/"}/`, {
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
await writeFile(output, await response.text(), "utf8");
console.log(`Rendered ${output}`);
