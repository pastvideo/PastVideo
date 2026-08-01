export function prefixStaticAssets(html, basePath) {
  const prefix = (basePath ?? "").replace(/\/$/, "");
  if (!prefix) return html;
  return html.replaceAll("/assets/", `${prefix}/assets/`);
}
