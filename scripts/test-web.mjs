import { readdirSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { spawnSync } from "node:child_process";

const root = fileURLToPath(new URL("../web/", import.meta.url));
function discover(directory) {
  return readdirSync(directory, { withFileTypes: true }).flatMap(entry => {
    const path = join(directory, entry.name);
    if (entry.isDirectory()) return discover(path);
    return entry.isFile() && entry.name.endsWith(".test.mjs") ? [path] : [];
  });
}
const files = ["app", "components", "lib"].flatMap(directory => discover(join(root, directory))).sort();
if (!files.length) throw new Error("No frontend tests found");
const result = spawnSync(process.execPath, ["--test", ...files], { stdio: "inherit" });
if (result.error) throw result.error;
process.exit(result.status ?? 1);
