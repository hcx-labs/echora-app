import { readFile, writeFile } from "node:fs/promises";

const increment = process.argv[2];
if (!["major", "minor", "patch"].includes(increment)) {
  throw new Error("Usage: node .github/scripts/bump-version.mjs <major|minor|patch>");
}

const packagePath = "web/package.json";
const lockPath = "package-lock.json";
const packageJson = JSON.parse(await readFile(packagePath, "utf8"));
const match = /^(\d+)\.(\d+)\.(\d+)$/.exec(packageJson.version);

if (!match) {
  throw new Error(`Invalid current version: ${packageJson.version}`);
}

let [, major, minor, patch] = match.map((part, index) => index === 0 ? part : BigInt(part));
if (increment === "major") {
  major += 1n;
  minor = 0n;
  patch = 0n;
} else if (increment === "minor") {
  minor += 1n;
  patch = 0n;
} else {
  patch += 1n;
}

const version = `${major}.${minor}.${patch}`;
packageJson.version = version;
const packageLock = JSON.parse(await readFile(lockPath, "utf8"));
const workspace = packageLock.packages?.["web"];
if (!workspace) {
  throw new Error("web is missing from package-lock.json");
}
workspace.version = version;
// The UI imports the package version directly; no component source needs updating.
await writeFile(packagePath, `${JSON.stringify(packageJson, null, 2)}\n`);
await writeFile(lockPath, `${JSON.stringify(packageLock, null, 2)}\n`);

process.stdout.write(version);
