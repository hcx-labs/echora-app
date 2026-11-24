import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { mkdtemp, mkdir, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import test from "node:test";

const script = fileURLToPath(new URL("./bump-version.mjs", import.meta.url));

async function fixture(run, lock = { packages: { "web": { version: "1.2.3" } } }) {
  const cwd = await mkdtemp(join(tmpdir(), "echora-version-test-"));
  try {
    await mkdir(join(cwd, "web"), { recursive: true });
    await writeFile(join(cwd, "web/package.json"), JSON.stringify({ version: "1.2.3" }));
    await writeFile(join(cwd, "package-lock.json"), JSON.stringify(lock));
    await run(cwd);
  } finally {
    await rm(cwd, { recursive: true, force: true });
  }
}

for (const [increment, expected] of [["patch", "1.2.4"], ["minor", "1.3.0"], ["major", "2.0.0"]]) {
  test(`${increment} updates package and lock versions without a hardcoded footer`, async () => {
    await fixture(async cwd => {
      const result = spawnSync(process.execPath, [script, increment], { cwd, encoding: "utf8" });
      assert.equal(result.status, 0, result.stderr);
      assert.equal(result.stdout, expected);
      const pkg = JSON.parse(await readFile(join(cwd, "web/package.json"), "utf8"));
      const lock = JSON.parse(await readFile(join(cwd, "package-lock.json"), "utf8"));
      assert.equal(pkg.version, expected);
      assert.equal(lock.packages["web"].version, expected);
    });
  });
}

test("a missing workspace fails before modifying the package version", async () => {
  await fixture(async cwd => {
    const result = spawnSync(process.execPath, [script, "patch"], { cwd, encoding: "utf8" });
    assert.notEqual(result.status, 0);
    assert.match(result.stderr, /web is missing/);
    const pkg = JSON.parse(await readFile(join(cwd, "web/package.json"), "utf8"));
    assert.equal(pkg.version, "1.2.3");
  }, { packages: {} });
});
