import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const source = readFileSync(new URL("./RootVisualizer.tsx", import.meta.url), "utf8");

test("root buffer upload stays within capacity after growth exceeds 900 segments", () => {
  // Exercise the actual upload function without a browser or WebGL context.
  const body = source.split("const updateRootBuffers = () => {")[1].split("\n    };", 1)[0];
  const upload = new Function("segments", "MAX_SEGMENTS", "BUCKETS", "palette", "levels", "mix", "rootPositions", "rootColors", "rootPaths", "rootGeometry", "drive", body);
  const positions = new Float32Array(900 * 18);
  const colors = new Float32Array(900 * 18);
  let count = 0;
  const geometry = { setDrawRange: (_, value) => { count = value; }, attributes: { position: {}, color: {}, path: {} } };
  const segments = Array.from({ length: 1100 }, () => ({ x1: 1, y1: 2, x2: 3, y2: 4, life: 1, bucket: 0, distance: 40 }));
  for (let frame = 0; frame < 600; frame++) {
    upload(segments, 900, 16, [[1, 1, 1], [1, 1, 1], [1, 1, 1]], new Float32Array(16), left => left, positions, colors, new Float32Array(900 * 12), geometry, .5);
    assert.equal(count, 5400);
  }
});
