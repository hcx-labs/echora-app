import assert from "node:assert/strict";
import test from "node:test";

import { effectsHiddenBySize } from "./smallScreen.ts";

test("small screens hide effects unless the viewer turned them on", () => {
  const small = { matches: true };
  const large = { matches: false };
  assert.equal(effectsHiddenBySize(small, { smallScreenEffects: false }), true);
  assert.equal(effectsHiddenBySize(small, { smallScreenEffects: true }), false);
  assert.equal(effectsHiddenBySize(large, { smallScreenEffects: false }), false);
  // Preferences saved before the switch existed read as off.
  assert.equal(effectsHiddenBySize(small, {}), true);
});
