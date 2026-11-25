type Color = [number, number, number];
export type TrackPalette = { accent: Color; background: Color; waves: [Color, Color, Color]; trails: Color[] };

const clamp = (n: number) => Math.max(0, Math.min(1, n));
const linear = (n: number) => n <= .04045 ? n / 12.92 : ((n + .055) / 1.055) ** 2.4;
const encoded = (n: number) => n <= .0031308 ? 12.92 * n : 1.055 * n ** (1 / 2.4) - .055;
const luminance = (rgb: Color) => rgb.map(n => linear(n / 255)).reduce((sum, n, i) => sum + n * [.2126, .7152, .0722][i], 0);
const distance = (a: Color, b: Color) => Math.hypot(...a.map((n, i) => n - b[i]));
const chroma = (lab: Color) => Math.hypot(lab[1], lab[2]);

function toLab(rgb: Color): Color {
  const [r, g, b] = rgb.map(n => linear(n / 255));
  const l = Math.cbrt(.4122214708 * r + .5363325363 * g + .0514459929 * b);
  const m = Math.cbrt(.2119034982 * r + .6806995451 * g + .1073969566 * b);
  const s = Math.cbrt(.0883024619 * r + .2817188376 * g + .6299787005 * b);
  return [.2104542553 * l + .793617785 * m - .0040720468 * s, 1.9779984951 * l - 2.428592205 * m + .4505937099 * s, .0259040371 * l + .7827717662 * m - .808675766 * s];
}

function fromLab([light, a, b]: Color): Color {
  const l = (light + .3963377774 * a + .2158037573 * b) ** 3;
  const m = (light - .1055613458 * a - .0638541728 * b) ** 3;
  const s = (light - .0894841775 * a - 1.291485548 * b) ** 3;
  return [4.0767416621 * l - 3.3077115913 * m + .2309699292 * s, -1.2684380046 * l + 2.6097574011 * m - .3413193965 * s, -.0041960863 * l - .7034186147 * m + 1.707614701 * s];
}

// Reduce chroma along the original hue ray, rather than clipping RGB or mixing white.
function gamut(light: number, source: Color, scale = 1): Color {
  const c = chroma(source);
  const a = c < .008 ? 0 : source[1] * scale;
  const b = c < .008 ? 0 : source[2] * scale;
  let low = 0, high = 1;
  for (let i = 0; i < 24; i++) {
    const mid = (low + high) / 2;
    const rgb = fromLab([light, a * mid, b * mid]);
    if (rgb.every(n => n >= 0 && n <= 1)) low = mid; else high = mid;
  }
  return fromLab([light, a * low, b * low]).map(n => Math.round(clamp(encoded(clamp(n))) * 255)) as Color;
}

/** A near-black panel tone that follows the artwork hue without lifting text backgrounds. */
export function panelColorFromPalette(palette: TrackPalette): Color {
  return gamut(.145, toLab(palette.accent), .1);
}

/** RGBA bytes in scan order. Preserve the canvas extractor's alpha >= 180 cutoff.
 * Empty/transparent artwork throws so the provider keeps its existing default palette.
 */
export function paletteFromPixels(pixels: ArrayLike<number>): TrackPalette {
  const histogram = new Map<number, number>();
  let total = 0;
  for (let i = 0; i + 3 < pixels.length; i += 4) {
    if (pixels[i + 3] < 180) continue;
    const key = pixels[i] * 65536 + pixels[i + 1] * 256 + pixels[i + 2];
    histogram.set(key, (histogram.get(key) || 0) + 1);
    total++;
  }
  if (!total) throw new Error("Artwork has no usable colors");
  // Most-supported seeds first; numeric tie breaks make pixel ordering irrelevant.
  const colors = [...histogram].sort((a, b) => b[1] - a[1] || a[0] - b[0]);
  const clusters: { seed: Color; sum: Color; count: number; lab: Color }[] = [];
  for (const [key, count] of colors) {
    const lab = toLab([Math.floor(key / 65536), Math.floor(key / 256) % 256, key % 256]);
    let nearest = clusters[0], best = Infinity;
    for (const cluster of clusters) {
      const d = distance(lab, cluster.seed);
      if (d < best) { best = d; nearest = cluster; }
    }
    if (nearest && best < .065) {
      nearest.count += count;
      lab.forEach((n, i) => { nearest.sum[i] += n * count; });
    } else clusters.push({ seed: lab, sum: lab.map(n => n * count) as Color, count, lab });
  }
  for (const cluster of clusters) cluster.lab = cluster.sum.map(n => n / cluster.count) as Color;
  // A colorful region needs >= 2% support; single-pixel specks cannot beat the cover.
  const supported = clusters.filter(c => c.count >= Math.max(2, total * .02));
  const candidates = supported.length ? supported : [clusters[0]];
  const vibrant = candidates.filter(c => chroma(c.lab) >= .055);
  const score = (c: typeof clusters[number]) => Math.sqrt(c.count / total) * (.04 + chroma(c.lab));
  const ranked = [...(vibrant.length ? vibrant : candidates)].sort((a, b) => score(b) - score(a));
  const source = ranked[0].lab;
  const background = gamut(.16, source, .35);
  // #16302d is the darker text used on Browse's accent fills. Also budget for
  // its 88%-opaque fill over black, stricter than the player's #07100f controls.
  const floor = Math.max(luminance([7, 16, 15]), luminance(background), luminance([22, 48, 45]));
  // Retain relative saturation when lifting very dark pigments; gamut mapping
  // limits the extra chroma without changing hue (neutrals remain neutral).
  const accentAt = (light: number) => gamut(light, source, Math.max(1, Math.min(2.5, light / Math.max(.1, source[0]))));
  let accent = accentAt(Math.max(.64, Math.min(.84, source[0])));
  for (let light = Math.max(.64, Math.min(.84, source[0])); light <= 1; light += .002) {
    accent = accentAt(light);
    const fill = accent.map(n => n * .88) as Color;
    if ((luminance(fill) + .05) / (floor + .05) >= 4.5) break;
  }
  // Select additional supported source colors by perceptual distance, not RGB sums.
  const selected: Color[] = [source];
  while (selected.length < 3) {
    const options = candidates.filter(c => selected.every(lab => distance(c.lab, lab) >= .12));
    options.sort((a, b) => {
      const separation = (c: typeof a) => Math.min(...selected.map(lab => distance(c.lab, lab))) * Math.sqrt(c.count / total);
      return separation(b) - separation(a);
    });
    if (!options.length) break;
    selected.push(options[0].lab);
  }
  // Missing hues become tonal variations of the cover, never a synthetic rainbow.
  const waves = [0, 1, 2].map(i => {
    const lab = selected[i] || source;
    const light = selected[i] ? Math.max(.48, Math.min(.8, lab[0])) : i === 1 ? .42 : .78;
    return gamut(light, lab).map(n => n / 255) as Color;
  }) as TrackPalette["waves"];
  // Lightning Fall gets up to five supported colors, without changing the
  // three-color palette used by the other backdrops. Compare display colors
  // so lifting dark pigments does not introduce near-duplicate trail colors.
  const trailColor = (lab: Color): Color => gamut(Math.max(.48, Math.min(.8, lab[0])), lab);
  const trailLabs: Color[] = [toLab(trailColor(source))];
  const trails: Color[] = [trailColor(source).map(n => n / 255) as Color];
  const trailCandidates = candidates.map(c => ({ ...c, rgb: trailColor(c.lab), displayLab: toLab(trailColor(c.lab)) }));
  while (trails.length < 5) {
    const options = trailCandidates.filter(c => trailLabs.every(lab => distance(c.displayLab, lab) >= .08));
    options.sort((a, b) => {
      const separation = (c: typeof a) => Math.min(...trailLabs.map(lab => distance(c.displayLab, lab))) * Math.sqrt(c.count / total);
      return separation(b) - separation(a);
    });
    if (!options.length) break;
    trailLabs.push(options[0].displayLab);
    trails.push(options[0].rgb.map(n => n / 255) as Color);
  }
  return { accent, background, waves, trails };
}
