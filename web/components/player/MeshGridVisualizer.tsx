"use client";

import type { TrackPalette } from "./artworkPalette";
import type { VisualFrame } from "./visualFeatures";
import * as THREE from "three";
import { useEffect, useRef } from "react";
import { readPlaybackPreferences, type PlaybackPreferences } from "./playbackPreferences";
import { SongJourney } from "./songJourney";
import styles from "./MeshGridVisualizer.module.css";

type Color = [number, number, number];
const SIZE = 80;
const SEGMENTS = 180;
const CELL = SIZE / SEGMENTS;
// Landscape per scene: [hill height, ridge sharpness, corridor width].
const LANDSCAPES: [number, number, number][] = [
  [0.9, 0, 5],
  [2.2, 0.2, 4],
  [1.4, 0.9, 6],
  [3.2, 0.45, 3.5],
  [0.5, 0.1, 8],
  [2.6, 1, 4.5],
];

const vertexShader = `
  attribute vec3 barycentric;
  uniform float scroll, travel, bass, mid, treble, pulse, impact, anticipation;
  uniform vec3 landFrom, landTo;
  uniform float landBlend;
  varying vec3 vBarycentric;
  varying float vCrest, vDistance, vDepth, vHeight;
  float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
  float hills(vec2 w, vec3 land) {
    float h = sin(w.x * .21 + sin(w.y * .13) * 1.7) + sin(w.y * .17 + w.x * .05) * .8 + sin((w.x + w.y) * .31) * .35;
    float ridges = 1.0 - abs(sin(w.x * .12 + sin(w.y * .09) * 2.0));
    h = mix(h, ridges * 2.0 - .6, land.y);
    // A flat corridor ahead, so the flight path stays open.
    float corridor = smoothstep(land.z * .4, land.z * 1.6, abs(w.x));
    return h * land.x * (.25 + corridor * .75);
  }
  void main() {
    // The mesh slides toward the camera one cell at a time while the terrain stays
    // fixed in the world, so the flight is endless and the jitter never pops.
    float rows = floor(scroll / ${CELL.toFixed(6)});
    float slide = scroll - rows * ${CELL.toFixed(6)};
    vec2 ground = position.xy;
    vec2 cellId = vec2(floor(ground.x / ${CELL.toFixed(6)} + .5), floor(ground.y / ${CELL.toFixed(6)} + .5) - rows);
    ground += (vec2(hash(cellId), hash(cellId + 17.3)) - .5) * ${(CELL * 0.5).toFixed(6)};
    ground.y += slide;
    vec2 world = vec2(ground.x, ground.y - scroll);
    float land = mix(hills(world, landFrom), hills(world, landTo), landBlend);
    vec2 focus = ground - vec2(0.0, -6.0);
    float radius = length(focus);
    float ring = sin(radius * 1.2 - travel * 6.0);
    float crest = pow(.5 + .5 * ring, 5.0);
    float rough = sin(world.x * 3.7 + world.y * 2.1 + travel) * cos(world.y * 4.3 - world.x * 1.4 - travel * .7);
    float height = land - ground.y * ground.y * .0012 - pow(abs(ground.x), 2.0) * .0018;
    height += ring * (.06 + bass * .7 + pulse * .35);
    height += rough * (.02 + mid * .3 + treble * .14);
    vec4 view = modelViewMatrix * vec4(ground.x, height, ground.y, 1.0);
    vBarycentric = barycentric;
    vCrest = crest;
    vDistance = -view.z;
    vDepth = world.y;
    vHeight = land;
    gl_Position = projectionMatrix * view;
  }
`;

const fragmentShader = `
  uniform vec3 colorA, colorB, colorC;
  uniform float drive, treble, visibility, pulse, anticipation;
  varying vec3 vBarycentric;
  varying float vCrest, vDistance, vDepth, vHeight;
  void main() {
    vec3 pixelDistance = vBarycentric / max(fwidth(vBarycentric), vec3(.00001));
    float edge = min(pixelDistance.x, min(pixelDistance.y, pixelDistance.z));
    float core = 1.0 - smoothstep(.25, 1.0, edge);
    float halo = exp(-edge * .85) * .22;
    // Colour bands are fixed to the land, so they stream toward you as you fly.
    float band = fract(vDepth * .045);
    vec3 color = mix(colorA, colorB, smoothstep(.2, .8, abs(band * 2.0 - 1.0)));
    color = mix(color, colorC, clamp(vHeight * .25, 0.0, .6));
    float light = .2 + vCrest * (.3 + drive * .5) + treble * .14 + pulse * .12;
    light *= .75 + drive * .55;
    float fog = 1.0 - smoothstep(18.0, 60.0, vDistance);
    vec3 rgb = color * (core + halo) * light * fog;
    gl_FragColor = vec4(rgb * visibility, 1.0);
  }
`;

const skyShader = `
  uniform vec3 colorA, colorC;
  uniform float drive, impact, anticipation, horizon, travel, treble, visibility;
  varying vec2 vUv;
  float hash(vec2 p) { return fract(sin(dot(p, vec2(41.3, 289.1))) * 43758.5453); }
  void main() {
    float above = vUv.y - horizon;
    // A glow on the horizon: where the journey is heading. It brightens with the
    // song's push and flares as a bigger section lands.
    float glow = exp(-abs(above) * (9.0 - anticipation * 4.0)) * (.08 + drive * .22 + impact * .15);
    float sun = exp(-length((vUv - vec2(.5, horizon)) * vec2(3.2, 7.0))) * (.05 + drive * .12 + anticipation * .15);
    vec3 rgb = mix(colorC, colorA, .3) * glow + mix(colorC, vec3(1.0), .3) * sun;
    // Stars drift by slowly above the horizon and twinkle with the treble.
    vec2 grid = vec2(vUv.x * 90.0 + travel * .4, vUv.y * 50.0);
    vec2 id = floor(grid);
    float star = step(.985, hash(id)) * smoothstep(.0, .2, above);
    float twinkle = .5 + .5 * sin(travel * 3.0 + hash(id + 3.0) * 30.0);
    float point = exp(-length(fract(grid) - .5) * 9.0);
    rgb += vec3(.8, .85, 1.0) * star * point * twinkle * (.12 + treble * .3);
    gl_FragColor = vec4(rgb * visibility, 1.0);
  }
`;

/** Flight over endless triangulated terrain toward a horizon light, through the song's scenes. */
export default function MeshGridVisualizer() {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)");
    const compact = window.matchMedia("(max-width:1199px), (max-height:719px)");
    let preferences = readPlaybackPreferences();
    let renderer: THREE.WebGLRenderer | null = null;
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(58, 1, 0.1, 100);
    const base = new THREE.PlaneGeometry(SIZE, SIZE, SEGMENTS, SEGMENTS);
    const geometry = base.toNonIndexed();
    base.dispose();
    const barycentric = new Float32Array(geometry.getAttribute("position").count * 3);
    for (let i = 0; i < barycentric.length / 3; i++) barycentric[i * 3 + (i % 3)] = 1;
    geometry.setAttribute("barycentric", new THREE.BufferAttribute(barycentric, 3));
    const colors = {
      colorA: new THREE.Vector3(0.48, 0.98, 0.92),
      colorB: new THREE.Vector3(0.76, 0.66, 1),
      colorC: new THREE.Vector3(0.55, 0.8, 1),
    };
    const uniforms = {
      ...(Object.fromEntries(
        Object.entries(colors).map(([name, value]) => [name, { value: value.clone() }]),
      ) as Record<keyof typeof colors, { value: THREE.Vector3 }>),
      scroll: { value: 0 },
      travel: { value: 0 },
      drive: { value: 0 },
      visibility: { value: 0.38 },
      bass: { value: 0 },
      mid: { value: 0 },
      treble: { value: 0 },
      pulse: { value: 0 },
      impact: { value: 0 },
      anticipation: { value: 0 },
      landFrom: { value: new THREE.Vector3(...LANDSCAPES[0]) },
      landTo: { value: new THREE.Vector3(...LANDSCAPES[0]) },
      landBlend: { value: 1 },
    };
    const material = new THREE.ShaderMaterial({
      depthTest: true,
      depthWrite: true,
      side: THREE.DoubleSide,
      uniforms,
      vertexShader,
      fragmentShader,
    });
    const mesh = new THREE.Mesh(geometry, material);
    mesh.frustumCulled = false;
    scene.add(mesh);
    const skyUniforms = {
      colorA: uniforms.colorA,
      colorC: uniforms.colorC,
      drive: uniforms.drive,
      impact: uniforms.impact,
      anticipation: uniforms.anticipation,
      travel: uniforms.travel,
      treble: uniforms.treble,
      visibility: uniforms.visibility,
      horizon: { value: 0.72 },
    };
    const skyGeometry = new THREE.PlaneGeometry(2, 2);
    const sky = new THREE.Mesh(
      skyGeometry,
      new THREE.ShaderMaterial({
        depthTest: false,
        depthWrite: false,
        uniforms: skyUniforms,
        fragmentShader: skyShader,
        vertexShader:
          "varying vec2 vUv; void main(){vUv=uv;gl_Position=vec4(position.xy,0.9999,1.0);}",
      }),
    );
    sky.frustumCulled = false;
    sky.renderOrder = -1;
    scene.add(sky);
    const journey = new SongJourney();
    let palette: Color[] = [
      [0.48, 0.98, 0.92],
      [0.76, 0.66, 1],
      [0.55, 0.8, 1],
    ];
    let frame = 0,
      lastFrame = 0,
      lastAudio = -Infinity,
      playing = true,
      lift = 0;
    let active = false,
      failed = false,
      fade = 0;
    let targetBass = 0,
      targetMid = 0,
      targetTreble = 0;
    const resize = () => {
      if (!renderer) return;
      const bounds = canvas.getBoundingClientRect();
      renderer.setPixelRatio(Math.min(1.5, window.devicePixelRatio || 1));
      renderer.setSize(Math.max(1, bounds.width), Math.max(1, bounds.height), false);
      camera.aspect = Math.max(1, bounds.width) / Math.max(1, bounds.height);
      camera.updateProjectionMatrix();
    };
    const draw = (now: number) => {
      if (!active || !renderer) return;
      frame = requestAnimationFrame(draw);
      const interval =
        preferences.waveFrameRate === "uncapped" ? 0 : 1000 / Number(preferences.waveFrameRate);
      if (now - lastFrame < interval) return;
      const dt = Math.min(0.05, (now - lastFrame) / 1000 || 0.016);
      lastFrame = now;
      const audible = playing && now - lastAudio < 700;
      // A reversible 800ms fade. Smoothstep softens both ends without a jump
      // when playback toggles again halfway through a transition.
      fade = THREE.MathUtils.clamp(fade + ((audible ? 1 : -1) * dt) / 0.8, 0, 1);
      uniforms.visibility.value = 0.38 + 0.62 * fade * fade * (3 - 2 * fade);
      const rate =
        preferences.animationSpeed === "slow"
          ? 0.65
          : preferences.animationSpeed === "fast"
            ? 1.45
            : 1;
      journey.step(dt, rate, now / 1000);
      for (const [name, target] of [
        ["bass", targetBass],
        ["mid", targetMid],
        ["treble", targetTreble],
      ] as const) {
        const current = uniforms[name].value,
          next = audible ? target : 0;
        // Fast attack preserves beats; slower release avoids strobing.
        uniforms[name].value += (next - current) * (1 - Math.exp(-dt * (next > current ? 18 : 5)));
      }
      uniforms.travel.value = journey.travel;
      uniforms.scroll.value = journey.travel * 6;
      uniforms.drive.value = journey.drive;
      uniforms.pulse.value = journey.beatPulse * preferences.bassReactivity;
      uniforms.impact.value = journey.impact;
      uniforms.anticipation.value = journey.anticipation;
      const blend = journey.sceneBlend * journey.sceneBlend * (3 - 2 * journey.sceneBlend);
      uniforms.landFrom.value.set(...LANDSCAPES[journey.previousScene % LANDSCAPES.length]);
      uniforms.landTo.value.set(...LANDSCAPES[journey.scene % LANDSCAPES.length]);
      uniforms.landBlend.value = blend;
      (["colorA", "colorB", "colorC"] as const).forEach((name, slot) =>
        uniforms[name].value.lerp(
          new THREE.Vector3(...journey.color(palette, slot)),
          1 - Math.exp(-dt * 2.5),
        ),
      );
      // The camera climbs while a build-up gathers, dips as it lands, and banks with the mids.
      // Eased, so a section change never jolts the camera.
      lift += (journey.anticipation * 2.6 - lift) * (1 - Math.exp(-dt * 1.5));
      const bank = Math.sin(journey.travel * 0.35) * (0.4 + journey.drive * 0.8);
      camera.position.set(bank, 4.6 + lift, 18);
      camera.lookAt(bank * 0.4, -2 + lift * 0.35, -7);
      camera.rotation.z += -bank * 0.03;
      renderer.render(scene, camera);
    };
    const sync = () => {
      const next =
        !failed &&
        preferences.wavesEnabled &&
        preferences.backdropPreset === "meshgrid" &&
        !compact.matches &&
        !reduced.matches &&
        !document.hidden;
      canvas.style.display = next ? "block" : "none";
      if (next && !renderer) {
        try {
          renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: false });
          resize();
        } catch {
          failed = true;
          canvas.style.display = "none";
          return;
        }
      }
      if (next === active) return;
      active = next;
      cancelAnimationFrame(frame);
      if (active) {
        lastFrame = performance.now();
        frame = requestAnimationFrame(draw);
      }
    };
    const receivePreferences = (event: Event) => {
      preferences = (event as CustomEvent<PlaybackPreferences>).detail;
      sync();
    };
    const receiveAudio = (event: Event) => {
      const detail = (event as CustomEvent<VisualFrame>).detail;
      journey.receive(detail, preferences);
      if (!detail.active) {
        reset();
        return;
      }
      playing = true;
      targetBass = Math.min(1, detail.bass * preferences.bassReactivity * 1.4);
      targetMid = Math.min(1, detail.mid * preferences.vocalReactivity * 1.5);
      targetTreble = Math.min(1, detail.treble * preferences.trebleReactivity * 1.8);
      lastAudio = performance.now();
    };
    const receivePalette = (event: Event) => {
      const { palette: next } = (event as CustomEvent<{ palette: TrackPalette | null }>).detail;
      // Pause publishes null. Keep the current artwork colors until a new palette arrives.
      if (next) palette = next.trails?.length >= 3 ? next.trails : next.waves;
    };
    const receiveState = (event: Event) => {
      playing = Boolean((event as CustomEvent<boolean>).detail);
    };
    const reset = () => {
      targetBass = targetMid = targetTreble = 0;
      lastAudio = -Infinity;
      // Neutral frames arrive repeatedly while paused. Let the render loop
      // release the bands, pulse, and speed instead of cutting them to zero.
    };
    const changeTrack = () => {
      reset();
      journey.reset();
    };
    const events: [string, EventListener][] = [
      ["echora:playback-preferences", receivePreferences],
      ["echora:visual-frame", receiveAudio],
      ["echora:track-palette", receivePalette],
      ["echora:playback-state", receiveState],
      ["echora:track-change", changeTrack],
      ["resize", resize],
    ];
    events.forEach(([name, handler]) => window.addEventListener(name, handler));
    compact.addEventListener("change", sync);
    reduced.addEventListener("change", sync);
    document.addEventListener("visibilitychange", sync);
    sync();
    window.dispatchEvent(new Event("echora:playback-state-request"));
    return () => {
      cancelAnimationFrame(frame);
      events.forEach(([name, handler]) => window.removeEventListener(name, handler));
      compact.removeEventListener("change", sync);
      reduced.removeEventListener("change", sync);
      document.removeEventListener("visibilitychange", sync);
      geometry.dispose();
      material.dispose();
      skyGeometry.dispose();
      sky.material.dispose();
      renderer?.dispose();
    };
  }, []);
  return <canvas ref={ref} className={styles.signal} aria-hidden="true" />;
}
