"use client";

import * as THREE from "three";
import { useEffect, useRef } from "react";
import { readPlaybackPreferences, type PlaybackPreferences } from "./playbackPreferences";
import { neutralVisualFrame, type VisualFrame } from "./visualFeatures";
import type { TrackPalette } from "./artworkPalette";
import { WaterSurface, waterControls } from "./waterPhysics";
import { SongJourney } from "./songJourney";
import styles from "./WaterDropVisualizer.module.css";

const vertexShader = `
  uniform sampler2D heights;
  uniform vec2 texel;
  varying vec3 vNormal, vWorld;
  varying float vHeight;
  void main() {
    float h = texture2D(heights, uv).r;
    float dx = texture2D(heights, uv + vec2(texel.x, 0.)).r - texture2D(heights, uv - vec2(texel.x, 0.)).r;
    float dz = texture2D(heights, uv + vec2(0., texel.y)).r - texture2D(heights, uv - vec2(0., texel.y)).r;
    vNormal = normalize(vec3(-dx * 4.8, 1., -dz * 4.8));
    vHeight = h;
    vec4 world = modelMatrix * vec4(position.x, h * .55, position.y, 1.);
    vWorld = world.xyz;
    gl_Position = projectionMatrix * viewMatrix * world;
  }
`;
const fragmentShader = `
  uniform vec3 colorA, colorB, colorC, colorD, colorE;
  uniform float energy, colorCount;
  varying vec3 vNormal, vWorld;
  varying float vHeight;
  vec3 palette(float index) {
    return index < 1. ? colorA : index < 2. ? colorB : index < 3. ? colorC : index < 4. ? colorD : colorE;
  }
  void main() {
    vec3 n = normalize(vNormal);
    vec3 view = normalize(cameraPosition - vWorld);
    float fresnel = pow(1. - max(0., dot(n, view)), 3.);
    vec3 light = normalize(vec3(-.4, .8, -.5));
    float specular = pow(max(0., dot(n, normalize(light + view))), 80.);
    float slope = length(n.xz);
    float crest = smoothstep(.025, .24, slope);
    float reflection = .5 + .5 * dot(reflect(-view, n), normalize(vec3(.2, .3, -.9)));
    float region = clamp((vWorld.x / 48. + .5) * .65 + (vWorld.z / 30. + .5) * .35 + (reflection - .5) * .2, 0., 1.) * (colorCount - 1.);
    vec3 tint = mix(palette(floor(region)), palette(min(floor(region) + 1., colorCount - 1.)), smoothstep(0., 1., fract(region)));
    vec3 base = vec3(.003, .005, .008) + tint * (.006 + fresnel * .045);
    vec3 ripples = tint * crest * (.32 + energy * .3);
    vec3 shine = mix(tint, vec3(1.), .3) * specular * (.25 + crest);
    float edge = 1. - smoothstep(.78, 1., max(abs(vWorld.x) / 24., abs(vWorld.z) / 15.));
    gl_FragColor = vec4(base + (ripples + shine) * edge, 1.);
  }
`;

type Drop = { x: number; z: number; y: number; velocity: number; radius: number; strength: number };
const MAX_DROPS = 40;

/** Falling drops feed a shared wave simulation, not independently drawn ripple rings. */
export default function WaterDropVisualizer() {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)");
    const compact = window.matchMedia("(max-width:1199px), (max-height:719px)");
    let preferences = readPlaybackPreferences();
    let renderer: THREE.WebGLRenderer | null = null;
    let failed = false,
      active = false,
      animation = 0,
      previous = 0;
    let audio = neutralVisualFrame(),
      lastAudio = -Infinity,
      playing = false;
    let pendingHit = 0,
      spawnCredit = 0,
      lastSpawn = -Infinity;
    const surface = new WaterSurface();
    const texture = new THREE.DataTexture(
      surface.height,
      surface.width,
      surface.depth,
      THREE.RedFormat,
      THREE.FloatType,
    );
    texture.minFilter = THREE.NearestFilter;
    texture.magFilter = THREE.NearestFilter;
    texture.needsUpdate = true;
    const colors = [
      new THREE.Vector3(0.48, 0.98, 0.92),
      new THREE.Vector3(0.76, 0.66, 1),
      new THREE.Vector3(0.55, 0.8, 1),
      new THREE.Vector3(0.48, 0.98, 0.92),
      new THREE.Vector3(0.76, 0.66, 1),
    ];
    const colorNames = ["colorA", "colorB", "colorC", "colorD", "colorE"] as const;
    let colorCount = 3;
    let artwork: [number, number, number][] = colors.map(
      (color) => color.toArray() as [number, number, number],
    );
    const journey = new SongJourney();
    let lastImpact = 0;
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(48, 1, 0.1, 160);
    camera.position.set(0, 24, 27);
    camera.lookAt(0, 0, 0);
    const geometry = new THREE.PlaneGeometry(48, 30, surface.width - 1, surface.depth - 1);
    const material = new THREE.ShaderMaterial({
      uniforms: {
        heights: { value: texture },
        texel: { value: new THREE.Vector2(1 / surface.width, 1 / surface.depth) },
        colorA: { value: colors[0].clone() },
        colorB: { value: colors[1].clone() },
        colorC: { value: colors[2].clone() },
        colorD: { value: colors[3].clone() },
        colorE: { value: colors[4].clone() },
        colorCount: { value: 3 },
        energy: { value: 0 },
      },
      vertexShader,
      fragmentShader,
      side: THREE.DoubleSide,
    });
    const water = new THREE.Mesh(geometry, material);
    water.frustumCulled = false;
    scene.add(water);
    // Drops remain in the simulation but are deliberately not rendered.
    const drops: Drop[] = [];
    let tension = 0.15,
      damping = 0.984;

    const resize = () => {
      if (!renderer) return;
      const bounds = canvas.getBoundingClientRect();
      renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.5));
      renderer.setSize(Math.max(1, bounds.width), Math.max(1, bounds.height), false);
      camera.aspect = Math.max(1, bounds.width) / Math.max(1, bounds.height);
      camera.updateProjectionMatrix();
    };
    const clear = () => {
      journey.reset();
      audio = neutralVisualFrame();
      lastAudio = -Infinity;
      pendingHit = 0;
      spawnCredit = 0;
      drops.length = 0;
      surface.clear();
      texture.needsUpdate = true;
    };
    const draw = (now: number) => {
      if (!active || !renderer) return;
      animation = requestAnimationFrame(draw);
      const interval =
        preferences.waveFrameRate === "uncapped" ? 0 : 1000 / Number(preferences.waveFrameRate);
      if (now - previous < interval) return;
      const dt = Math.min(0.05, (now - previous) / 1000 || 0.016);
      previous = now;
      const audible = playing && now - lastAudio < 700 && audio.active;
      const controls = waterControls(audible ? audio : neutralVisualFrame(), preferences);
      const rate =
        preferences.animationSpeed === "slow"
          ? 0.65
          : preferences.animationSpeed === "fast"
            ? 1.45
            : 1;
      const seconds = dt * rate;
      journey.step(dt, rate, now / 1000);
      // A bigger section landing drops a stone in the middle of the pool.
      if (audible && journey.impact > lastImpact + 0.2) {
        drops.push({
          x: (Math.random() - 0.5) * 8,
          z: (Math.random() - 0.5) * 5,
          y: 9,
          velocity: controls.fallSpeed + 8,
          radius: controls.radius * 2.6,
          strength: controls.impact * 2.4 + 0.3,
        });
      }
      lastImpact = journey.impact;
      tension += (controls.tension - tension) * (1 - Math.exp(-dt * 4));
      damping += (controls.damping - damping) * (1 - Math.exp(-dt * 4));
      // The song's push thickens the rain; a build-up quickens it further.
      spawnCredit = audible
        ? Math.min(
            1.5,
            spawnCredit +
              seconds * controls.rate * (0.5 + journey.drive + journey.anticipation * 1.5),
          )
        : 0;
      if (
        audible &&
        controls.energy > 0.015 &&
        now - lastSpawn > 130 &&
        (pendingHit > 0.05 || spawnCredit >= 1) &&
        drops.length < MAX_DROPS
      ) {
        const hit = pendingHit;
        drops.push({
          x: (Math.random() - 0.5) * 36,
          z: (Math.random() - 0.5) * 22,
          y: 7 + Math.random() * 6,
          velocity: controls.fallSpeed + hit * 5,
          radius: controls.radius,
          strength: controls.impact + hit * 0.15,
        });
        spawnCredit = Math.max(0, spawnCredit - 1);
        lastSpawn = now;
      }
      pendingHit = 0;
      // Already-falling drops finish their impacts after pause; no new drops spawn.
      for (let i = drops.length - 1; i >= 0; i--) {
        const drop = drops[i];
        drop.velocity += controls.gravity * seconds;
        drop.y -= drop.velocity * seconds;
        if (drop.y <= 0) {
          surface.impact(
            drop.x / 48 + 0.5,
            drop.z / 30 + 0.5,
            drop.radius * 1.5,
            drop.strength * Math.min(1.5, drop.velocity / 16),
          );
          drops.splice(i, 1);
        }
      }
      surface.advance(seconds, tension, damping);
      texture.needsUpdate = true;
      const response = 1 - Math.exp(-dt * 2);
      colors.forEach((color, index) => color.set(...journey.color(artwork, index)));
      colorNames.forEach((name, index) =>
        material.uniforms[name].value.lerp(colors[index], response),
      );
      material.uniforms.energy.value +=
        (controls.energy * (0.6 + journey.drive * 0.8) - material.uniforms.energy.value) * response;
      renderer.render(scene, camera);
    };
    const sync = () => {
      const next =
        !failed &&
        preferences.wavesEnabled &&
        preferences.backdropPreset === "waterdrops" &&
        !compact.matches &&
        !reduced.matches &&
        !document.hidden;
      canvas.style.display = next ? "block" : "none";
      if (next && !renderer) {
        try {
          renderer = new THREE.WebGLRenderer({ canvas, antialias: false, alpha: false });
          renderer.setClearColor(0x020408);
          resize();
        } catch {
          failed = true;
          canvas.style.display = "none";
          return;
        }
      }
      if (next === active) return;
      active = next;
      cancelAnimationFrame(animation);
      if (active) {
        previous = performance.now();
        pendingHit = 0;
        spawnCredit = 0;
        animation = requestAnimationFrame(draw);
      } else clear();
    };
    const receiveAudio = (event: Event) => {
      const next = (event as CustomEvent<VisualFrame>).detail;
      audio = next;
      journey.receive(next, preferences);
      if (!active || !next.active) {
        pendingHit = 0;
        lastAudio = -Infinity;
        return;
      }
      playing = true;
      lastAudio = performance.now();
      const controls = waterControls(next, preferences);
      pendingHit = Math.max(
        pendingHit,
        controls.attack,
        next.beat || next.onset ? controls.energy * 0.65 : 0,
      );
    };
    const receivePalette = (event: Event) => {
      const palette = (event as CustomEvent<{ palette: TrackPalette | null }>).detail.palette;
      if (palette) {
        const extracted = palette.trails?.length ? palette.trails : palette.waves;
        colorCount = Math.min(5, extracted.length);
        material.uniforms.colorCount.value = colorCount;
        artwork = extracted.slice(0, 5) as [number, number, number][];
      }
    };
    const receiveState = (event: Event) => {
      playing = Boolean((event as CustomEvent<boolean>).detail);
      if (!playing) {
        pendingHit = 0;
        spawnCredit = 0;
      }
    };
    const receivePreferences = (event: Event) => {
      preferences = (event as CustomEvent<PlaybackPreferences>).detail;
      sync();
    };
    const events: [string, EventListener][] = [
      ["echora:visual-frame", receiveAudio],
      ["echora:track-palette", receivePalette],
      ["echora:playback-state", receiveState],
      ["echora:track-change", clear],
      ["echora:playback-preferences", receivePreferences],
      ["resize", resize],
    ];
    events.forEach(([name, handler]) => window.addEventListener(name, handler));
    reduced.addEventListener("change", sync);
    compact.addEventListener("change", sync);
    document.addEventListener("visibilitychange", sync);
    sync();
    window.dispatchEvent(new Event("echora:playback-state-request"));
    return () => {
      cancelAnimationFrame(animation);
      events.forEach(([name, handler]) => window.removeEventListener(name, handler));
      reduced.removeEventListener("change", sync);
      compact.removeEventListener("change", sync);
      document.removeEventListener("visibilitychange", sync);
      texture.dispose();
      geometry.dispose();
      material.dispose();
      renderer?.dispose();
    };
  }, []);
  return <canvas ref={ref} className={styles.water} aria-hidden="true" />;
}
