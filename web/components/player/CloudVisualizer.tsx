"use client";

import type { VisualFrame } from "./visualFeatures";
import * as THREE from "three";
import { useEffect, useRef } from "react";
import { readPlaybackPreferences, type PlaybackPreferences } from "./playbackPreferences";
import { CloudResponse } from "./cloudResponse";
import { cloudFragmentShader } from "./cloudShader";
import { SongJourney } from "./songJourney";
import styles from "./CloudVisualizer.module.css";

export default function CloudVisualizer() {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const compact = window.matchMedia("(max-width:1199px), (max-height:719px)");
    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)");
    let preferences = readPlaybackPreferences();
    let renderer: THREE.WebGLRenderer | null = null;
    let active = false,
      failed = false,
      playing = false,
      frame = 0,
      lastFrame = 0;
    const response = new CloudResponse();
    const targetTint = new THREE.Vector3(0.38, 0.48, 0.63);
    const targetSecondaryTint = new THREE.Vector3(0.38, 0.34, 0.52);
    const scene = new THREE.Scene();
    const camera = new THREE.Camera();
    const geometry = new THREE.PlaneGeometry(2, 2);
    const material = new THREE.ShaderMaterial({
      depthTest: false,
      depthWrite: false,
      uniforms: {
        resolution: { value: new THREE.Vector2(1, 1) },
        tint: { value: targetTint.clone() },
        secondaryTint: { value: targetSecondaryTint.clone() },
        travel: { value: 0 },
        bass: { value: 0 },
        mid: { value: 0 },
        treble: { value: 0 },
        glow: { value: 0 },
        strike: { value: 0 },
        seed: { value: 1 },
        flight: { value: 0 },
        skyLevel: { value: 0 },
        skyTint: { value: new THREE.Vector3(0.4, 0.5, 0.7) },
      },
      vertexShader: "varying vec2 vUv; void main(){vUv=uv;gl_Position=vec4(position.xy,0.0,1.0);}",
      fragmentShader: cloudFragmentShader,
    });
    const journey = new SongJourney();
    let palette: [number, number, number][] = [
      [0.38, 0.48, 0.63],
      [0.38, 0.34, 0.52],
      [0.5, 0.6, 0.8],
    ];
    let lastImpact = 0;
    const mesh = new THREE.Mesh(geometry, material);
    mesh.frustumCulled = false;
    scene.add(mesh);
    const resize = () => {
      if (!renderer) return;
      const bounds = canvas.getBoundingClientRect();
      // More pixels resolve eroded edges without paying for retina-sized ray marches.
      // Bound both dimensions so tall displays cannot allocate an oversized target.
      const scale = Math.min(
        0.95,
        1440 / Math.max(1, bounds.width),
        1000 / Math.max(1, bounds.height),
      );
      renderer.setPixelRatio(1);
      renderer.setSize(
        Math.max(1, Math.round(bounds.width * scale)),
        Math.max(1, Math.round(bounds.height * scale)),
        false,
      );
      renderer.getDrawingBufferSize(material.uniforms.resolution.value);
    };
    const draw = (now: number) => {
      if (!active || !renderer) return;
      frame = requestAnimationFrame(draw);
      const interval =
        preferences.waveFrameRate === "uncapped" ? 0 : 1000 / Number(preferences.waveFrameRate);
      if (now - lastFrame < interval) return;
      const dt = Math.min(0.1, (now - lastFrame) / 1000);
      lastFrame = now;
      const rate =
        preferences.animationSpeed === "slow"
          ? 0.65
          : preferences.animationSpeed === "fast"
            ? 1.45
            : 1;
      response.step(dt, now / 1000, playing, rate);
      journey.step(dt, rate, now / 1000);
      // A build-up charges light inside the banks; the landing releases a storm.
      if (journey.impact > lastImpact + 0.2) {
        response.strike = Math.max(response.strike, 0.85);
        response.seed += 7.13;
      }
      lastImpact = journey.impact;
      response.glow = Math.max(response.glow, journey.anticipation * 0.7);
      for (const key of ["travel", "bass", "mid", "treble", "glow", "strike", "seed"] as const)
        material.uniforms[key].value = response[key];
      material.uniforms.flight.value = journey.travel * 0.2;
      material.uniforms.skyLevel.value =
        0.02 +
        journey.drive * journey.sectionEnergy * 0.16 +
        journey.anticipation * 0.06 +
        journey.impact * 0.1;
      // Scenes recolour the clouds; the artwork palette still decides which colours.
      targetTint.set(...journey.color(palette, 0));
      targetSecondaryTint.set(...journey.color(palette, 1));
      material.uniforms.skyTint.value.lerp(
        new THREE.Vector3(...journey.color(palette, 2)),
        1 - Math.exp(-dt * 0.8),
      );
      material.uniforms.tint.value.lerp(targetTint, 1 - Math.exp(-dt * 0.6));
      material.uniforms.secondaryTint.value.lerp(targetSecondaryTint, 1 - Math.exp(-dt * 0.6));
      renderer.render(scene, camera);
    };
    const sync = () => {
      const next =
        !failed &&
        preferences.wavesEnabled &&
        preferences.backdropPreset === "clouds" &&
        !compact.matches &&
        !reduced.matches &&
        !document.hidden;
      canvas.style.display = next ? "block" : "none";
      if (next && !renderer) {
        try {
          renderer = new THREE.WebGLRenderer({
            canvas,
            alpha: false,
            antialias: false,
            powerPreference: "low-power",
          });
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
      response.reset();
      if (active) {
        resize();
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
        response.reset();
        return;
      }
      playing = true;
      if (active) response.receive(detail, preferences, performance.now() / 1000);
    };
    const receivePalette = (event: Event) => {
      const detail = (
        event as CustomEvent<{ active: boolean; palette: { waves: number[][] } | null }>
      ).detail;
      // Pause clears the shared accent. Keep cloud pigments until new artwork arrives.
      if (detail.palette?.waves[0])
        palette = (
          detail.palette.waves.length >= 3
            ? detail.palette.waves
            : [
                detail.palette.waves[0],
                detail.palette.waves[1] || detail.palette.waves[0],
                detail.palette.waves[0],
              ]
        ) as [number, number, number][];
    };
    const receiveState = (event: Event) => {
      playing = Boolean((event as CustomEvent<boolean>).detail);
      if (!playing) response.reset();
    };
    const reset = () => {
      response.reset();
      journey.reset();
    };
    const lost = (event: Event) => {
      event.preventDefault();
      failed = true;
      sync();
    };
    const restored = () => {
      failed = false;
      sync();
    };
    const events: [string, EventListener][] = [
      ["echora:playback-preferences", receivePreferences],
      ["echora:visual-frame", receiveAudio],
      ["echora:track-palette", receivePalette],
      ["echora:playback-state", receiveState],
      ["echora:track-change", reset],
      ["resize", resize],
    ];
    events.forEach(([name, handler]) => window.addEventListener(name, handler));
    // Provider effects register after this sibling backdrop. Defer one turn so
    // the request can retrieve the current state even when playback began first.
    const stateRequest = window.setTimeout(
      () => window.dispatchEvent(new Event("echora:playback-state-request")),
      0,
    );
    compact.addEventListener("change", sync);
    reduced.addEventListener("change", sync);
    document.addEventListener("visibilitychange", sync);
    canvas.addEventListener("webglcontextlost", lost);
    canvas.addEventListener("webglcontextrestored", restored);
    sync();
    return () => {
      cancelAnimationFrame(frame);
      window.clearTimeout(stateRequest);
      events.forEach(([name, handler]) => window.removeEventListener(name, handler));
      compact.removeEventListener("change", sync);
      reduced.removeEventListener("change", sync);
      document.removeEventListener("visibilitychange", sync);
      canvas.removeEventListener("webglcontextlost", lost);
      canvas.removeEventListener("webglcontextrestored", restored);
      geometry.dispose();
      material.dispose();
      renderer?.dispose();
    };
  }, []);
  // Backdrop's generic canvas rule is deliberately broad. Start hidden so this
  // opaque layer cannot cover the existing visualizers before sync enables it.
  return (
    <canvas ref={ref} className={styles.clouds} style={{ display: "none" }} aria-hidden="true" />
  );
}
