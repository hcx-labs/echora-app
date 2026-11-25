"use client";

import { setUrl } from "../shell/urlState";
import Artwork from "../ui/Artwork";
import { Info, ListFilter, LocateFixed, Pause, Play, Route, ScanSearch, Search, X } from "lucide-react";
import { coverArtUrl } from "../media/coverArt"
import { mediaUrl } from "../media/mediaOrigin";
import { PointerEvent, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { usePlayer } from "../player/PlayerProvider";
import TransitionLink from "../shell/TransitionLink";
import { Button } from "../ui/button";
import { Input } from "../ui/input";
import { Tabs, TabsList, TabsTrigger } from "../ui/tabs";
import { Checkbox } from "../ui/checkbox";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "../ui/tooltip";
import { Spinner } from "../ui/spinner";
import GalaxyPanel from "./GalaxyPanel";
import styles from "./MusicGalaxy.module.css";

type Neighbor = { id: string; title: string; artist?: string; similarity: number };
type Point = { id: string; title: string; artist?: string; album?: string; duration_seconds: number; source_id?: string; cover_art?: string; x: number; y: number; cluster: number; cluster_affinity: number; cluster_memberships: { cluster: number; strength: number }[]; bridge_score: number; neighbors: Neighbor[] };
type Community = { id: number; label: string; x: number; y: number; size: number; cohesion: number; top_genres: { name: string; tracks: number }[]; representative_tracks: { id: string; title: string; artist?: string }[] };
type Edge = { source_id: string; target_id: string; similarity: number; mutual: boolean; cross_community: boolean };
type Clustering = { algorithm: string; neighbors: number; resolution: number; silhouette: number; seed_stability_ari: number; model_revision?: string; track_count: number };
type Concept = { name: string; group?: string };
type ConceptScore = { name: string; raw_score: number; percentile: number; semantic_percentile?: number; lyrics_percentile?: number; lyrics_available?: boolean };
type JourneyStep = { id: string; title: string; artist?: string; album?: string; duration_seconds: number; source_id?: string; cover_art?: string; position: number; target_progress: number; target_similarity: number; transition_similarity: number };
type View = { x: number; y: number; zoom: number };
type GalaxySession = {
  view: View;
  showConnections: boolean;
  selectedConcepts: string[];
  conceptScores: Record<string, ConceptScore[]>;
  conceptThreshold: number;
  model: string;
  semanticWeight: number;
  selected: Point | null;
  query: string;
  journeyStart: Point | null;
  journey: JourneyStep[];
};

const galaxySession: GalaxySession = {
  view: { x: 0, y: 0, zoom: 1 }, showConnections: true, selectedConcepts: [], conceptScores: {},
  conceptThreshold: .8, model: "muq_mulan", semanticWeight: 1, selected: null, query: "",
  journeyStart: null, journey: [],
};
const palette = ["#8ff5e7", "#b9a2ff", "#86bcff", "#ff9bc8", "#f4d889", "#9de79b", "#e7a3ff", "#83e2ff", "#ffae8c", "#d4f3ba"];
const clusterColor = (cluster: number) => cluster < 0 ? "#a5b0b5" : palette[cluster % palette.length];

export default function MusicGalaxy() {
  const canvas = useRef<HTMLCanvasElement>(null);
  const drag = useRef<{ x: number; y: number; viewX: number; viewY: number } | null>(null);
  const view = useRef<View>({ ...galaxySession.view });
  const region = useRef<{ startX: number; startY: number; x: number; y: number } | null>(null);
  const selectedId = useRef(galaxySession.selected?.id || "");
  const restoringSession = useRef(Boolean(galaxySession.selected));
  // A track named in the URL (/galaxy?track=<id>) wins over the remembered session selection.
  const requestedTrack = useRef<string | null>(null);
  const loadSequence = useRef(0);
  const pointsRef = useRef<Point[]>([]);
  const transition = useRef<{ started: number; from: Map<string, { x: number; y: number }> } | null>(null);
  const pointers = useRef(new Map<number, { x: number; y: number }>());
  const pinch = useRef<{ distance: number; zoom: number; midX: number; midY: number; viewX: number; viewY: number } | null>(null);

  function pinchFrame() {
    const list = [...pointers.current.values()];
    if (list.length < 2) return null;
    const midX = (list[0].x + list[1].x) / 2, midY = (list[0].y + list[1].y) / 2;
    return { distance: Math.hypot(list[0].x - list[1].x, list[0].y - list[1].y), midX, midY };
  }
  const player = usePlayer();
  const [points, setPoints] = useState<Point[]>([]);
  const [communities, setCommunities] = useState<Community[]>([]);
  const [edges, setEdges] = useState<Edge[]>([]);
  const [clustering, setClustering] = useState<Clustering | null>(null);
  const [showConnections, setShowConnections] = useState(galaxySession.showConnections);
  const [showConceptLens, setShowConceptLens] = useState(false);
  const [activePanel, setActivePanel] = useState("map");
  const [concepts, setConcepts] = useState<Concept[]>([]);
  const [selectedConcepts, setSelectedConcepts] = useState<string[]>(galaxySession.selectedConcepts);
  const [conceptScores, setConceptScores] = useState<Record<string, ConceptScore[]>>(galaxySession.conceptScores);
  const [conceptThreshold, setConceptThreshold] = useState(galaxySession.conceptThreshold);
  const [model, setModel] = useState(galaxySession.model);
  const [semanticWeight, setSemanticWeight] = useState(galaxySession.semanticWeight);
  const [selected, setSelected] = useState<Point | null>(galaxySession.selected);
  const [hovered, setHovered] = useState<Point | null>(null);
  const [connectionId, setConnectionId] = useState("");
  const [query, setQuery] = useState(galaxySession.query);
  const [loading, setLoading] = useState(true);
  const [transitioning, setTransitioning] = useState(false);
  const [regionMode, setRegionMode] = useState(false);
  const [journeyStart, setJourneyStart] = useState<Point | null>(galaxySession.journeyStart);
  const [journey, setJourney] = useState<JourneyStep[]>(galaxySession.journey);
  const [journeyLoading, setJourneyLoading] = useState(false);
  const regions = useMemo(() => {
    type Coordinate = { x: number; y: number };
    const groups = new Map<number, Point[]>(); for (const point of points) if (point.cluster >= 0) groups.set(point.cluster, [...(groups.get(point.cluster) || []), point]);
    const centers = [...groups].map(([cluster, members]) => ({ cluster, x: members.reduce((sum, point) => sum + point.x, 0) / members.length, y: members.reduce((sum, point) => sum + point.y, 0) / members.length }));
    const clip = (polygon: Coordinate[], center: Coordinate, other: Coordinate) => { const a = 2 * (other.x - center.x), b = 2 * (other.y - center.y), c = other.x * other.x + other.y * other.y - center.x * center.x - center.y * center.y, result: Coordinate[] = []; for (let index = 0; index < polygon.length; index++) { const start = polygon[index], end = polygon[(index + 1) % polygon.length], startValue = c - a * start.x - b * start.y, endValue = c - a * end.x - b * end.y, startInside = startValue >= 0, endInside = endValue >= 0; if (startInside) result.push(start); if (startInside !== endInside) { const amount = startValue / (startValue - endValue); result.push({ x: start.x + (end.x - start.x) * amount, y: start.y + (end.y - start.y) * amount }); } } return result; };
    const cells = centers.map(center => { let polygon: Coordinate[] = [{ x: -50, y: -50 }, { x: 50, y: -50 }, { x: 50, y: 50 }, { x: -50, y: 50 }]; for (const other of centers) if (other.cluster !== center.cluster) polygon = clip(polygon, center, other); return { cluster: center.cluster, polygon }; });
    const seen = new Set<string>(), boundaries: [Coordinate, Coordinate][] = [];
    for (const cell of cells) for (let index = 0; index < cell.polygon.length; index++) { const start = cell.polygon[index], end = cell.polygon[(index + 1) % cell.polygon.length]; const onOuterEdge = (Math.abs(start.x + 50) < .01 && Math.abs(end.x + 50) < .01) || (Math.abs(start.x - 50) < .01 && Math.abs(end.x - 50) < .01) || (Math.abs(start.y + 50) < .01 && Math.abs(end.y + 50) < .01) || (Math.abs(start.y - 50) < .01 && Math.abs(end.y - 50) < .01); if (onOuterEdge) continue; const first = `${start.x.toFixed(4)},${start.y.toFixed(4)}`, second = `${end.x.toFixed(4)},${end.y.toFixed(4)}`, key = [first, second].sort().join("|"); if (!seen.has(key)) { seen.add(key); boundaries.push([start, end]); } }
    return { cells, boundaries };
  }, [points]);
  const pointsById = useMemo(() => new Map(points.map(point => [point.id, point])), [points]);
  const activeEdges = useMemo(() => {
    if (!showConnections) return [];
    const activeId = selected?.id || hovered?.id;
    if (!activeId) return [];
    return edges.filter(edge => edge.source_id === activeId || edge.target_id === activeId);
  }, [edges, hovered?.id, selected?.id, showConnections]);

  useEffect(() => {
    Object.assign(galaxySession, { showConnections, selectedConcepts, conceptScores, conceptThreshold, model, semanticWeight, selected, query, journeyStart, journey });
    return () => { galaxySession.view = { ...view.current }; };
  }, [conceptScores, conceptThreshold, journey, journeyStart, model, query, selected, selectedConcepts, semanticWeight, showConnections]);

  useEffect(() => {
    window.dispatchEvent(new CustomEvent("echora:backdrop-mode", { detail: { waveOpacity: .34 } }));
    return () => { window.dispatchEvent(new CustomEvent("echora:backdrop-mode", { detail: { waveOpacity: 1 } })); };
  }, []);
  useEffect(() => { fetch("/analysis/auth/me").then(response => response.json()).then(user => setConnectionId(user.navidrome_connection_id || "")).catch(() => {}); }, []);
  useEffect(() => {
    fetch("/analysis/library/concepts").then(response => response.json()).then(body => setConcepts([...(body.predefined || []), ...(body.personal || [])])).catch(() => {});
  }, []);
  useEffect(() => {
    if (!selectedConcepts.length) return;
    const controller = new AbortController();
    fetch("/analysis/library/concepts/lens", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ concepts: selectedConcepts, minimum_percentile: 0, representation: "hybrid" }), signal: controller.signal })
      .then(response => response.ok ? response.json() : Promise.reject(new Error("Concept scoring failed")))
      .then(body => setConceptScores(body.scores || {})).catch(error => { if (error.name !== "AbortError") setConceptScores({}); });
    return () => controller.abort();
  }, [model, selectedConcepts]);
  useEffect(() => {
    requestedTrack.current = new URLSearchParams(window.location.search).get("track");
    if (requestedTrack.current) { selectedId.current = requestedTrack.current; restoringSession.current = false; }
  }, []);
  // Mirror the selection into the URL so a track can be linked. Replace: selecting is not navigation.
  useEffect(() => {
    if (requestedTrack.current && selected?.id !== requestedTrack.current) return;
    requestedTrack.current = null;
    setUrl(selected ? `/galaxy?track=${encodeURIComponent(selected.id)}` : "/galaxy", "replace");
  }, [selected]);
  useEffect(() => {
    const controller = new AbortController();
    const timer = window.setTimeout(() => {
      const sequence = ++loadSequence.current;
      const blend = semanticWeight > 0 && semanticWeight < 1;
      const url = blend ? `/analysis/library/map?model=blend&semantic_weight=${semanticWeight}` : `/analysis/library/map?model=${model}`;
      fetch(url, { signal: controller.signal }).then(response => response.json()).then(body => { const next: Point[] = body.points || []; transition.current = { started: performance.now(), from: new Map(pointsRef.current.map(point => [point.id, { x: point.x, y: point.y }])) }; pointsRef.current = next; setTransitioning(true); window.setTimeout(() => { transition.current = null; setTransitioning(false); }, 900); setPoints(next); setCommunities(body.communities || []); setEdges(body.edges || []); setClustering(body.clustering || null); const match = next.find(point => point.id === selectedId.current); if (!match && selectedId.current) { if (requestedTrack.current) setUrl("/galaxy", "replace"); selectedId.current = ""; requestedTrack.current = null; setSelected(null); } if (match) { if (restoringSession.current) { restoringSession.current = false; setSelected(match); } else requestAnimationFrame(() => { const box = canvas.current?.getBoundingClientRect(); if (!box) return; view.current = { x: -match.x * box.width * .37, y: -match.y * box.height * .37, zoom: 2.2 }; setSelected(match); }); } setJourneyStart(current => current ? next.find(point => point.id === current.id) || current : null); }).catch(error => { if (error.name !== "AbortError") console.error(error); }).finally(() => { if (loadSequence.current === sequence) setLoading(false); });
    }, 280);
    return () => { window.clearTimeout(timer); controller.abort(); };
  }, [model, semanticWeight]);

  useEffect(() => {
    const blockPageZoom = (event: WheelEvent) => {
      if (event.ctrlKey || event.metaKey) event.preventDefault();
    };
    window.addEventListener("wheel", blockPageZoom, { passive: false });
    return () => window.removeEventListener("wheel", blockPageZoom);
  }, []);

  useEffect(() => {
    const zoomStep = (factor: number) => {
      const element = canvas.current; if (!element) return;
      const nextZoom = Math.min(9, Math.max(.55, view.current.zoom * factor));
      view.current.x = view.current.x * (view.current.zoom / nextZoom);
      view.current.y = view.current.y * (view.current.zoom / nextZoom);
      view.current.zoom = nextZoom;
    };
    const onKeyDown = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      if (target && (target.tagName === "INPUT" || target.tagName === "TEXTAREA" || target.isContentEditable)) return;
      if (event.key === "+" || event.key === "=") zoomStep(1.1);
      else if (event.key === "-" || event.key === "_") zoomStep(1 / 1.1);
      else if (event.key === "0") view.current = { x: 0, y: 0, zoom: 1 };
      else return;
      event.preventDefault();
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, []);

  function setBlend(value: number) {
    setLoading(true); setSemanticWeight(value); setModel(value === 1 ? "muq_mulan" : value === 0 ? "mert" : "blend");
  }

  function playerTrack(point: Point | JourneyStep) {
    if (!connectionId || !point.source_id) return null;
    return { id: point.id, title: point.title, artist: point.artist, album: point.album, durationSeconds: point.duration_seconds,
      streamUrl: mediaUrl(`/navidrome/connections/${connectionId}/stream/${encodeURIComponent(point.source_id)}`),
      connectionId, sourceId: point.source_id,
      coverUrl: point.cover_art ? coverArtUrl(connectionId, point.cover_art) : undefined };
  }
  function generateJourney(destination: Point) {
    if (!journeyStart || destination.id === journeyStart.id) return;
    setJourneyLoading(true);
    fetch("/analysis/library/journeys/preview", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({
      start_track_id: journeyStart.id, end_track_id: destination.id, mode: model === "blend" ? "blend" : model === "mert" ? "acoustic" : "semantic",
      semantic_weight: semanticWeight, length: 15,
    }) }).then(response => response.ok ? response.json() : Promise.reject(new Error("Could not build journey")))
      .then(body => { setJourney(body.steps || []); setJourneyStart(null); }).finally(() => setJourneyLoading(false));
  }

  useEffect(() => {
    const element = canvas.current; if (!element) return;
    const context = element.getContext("2d"); if (!context) return;
    let frame = 0;
    const paint = (now: number) => {
      const box = element.getBoundingClientRect(); const ratio = Math.min(devicePixelRatio, 1.5);
      if (element.width !== box.width * ratio || element.height !== box.height * ratio) { element.width = box.width * ratio; element.height = box.height * ratio; }
      context.setTransform(ratio, 0, 0, ratio, 0, 0); context.clearRect(0, 0, box.width, box.height);
      const map = (point: Point) => { const previous = transition.current?.from.get(point.id); const progress = transition.current ? Math.min(1, (now - transition.current.started) / 850) : 1; const eased = 1 - Math.pow(1 - progress, 3); const x = previous ? previous.x + (point.x - previous.x) * eased : point.x; const y = previous ? previous.y + (point.y - previous.y) * eased : point.y; return { x: box.width / 2 + (x * box.width * .37 + view.current.x) * view.current.zoom, y: box.height / 2 + (y * box.height * .37 + view.current.y) * view.current.zoom }; };
      const toScreen = (position: { x: number; y: number }) => ({ x: box.width / 2 + (position.x * box.width * .37 + view.current.x) * view.current.zoom, y: box.height / 2 + (position.y * box.height * .37 + view.current.y) * view.current.zoom });
      const cellsByCluster = new Map<number, typeof regions.cells>(); for (const cell of regions.cells) cellsByCluster.set(cell.cluster, [...(cellsByCluster.get(cell.cluster) || []), cell]);
      for (const [cluster, cells] of cellsByCluster) { const edgeTint = context.createRadialGradient(box.width / 2, box.height / 2, 0, box.width / 2, box.height / 2, Math.hypot(box.width, box.height) * .55); edgeTint.addColorStop(0, `${clusterColor(cluster)}0c`); edgeTint.addColorStop(.62, `${clusterColor(cluster)}07`); edgeTint.addColorStop(1, "transparent"); context.fillStyle = edgeTint; context.beginPath(); for (const cell of cells) { const polygon = cell.polygon.map(toScreen); if (polygon.length < 3) continue; context.moveTo(polygon[0].x, polygon[0].y); for (const point of polygon.slice(1)) context.lineTo(point.x, point.y); context.closePath(); } context.fill(); }
      context.strokeStyle = "rgba(176,224,224,.25)"; context.lineWidth = 1;
      for (const [start, end] of regions.boundaries) { const from = toScreen(start), to = toScreen(end); context.beginPath(); context.moveTo(from.x, from.y); context.lineTo(to.x, to.y); context.stroke(); }
      for (const community of communities) { const center = toScreen(community); context.fillStyle = clusterColor(community.id); context.font = "500 13px Manrope"; context.shadowColor = "rgba(0,0,0,.95)"; context.shadowBlur = 5; context.fillText(`${community.label.slice(0, 34)} · ${community.size}`, center.x + 12, center.y - 12); context.shadowBlur = 0; }
      for (const edge of activeEdges) { const source = pointsById.get(edge.source_id), target = pointsById.get(edge.target_id); if (!source || !target) continue; const from = map(source), to = map(target); context.strokeStyle = edge.cross_community ? "rgba(185,162,255,.62)" : "rgba(143,245,231,.38)"; context.globalAlpha = Math.max(.18, Math.min(.9, edge.similarity)); context.lineWidth = edge.mutual ? 1.2 : .8; context.setLineDash(edge.mutual ? [] : [4, 4]); context.beginPath(); context.moveTo(from.x, from.y); context.lineTo(to.x, to.y); context.stroke(); }
      context.setLineDash([]); context.globalAlpha = 1;
      if (journey.length > 1) { const path = journey.map(step => pointsById.get(step.id)).filter((point): point is Point => Boolean(point)); context.strokeStyle = "rgba(255,255,255,.82)"; context.lineWidth = 1.5; context.beginPath(); path.forEach((point, index) => { const position = map(point); if (index === 0) context.moveTo(position.x, position.y); else context.lineTo(position.x, position.y); }); context.stroke(); for (let index = 0; index < path.length; index++) { const position = map(path[index]); context.fillStyle = "#07100f"; context.strokeStyle = "#fff"; context.beginPath(); context.arc(position.x, position.y, 8, 0, Math.PI * 2); context.fill(); context.stroke(); context.fillStyle = "#fff"; context.font = "11px Manrope"; context.textAlign = "center"; context.textBaseline = "middle"; context.fillText(String(index + 1), position.x, position.y); } context.textAlign = "start"; context.textBaseline = "alphabetic"; }
      context.globalCompositeOperation = "lighter";
      for (const point of points) {
        const position = map(point); const active = player.track?.id === point.id; const chosen = selected?.id === point.id; const pulse = active ? 2 + Math.sin(now / 240) * 1.2 : 0; const radius = (chosen ? 5 : 2.3) + pulse;
        const matches = (conceptScores[point.id] || []).filter(item => item.percentile >= conceptThreshold); const strongest = matches.reduce<ConceptScore | null>((best, item) => !best || item.percentile > best.percentile ? item : best, null); const conceptIndex = strongest ? selectedConcepts.indexOf(strongest.name) : -1;
        const shade = selectedConcepts.length ? (conceptIndex >= 0 ? palette[conceptIndex % palette.length] : "#34444a") : clusterColor(point.cluster); context.globalAlpha = strongest ? .14 + strongest.percentile * .12 : .1; context.fillStyle = shade; context.beginPath(); context.arc(position.x, position.y, radius * 3.4, 0, Math.PI * 2); context.fill(); context.globalAlpha = selectedConcepts.length && !strongest ? .22 : 1;
        context.fillStyle = chosen || active ? "#fff" : shade; context.beginPath(); context.arc(position.x, position.y, radius, 0, Math.PI * 2); context.fill(); context.globalAlpha = 1;
      }
      context.globalCompositeOperation = "source-over";
      if (region.current) {
        const selection = region.current; const left = Math.min(selection.startX, selection.x), top = Math.min(selection.startY, selection.y), width = Math.abs(selection.x - selection.startX), height = Math.abs(selection.y - selection.startY);
        context.fillStyle = "rgba(155,228,221,.09)"; context.strokeStyle = "rgba(155,228,221,.9)"; context.lineWidth = 1; context.setLineDash([5, 4]); context.fillRect(left, top, width, height); context.strokeRect(left, top, width, height); context.setLineDash([]);
      }
      frame = requestAnimationFrame(paint);
    };
    frame = requestAnimationFrame(paint); return () => cancelAnimationFrame(frame);
  }, [activeEdges, communities, conceptScores, conceptThreshold, journey, player.track?.id, points, pointsById, regions, selected, selectedConcepts]);

  function pointAt(event: PointerEvent<HTMLCanvasElement>) {
    const box = event.currentTarget.getBoundingClientRect(); let nearest: Point | null = null; let best = 16;
    for (const point of points) { const x = box.width / 2 + (point.x * box.width * .37 + view.current.x) * view.current.zoom; const y = box.height / 2 + (point.y * box.height * .37 + view.current.y) * view.current.zoom; const distance = Math.hypot(event.clientX - box.left - x, event.clientY - box.top - y); if (distance < best) { best = distance; nearest = point; } }
    return nearest;
  }
  function play(point: Point) { const playable = playerTrack(point); if (playable) player.play(playable); }
  function focus(point: Point) { const box = canvas.current?.getBoundingClientRect(); if (!box) return; selectedId.current = point.id; view.current = { x: -point.x * box.width * .37, y: -point.y * box.height * .37, zoom: 2.2 }; setSelected(point); setActivePanel("track"); }
  const matches = query.trim() ? points.filter(point => `${point.title} ${point.artist} ${point.album}`.toLowerCase().includes(query.toLowerCase())).slice(0, 6) : [];
  const cover = selected?.cover_art && connectionId ? coverArtUrl(connectionId, selected.cover_art, 340) : "";
  const selectedCommunity = selected ? communities.find(community => community.id === selected.cluster) : undefined;
  const selectedConceptMatches = selected && selectedConcepts.length ? (conceptScores[selected.id] || []).filter(item => item.percentile >= conceptThreshold).sort((left, right) => right.percentile - left.percentile) : [];

  return <>
    <div className={styles.workspace}>
      <div className={styles.map}>
      <div className={styles.stage}><canvas aria-label="Interactive music galaxy. Drag to pan, scroll to zoom, or search to select a track." className={`${regionMode ? styles.selecting : ""} ${loading || transitioning ? styles.reforming : ""}`} ref={canvas} onPointerDown={event => { const box = event.currentTarget.getBoundingClientRect(); pointers.current.set(event.pointerId, { x: event.clientX - box.left, y: event.clientY - box.top }); if (pointers.current.size === 2) { drag.current = null; const frame = pinchFrame()!; pinch.current = { distance: frame.distance, zoom: view.current.zoom, midX: frame.midX, midY: frame.midY, viewX: view.current.x, viewY: view.current.y }; } else if (regionMode) region.current = { startX: event.clientX - box.left, startY: event.clientY - box.top, x: event.clientX - box.left, y: event.clientY - box.top }; else drag.current = { x: event.clientX, y: event.clientY, viewX: view.current.x, viewY: view.current.y }; event.currentTarget.setPointerCapture(event.pointerId); }} onPointerMove={event => { const box = event.currentTarget.getBoundingClientRect(); if (pointers.current.has(event.pointerId)) pointers.current.set(event.pointerId, { x: event.clientX - box.left, y: event.clientY - box.top }); if (pinch.current && pointers.current.size >= 2) { const frame = pinchFrame()!; const scale = frame.distance / pinch.current.distance; const nextZoom = Math.min(9, Math.max(.55, pinch.current.zoom * scale)); const worldX = (pinch.current.midX - box.width / 2) / pinch.current.zoom - pinch.current.viewX, worldY = (pinch.current.midY - box.height / 2) / pinch.current.zoom - pinch.current.viewY; view.current.zoom = nextZoom; view.current.x = (pinch.current.midX - box.width / 2) / nextZoom - worldX; view.current.y = (pinch.current.midY - box.height / 2) / nextZoom - worldY; view.current.x += (frame.midX - pinch.current.midX) / nextZoom; view.current.y += (frame.midY - pinch.current.midY) / nextZoom; } else if (region.current) { region.current.x = event.clientX - box.left; region.current.y = event.clientY - box.top; } else if (drag.current) { view.current.x = drag.current.viewX + (event.clientX - drag.current.x) / view.current.zoom; view.current.y = drag.current.viewY + (event.clientY - drag.current.y) / view.current.zoom; } else { const next = pointAt(event); setHovered(current => current?.id === next?.id ? current : next); } }} onPointerUp={event => { pointers.current.delete(event.pointerId); if (pointers.current.size < 2) pinch.current = null; if (pointers.current.size > 0) return; if (region.current) { const box = event.currentTarget.getBoundingClientRect(), area = region.current; const left = Math.min(area.startX, area.x), right = Math.max(area.startX, area.x), top = Math.min(area.startY, area.y), bottom = Math.max(area.startY, area.y); const chosen = points.filter(point => { const x = box.width / 2 + (point.x * box.width * .37 + view.current.x) * view.current.zoom, y = box.height / 2 + (point.y * box.height * .37 + view.current.y) * view.current.zoom; return x >= left && x <= right && y >= top && y <= bottom; }); if (chosen.length) { const minX = Math.min(...chosen.map(point => point.x)), maxX = Math.max(...chosen.map(point => point.x)), minY = Math.min(...chosen.map(point => point.y)), maxY = Math.max(...chosen.map(point => point.y)); const nextZoom = Math.min(9, Math.max(1.15, Math.min(.72 / Math.max((maxX - minX) * .37, .08), .72 / Math.max((maxY - minY) * .37, .08)))); view.current = { x: -(minX + maxX) / 2 * box.width * .37, y: -(minY + maxY) / 2 * box.height * .37, zoom: nextZoom }; } region.current = null; setRegionMode(false); } else if (drag.current && Math.hypot(event.clientX - drag.current.x, event.clientY - drag.current.y) < 5) { const point = pointAt(event); if (point && journeyStart) { generateJourney(point); } else if (point) { selectedId.current = point.id; setSelected(point); setActivePanel("track"); } else { selectedId.current = ""; setSelected(null); } } drag.current = null; }} onPointerCancel={event => { pointers.current.delete(event.pointerId); if (pointers.current.size < 2) pinch.current = null; drag.current = null; region.current = null; }} onPointerLeave={event => { pointers.current.delete(event.pointerId); if (pointers.current.size < 2) pinch.current = null; setHovered(null); }} onWheel={event => { event.preventDefault(); view.current.zoom = Math.min(9, Math.max(.55, view.current.zoom * (event.deltaY > 0 ? .9 : 1.1))); }} />
      </div>
      <div className={styles.leftPanels}><GalaxyPanel title="Explore galaxy" className={styles.controlsPanel} active={activePanel === "explore"}><div className={styles.topbar}><div className={styles.blend}><Tabs value={model === "lyrics" ? "lyrics" : semanticWeight === 0 ? "acoustic" : "semantic"} onValueChange={value => { if (value === "lyrics") { setLoading(true); setSemanticWeight(1); setModel("lyrics"); } else setBlend(value === "acoustic" ? 0 : 1); }}><TabsList aria-label="Map representation"><TabsTrigger value="semantic">Semantic</TabsTrigger><TabsTrigger value="acoustic">Acoustic</TabsTrigger><TabsTrigger value="lyrics">Lyrics</TabsTrigger></TabsList></Tabs><label><span>Sound blend</span><input aria-label="Semantic and acoustic similarity blend" type="range" min="0" max="1" step="0.05" value={1 - semanticWeight} onChange={event => setBlend(1 - Number(event.target.value))} /><span>{Math.round((1 - semanticWeight) * 100)}%</span></label></div><div className={styles.search}><div className={styles.searchField}><Search /><Input aria-label="Search the galaxy" value={query} onChange={event => setQuery(event.target.value)} placeholder={loading ? "Calculating projection" : "Search the galaxy"} /></div>{matches.length > 0 && <div>{matches.map(point => <Button variant="ghost" key={point.id} onClick={() => focus(point)}><span>{point.title}</span><small>{point.artist}</small></Button>)}</div>}</div><fieldset className={styles.tools}><legend>Display &amp; view</legend>
        <label className={styles.connectionControl}><Checkbox checked={showConnections} onCheckedChange={value => setShowConnections(value === true)} /> Show similarity connections</label>
        <div className={styles.toolRow}><Button variant="outline" aria-pressed={showConceptLens} onClick={() => { setShowConceptLens(value => !value); setActivePanel(showConceptLens ? "map" : "concepts"); }}><ListFilter />Concepts</Button>
        <Button variant="outline" aria-pressed={regionMode} onClick={() => { setRegionMode(value => !value); setActivePanel("map"); }}><ScanSearch />{regionMode ? "Cancel" : "Region"}</Button>
        <Button variant="outline" onClick={() => { view.current = { x: 0, y: 0, zoom: 1 }; }}><LocateFixed />Reset</Button></div>
      </fieldset></div></GalaxyPanel>
      <GalaxyPanel className={styles.modelPanel} active={activePanel === "model"} title={model === "lyrics" ? "Lyrics space" : model === "blend" ? "Blended space" : model === "mert" ? "Acoustic space" : "Semantic space"}><div className={styles.modelInfo}><span>{model === "lyrics" ? "BGE-M3" : model === "blend" ? `${Math.round(semanticWeight * 100)}% MuQ-MuLan · ${Math.round((1 - semanticWeight) * 100)}% MERT` : model === "mert" ? "MERT-v1-95M" : "MuQ-MuLan-large"}</span><p>{model === "lyrics" ? "A multilingual text representation of available lyrics. Distance emphasizes themes, language, imagery, and narrative rather than sound." : model === "blend" ? "Full-dimensional semantic and acoustic similarities are combined before projection, clustering, and neighbor search." : model === "mert" ? "A self-supervised music representation. Distance emphasizes audio structure, timbre, rhythm, pitch, and production." : "An audio-language representation. Distance emphasizes musical concepts that can align with words, including style, mood, vocals, and instrumentation."}</p>{clustering && <dl><div><dt>Tracks</dt><dd>{clustering.track_count}</dd></div><div><dt>Communities</dt><dd>{communities.length}</dd></div><div><dt>Stability</dt><dd>{Math.round(clustering.seed_stability_ari * 100)}%</dd></div></dl>}</div></GalaxyPanel>
      {selected && <GalaxyPanel title={selected.title} header={<div className={styles.cardHead}><Artwork trackId={selected.id} src={cover || undefined} sizes="170px" className={styles.cardArtwork} /><strong className={styles.cardTitle}>{selected.title}</strong>{selected.artist ? <TransitionLink className={styles.artistLink} href={`/artists/${encodeURIComponent(selected.artist)}`}>{selected.artist}</TransitionLink> : <p>Unknown artist</p>}<span>{selected.album}</span></div>} className={styles.trackPanel} active={activePanel === "track"} actions={<><Button variant="ghost" size="sm" onClick={() => player.track?.id === selected.id ? player.toggle() : play(selected)}>{player.track?.id === selected.id && player.playing ? <Pause /> : <Play />}{player.track?.id === selected.id && player.playing ? "Pause" : "Play"}</Button><Button variant="ghost" size="icon-sm" aria-label="Clear selected track" onClick={() => { selectedId.current = ""; setSelected(null); setActivePanel("map"); }}><X /></Button></>}><div className={styles.card}><TooltipProvider delayDuration={200}><dl className={styles.stats}>
        <Stat label="Community" help="The group of tracks this one clusters with on this map.">{selectedCommunity?.label || `Community ${selected.cluster + 1}`}</Stat>
        <Stat label="Affinity" help="How firmly the track belongs to its community. Higher means closer to the community's core.">{Math.round(selected.cluster_affinity * 100)}%</Stat>
        <Stat label="Community mix" help="Its strongest community memberships. A track can belong partly to several communities.">{selected.cluster_memberships.slice(0, 2).map(item => `C${item.cluster + 1} ${Math.round(item.strength * 100)}%`).join(" · ")}</Stat>
        <Stat label="Bridge score" help="How much the track connects different communities. High scores sit between clusters.">{Math.round(selected.bridge_score * 100)}%</Stat>
        {selectedCommunity && <Stat wide label="Community traits" help="What characterizes this community: its top genres, or embedding similarity when genres are missing.">{selectedCommunity.top_genres.map(item => item.name).join(" · ") || "Similar sound, no shared genre tags"}</Stat>}
        {selectedConceptMatches.length > 0 && <Stat wide label="Concept matches" help="How strongly the track fits each selected concept, as a percentile of your library.">{selectedConceptMatches.map(item => `${item.name} top ${Math.max(1, 100 - Math.round(item.percentile * 100))}%`).join(" · ")}</Stat>}
      </dl></TooltipProvider><section className={styles.nearest}><label>NEAREST IN {model === "lyrics" ? "LYRICS" : model === "blend" ? "BLENDED" : model === "mert" ? "ACOUSTIC" : "SEMANTIC"} SPACE</label>{selected.neighbors.filter((neighbor, index, all) => all.findIndex(item => item.title.trim().toLowerCase() === neighbor.title.trim().toLowerCase()) === index).map(neighbor => <Button variant="ghost" key={neighbor.id} onClick={() => { const point = points.find(item => item.id === neighbor.id); if (point) focus(point); }}><b>{neighbor.title}</b><i>{Math.round(neighbor.similarity * 100)}%</i></Button>)}</section></div></GalaxyPanel>}
      </div><div className={styles.rightPanels}>
      <GalaxyPanel className={styles.journeyPanel} active={activePanel === "journey"} title="Sonic journey"
        description={journeyLoading ? "Building the path…" : journeyStart ? `From ${journeyStart.title}. Select a destination star.` : journey.length ? `${journey.length} stops` : model === "lyrics" ? "Journeys use semantic or acoustic space." : selected ? `Start from ${selected.title}, then pick a destination.` : "Select a star to start from."}
        actions={(journeyStart || journey.length > 0) && <Button variant="ghost" size="icon-sm" onClick={() => { setJourneyStart(null); setJourney([]); }} aria-label="Clear journey"><X /></Button>}
        footer={journey.length > 0 ? <Button className="flex-1" onClick={() => { const queue = journey.map(playerTrack).filter((track): track is NonNullable<typeof track> => Boolean(track)); if (queue.length) player.playQueue(queue); }}><Play fill="currentColor" />Play journey</Button> : undefined}>
        {!journey.length && !journeyLoading && <Button variant="outline" size="sm" className="w-full" disabled={model === "lyrics" || (!selected && !journeyStart)} onClick={() => { setRegionMode(false); setJourney([]); setJourneyStart(current => current ? null : selected); }}><Route />{journeyStart ? "Cancel" : "Start journey"}</Button>}
        {journey.length > 0 && <div>{journey.map((step, index) => <Button variant="ghost" key={step.id} onClick={() => { const point = pointsById.get(step.id); if (point) focus(point); }}><i>{index + 1}</i><span><strong>{step.title}</strong><small>{step.artist || "Unknown artist"}</small></span></Button>)}</div>}
      </GalaxyPanel>
      <GalaxyPanel className={`${styles.conceptPanel} ${showConceptLens ? "" : styles.conceptPanelHidden}`} active={activePanel === "concepts"} title="Concept lens" description={`${selectedConcepts.length}/4 selected`} actions={selectedConcepts.length > 0 && <Button variant="ghost" aria-label="Clear selected concepts" title="Clear selected concepts" onClick={() => { setSelectedConcepts([]); setConceptScores({}); }}><X /></Button>}><div className={styles.conceptList}>{concepts.map(concept => { const checked = selectedConcepts.includes(concept.name); return <label key={concept.name}><Checkbox checked={checked} disabled={!checked && selectedConcepts.length >= 4} onCheckedChange={() => setSelectedConcepts(current => { const next = checked ? current.filter(name => name !== concept.name) : [...current, concept.name]; if (!next.length) setConceptScores({}); return next; })} /><i style={{ background: checked ? palette[selectedConcepts.indexOf(concept.name) % palette.length] : "transparent" }} /> <span>{concept.name}</span><small>{concept.group || "Personal"}</small></label>; })}</div><footer><label><span>Show top {Math.round((1 - conceptThreshold) * 100)}%</span><input type="range" aria-label="Concept percentile threshold" min="0.5" max="0.95" step="0.05" value={conceptThreshold} onChange={event => setConceptThreshold(Number(event.target.value))} /></label><p>Colors combine MuQ-MuLan sound evidence with BGE-M3 lyrics evidence. Tracks without lyrics receive a neutral lyrics score. Concepts may overlap.</p></footer></GalaxyPanel>


      </div>
      <Tabs className={styles.mobileTabs} value={activePanel} onValueChange={value => { setActivePanel(value); if (value === "concepts") setShowConceptLens(true); }}><TabsList aria-label="Galaxy panels"><TabsTrigger value="map">Map</TabsTrigger><TabsTrigger value="explore">Explore</TabsTrigger><TabsTrigger value="model">Model</TabsTrigger><TabsTrigger value="concepts">Lens</TabsTrigger><TabsTrigger value="journey">Journey</TabsTrigger><TabsTrigger value="track" disabled={!selected}>Track</TabsTrigger></TabsList></Tabs>
      {(loading || journeyLoading) && <div className={styles.loadingBadge} role="status"><Spinner className="text-primary" />{journeyLoading ? "Building journey…" : "Calculating projection…"}</div>}
      {hovered && <span className={styles.hover}>{hovered.title}<small>{hovered.artist}</small></span>}
      <div className={styles.hint}>Drag to pan&nbsp;&nbsp; / &nbsp;&nbsp;Scroll to explore&nbsp;&nbsp; / &nbsp;&nbsp;Use region select to frame a cluster</div>
      </div>
    </div>
  </>;
}

function Stat({ label, help, wide = false, children }: { label: string; help: string; wide?: boolean; children: ReactNode }) {
  return <div className={wide ? `${styles.stat} ${styles.statWide}` : styles.stat}>
    <dt>{label}<Tooltip><TooltipTrigger asChild><button type="button" className={styles.statHelp} aria-label={`About ${label.toLowerCase()}`}><Info /></button></TooltipTrigger><TooltipContent side="top" className="max-w-64 text-xs leading-relaxed">{help}</TooltipContent></Tooltip></dt>
    <dd title={typeof children === "string" ? children : undefined}>{children}</dd>
  </div>;
}
