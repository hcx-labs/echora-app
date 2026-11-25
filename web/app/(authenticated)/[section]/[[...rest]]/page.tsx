import { notFound } from "next/navigation";
import SectionView from "@/components/shell/SectionView";
import MusicGalaxy from "@/components/map/MusicGalaxy";
import SyncLibrary from "@/components/sync/SyncLibrary";
import CurateLibrary from "@/components/curate/CurateLibrary";
import SettingsView from "@/components/settings/SettingsView";

const sections: Record<string, string> = {
  curate: "Build and maintain playlists from natural-language recipes.",
  galaxy: "Explore the embedding galaxy.",
  evaluate: "Model evaluation will live here.",
  imports: "Library imports will live here.",
  sync: "Navidrome synchronization will live here.",
  jobs: "Processing history will live here.",
  labels: "Personal music concepts will live here.",
  settings: "Server and model settings will live here.",
};

// Sections that keep a sub-view in the path, e.g. /settings/appearance or /curate/<id>.
const nested = new Set(["settings", "curate"]);

export default async function SectionPage({ params }: { params: Promise<{ section: string; rest?: string[] }> }) {
  const { section, rest = [] } = await params;
  if (!sections[section]) notFound();
  if (rest.length > (nested.has(section) ? 1 : 0)) notFound();
  if (section === "galaxy") return <MusicGalaxy />;
  if (section === "sync") return <SyncLibrary />;
  if (section === "curate") return <CurateLibrary />;
  if (section === "settings") return <SettingsView />;
  return <SectionView name={section} description={sections[section]} />;
}
