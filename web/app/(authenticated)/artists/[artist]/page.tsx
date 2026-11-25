import ArtistProfileView from "@/components/artists/ArtistProfileView";

export default async function ArtistPage({ params }: { params: Promise<{ artist: string }> }) {
  const { artist } = await params;
  return <ArtistProfileView artist={decodeURIComponent(artist)} />;
}
