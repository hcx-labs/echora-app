import type { Metadata } from "next";
import Backdrop from "@/components/shell/Backdrop";
import { PlayerProvider } from "@/components/player/PlayerProvider";
import MotionPreferences from "@/components/shell/MotionPreferences";
import RuntimeConfigBootstrap from "@/components/runtime/RuntimeConfigBootstrap";
import SessionRenewal from "@/components/session/SessionRenewal";
// Legacy surface tokens first, so the design system in globals.css wins where names overlap.
import "./theme.css";
import "./globals.css";
import { Toaster } from "@/components/ui/sonner";

export const metadata: Metadata = {
  title: "Echora",
  description: "A personal map of your music",
};

// The media base path comes from the environment at request time; without
// this the layout is prerendered and the value frozen at build time.
export const dynamic = "force-dynamic";

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  // Runtime, not build-time: the media base path depends on the deployment's
  // reverse proxy shape, which must not require an image rebuild.
  const mediaBase = process.env.ECHORA_MEDIA_BASE ?? "/analysis";
  return (
    <html lang="en" className="dark">
      <head>
        <script
          dangerouslySetInnerHTML={{ __html: `window.__ECHORA_MEDIA_BASE__=${JSON.stringify(mediaBase)};` }}
        />
      </head>
      <body><div className="app-viewport"><RuntimeConfigBootstrap /><MotionPreferences /><Backdrop /><PlayerProvider><SessionRenewal />{children}</PlayerProvider><Toaster /></div></body>
    </html>
  );
}
