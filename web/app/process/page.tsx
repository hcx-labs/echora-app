import { redirect } from "next/navigation";

// Onboarding moved to /setup; old links land at its start.
export default function LegacySetupPage() {
  redirect("/setup");
}
