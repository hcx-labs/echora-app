import Onboarding from "@/components/onboarding/Onboarding";

export default async function SetupPage({ params }: { params: Promise<{ step?: string[] }> }) {
  const { step } = await params;
  return <Onboarding initialStep={step?.[0]} />;
}
