"use client";

import { useEffect } from "react";
import AppLoading from "./shell/AppLoading";

export default function OnboardingGate() {
  useEffect(() => {
    fetch("/api/v1/auth/me")
      .then((response) => {
        if (response.status === 401) {
          window.location.replace("/login");
          return null;
        }
        if (!response.ok) throw new Error("Could not read user preferences");
        return response.json();
      })
      .then((user) => {
        if (user) window.location.replace(user.onboarding_complete ? "/home" : "/setup");
      })
      .catch(() => window.location.replace("/login"));
  }, []);

  return <AppLoading />;
}
