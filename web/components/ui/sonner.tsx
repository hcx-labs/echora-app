"use client"

import type { CSSProperties } from "react"
import { CircleAlert, CircleCheck, Info, TriangleAlert } from "lucide-react"
import { Toaster as Sonner, type ToasterProps } from "sonner"

import { Spinner } from "./spinner"

/**
 * Transient feedback for actions: saved, failed, needs attention. Sits above
 * the player bar. Lasting caveats stay inline as a Notice.
 */
function Toaster(props: ToasterProps) {
  return (
    <Sonner
      theme="dark"
      position="bottom-right"
      offset={{ bottom: "calc(var(--player-height) + 16px)", right: 16 }}
      mobileOffset={{ bottom: "calc(var(--player-height) + 12px)" }}
      gap={8}
      icons={{
        success: <CircleCheck className="size-4 text-success" />,
        info: <Info className="size-4 text-primary" />,
        warning: <TriangleAlert className="size-4 text-warning" />,
        error: <CircleAlert className="size-4 text-destructive" />,
        loading: <Spinner className="size-4" />,
      }}
      toastOptions={{
        classNames: {
          toast: "!rounded-none !border !border-border-strong !bg-raised !text-foreground !shadow-2xl !shadow-black/60 !font-sans !gap-2.5 !px-3.5 !py-3",
          title: "!text-[13px] !font-medium",
          description: "!text-xs !text-muted-foreground !leading-relaxed",
          actionButton: "!rounded-none !bg-primary !text-primary-foreground !text-xs !font-medium",
          cancelButton: "!rounded-none !bg-hover !text-foreground !text-xs",
          closeButton: "!rounded-none !border-border-strong !bg-raised !text-muted-foreground hover:!text-foreground",
          success: "!border-l-2 !border-l-success",
          info: "!border-l-2 !border-l-primary",
          warning: "!border-l-2 !border-l-warning",
          error: "!border-l-2 !border-l-destructive",
        },
      }}
      style={{ zIndex: 1200 } as CSSProperties}
      {...props}
    />
  )
}

export { Toaster }
