import * as React from "react"

import { cn } from "@/lib/utils"

function Textarea({ className, ...props }: React.ComponentProps<"textarea">) {
  return (
    <textarea
      data-slot="textarea"
      className={cn(
        "flex field-sizing-content min-h-16 w-full rounded-none border border-border-strong bg-surface px-3 py-2 text-base leading-relaxed transition-colors outline-none placeholder:text-subtle-foreground hover:border-[#444] focus-visible:border-ring focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/30 disabled:cursor-not-allowed disabled:opacity-50 aria-invalid:border-destructive md:text-[13px]",
        className
      )}
      {...props}
    />
  )
}

export { Textarea }
