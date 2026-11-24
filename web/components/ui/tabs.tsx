"use client"

import * as React from "react"
import { cva, type VariantProps } from "class-variance-authority"
import { Tabs as TabsPrimitive } from "radix-ui"

import { cn } from "@/lib/utils"

function Tabs({ className, ...props }: React.ComponentProps<typeof TabsPrimitive.Root>) {
  return <TabsPrimitive.Root data-slot="tabs" className={cn("flex min-w-0 flex-col gap-5", className)} {...props} />
}

// "segmented" is a compact switcher for 2–4 peer views; "line" is section navigation.
const tabsListVariants = cva("group/tabs-list flex min-w-0 items-center", {
  variants: {
    variant: {
      segmented: "inline-flex h-9 w-fit max-w-full gap-0.5 overflow-x-auto overflow-y-hidden border border-border-strong bg-rail p-0.5 [scrollbar-width:none] [&::-webkit-scrollbar]:hidden",
      line: "w-full gap-1 overflow-x-auto overflow-y-hidden border-b border-border [scrollbar-width:none] [&::-webkit-scrollbar]:hidden",
    },
  },
  defaultVariants: { variant: "segmented" },
})

function TabsList({ className, variant = "segmented", ...props }: React.ComponentProps<typeof TabsPrimitive.List> & VariantProps<typeof tabsListVariants>) {
  return <TabsPrimitive.List data-slot="tabs-list" data-variant={variant} className={cn(tabsListVariants({ variant }), className)} {...props} />
}

function TabsTrigger({ className, ...props }: React.ComponentProps<typeof TabsPrimitive.Trigger>) {
  return (
    <TabsPrimitive.Trigger
      data-slot="tabs-trigger"
      className={cn(
        "relative inline-flex shrink-0 items-center justify-center gap-2 text-[13px] font-medium whitespace-nowrap text-muted-foreground transition-colors outline-none hover:text-foreground focus-visible:ring-2 focus-visible:ring-ring/60 disabled:pointer-events-none disabled:opacity-45 [&_svg]:pointer-events-none [&_svg]:shrink-0 [&_svg:not([class*='size-'])]:size-4",
        "group-data-[variant=segmented]/tabs-list:h-full group-data-[variant=segmented]/tabs-list:px-3 group-data-[variant=segmented]/tabs-list:data-[state=active]:bg-hover group-data-[variant=segmented]/tabs-list:data-[state=active]:text-foreground",
        "group-data-[variant=line]/tabs-list:h-11 group-data-[variant=line]/tabs-list:px-1 group-data-[variant=line]/tabs-list:mr-4 group-data-[variant=line]/tabs-list:data-[state=active]:text-foreground",
        "after:absolute after:inset-x-0 after:bottom-0 after:h-0.5 after:bg-primary after:opacity-0 group-data-[variant=line]/tabs-list:data-[state=active]:after:opacity-100",
        className
      )}
      {...props}
    />
  )
}

function TabsContent({ className, ...props }: React.ComponentProps<typeof TabsPrimitive.Content>) {
  return <TabsPrimitive.Content data-slot="tabs-content" className={cn("motion-enter min-w-0 outline-none", className)} {...props} />
}

export { Tabs, TabsList, TabsTrigger, TabsContent, tabsListVariants }
