"use client";

import { useRouter } from "next/navigation";
import type { AnchorHTMLAttributes, MouseEvent, ReactNode } from "react";

type Props = AnchorHTMLAttributes<HTMLAnchorElement> & { href: string; children: ReactNode };

export default function TransitionLink({ href, children, onClick, ...props }: Props) {
  const router = useRouter();
  function navigate(event: MouseEvent<HTMLAnchorElement>) {
    onClick?.(event);
    if (event.defaultPrevented || event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
    if (props.target && props.target !== "_self" || props.download != null) return;
    event.preventDefault();
    const destination = new URL(href, window.location.href);
    if (destination.href === window.location.href) return;
    router.push(href);
  }
  return <a href={href} onClick={navigate} {...props}>{children}</a>;
}
