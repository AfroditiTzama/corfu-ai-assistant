import type { HTMLAttributes } from "react";
import { cn } from "../../lib/utils";

type BadgeTone = "neutral" | "teal" | "sand" | "green";

const tones: Record<BadgeTone, string> = {
  neutral: "bg-stone-100 text-stone-600",
  teal: "bg-teal-50 text-teal-800",
  sand: "bg-amber-50 text-amber-800",
  green: "bg-emerald-50 text-emerald-800",
};

interface BadgeProps extends HTMLAttributes<HTMLSpanElement> {
  tone?: BadgeTone;
}

export function Badge({ className, tone = "neutral", ...props }: BadgeProps) {
  return (
    <span
      className={cn("inline-flex items-center gap-1 rounded-full px-2.5 py-1 text-[11px] font-semibold", tones[tone], className)}
      {...props}
    />
  );
}
