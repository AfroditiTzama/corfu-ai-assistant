import type { InputHTMLAttributes } from "react";
import { cn } from "../../lib/utils";

export function Input({ className, ...props }: InputHTMLAttributes<HTMLInputElement>) {
  return (
    <input
      className={cn(
        "h-11 w-full rounded-xl border border-stone-200 bg-white px-3.5 text-sm text-ink outline-none transition placeholder:text-stone-400 focus:border-sea focus:ring-2 focus:ring-sea/15",
        className,
      )}
      {...props}
    />
  );
}
