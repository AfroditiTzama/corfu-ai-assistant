import type { SVGProps } from "react";

export function BrandMark(props: SVGProps<SVGSVGElement>) {
  return (
    <svg
      viewBox="0 0 48 40"
      fill="none"
      aria-hidden="true"
      {...props}
    >
      <text
        x="24"
        y="31"
        textAnchor="middle"
        fontFamily="Georgia, 'Times New Roman', serif"
        fontSize="34"
        fontWeight="500"
        fill="currentColor"
      >
        C
      </text>
      <path
        d="M10 31c4-3 7 3 11 0s7-3 11 0 7 3 11 0"
        stroke="#55B2A3"
        strokeWidth="1.8"
        strokeLinecap="round"
      />
    </svg>
  );
}
