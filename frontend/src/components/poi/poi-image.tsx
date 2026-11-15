import { useState } from "react";
import {
  Building2,
  Church,
  Landmark,
  Palette,
  Trees,
  Utensils,
  type LucideIcon,
} from "lucide-react";
import { getPoiImage } from "../../lib/poi-image-manifest";
import type { POI } from "../../lib/types";

const categoryIcons: Record<string, LucideIcon> = {
  arts_culture: Palette,
  food_drink: Utensils,
  garden: Trees,
  history: Landmark,
  museum: Landmark,
  nature: Trees,
  religious_site: Church,
};

function fallbackIcon(poi: POI): LucideIcon {
  const category = poi.category.toLocaleLowerCase();
  if (categoryIcons[category]) return categoryIcons[category];
  if (poi.tags.some((tag) => ["garden", "park", "nature", "grove"].includes(tag))) return Trees;
  if (poi.tags.some((tag) => ["church", "religious", "monastery"].includes(tag))) return Church;
  if (poi.tags.some((tag) => ["museum", "history", "fortress", "heritage"].includes(tag))) return Landmark;
  if (poi.tags.some((tag) => ["art", "gallery", "craft"].includes(tag))) return Palette;
  if (poi.tags.some((tag) => ["restaurant", "food", "cafe", "bar"].includes(tag))) return Utensils;
  return Building2;
}

interface PoiImageProps {
  poi: POI;
  className?: string;
}

export function PoiImage({ poi, className = "" }: PoiImageProps) {
  const image = getPoiImage(poi.id);
  const [unavailable, setUnavailable] = useState(false);

  if (!image || unavailable) {
    const Icon = fallbackIcon(poi);
    return (
      <div
        role="img"
        aria-label={`${poi.name} illustration`}
        className={`relative flex h-full w-full items-center justify-center overflow-hidden bg-gradient-to-br from-teal-50 via-[#e8f1ec] to-amber-50 text-sea ${className}`}
      >
        <span className="absolute -right-8 -top-10 h-36 w-36 rounded-full border-[18px] border-white/40" />
        <span className="absolute -bottom-12 -left-6 h-36 w-36 rounded-full border-[16px] border-white/40" />
        <span className="relative flex h-16 w-16 items-center justify-center rounded-3xl border border-white/80 bg-white/70 shadow-sm">
          <Icon aria-hidden="true" className="h-8 w-8" strokeWidth={1.5} />
        </span>
      </div>
    );
  }

  return (
    <img
      src={image.assetPath}
      alt={image.altText}
      loading="lazy"
      decoding="async"
      onError={() => setUnavailable(true)}
      className={`h-full w-full object-cover ${className}`}
    />
  );
}
