import { ArrowUpRight, Clock3, ExternalLink, Info } from "lucide-react";
import { getPoiImage } from "../../lib/poi-image-manifest";
import type { POI } from "../../lib/types";
import { formatDuration } from "../../lib/utils";
import { PoiImage } from "./poi-image";

interface PoiCardProps {
  poi: POI;
  description?: string;
  showDescription?: boolean;
  sourceUrls?: string[];
  onDetails: (poi: POI) => void;
}

function humanize(value: string): string {
  return value.replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toLocaleUpperCase());
}

function sourceName(url: string): string {
  try {
    return new URL(url).hostname.replace(/^www\./, "");
  } catch {
    return "Background source";
  }
}

export function PoiPhotoAttribution({ poi }: { poi: POI }) {
  const image = getPoiImage(poi.id);
  if (!image) return null;
  return (
    <p className="truncate text-[10px] leading-4 text-stone-500">
      Photo by{" "}
      <a
        href={image.originalSourceUrl}
        target="_blank"
        rel="noreferrer"
        aria-label={`Photo attribution for ${poi.name}: ${image.creator}`}
        className="font-medium text-stone-600 underline decoration-stone-300 underline-offset-2 hover:text-sea"
      >
        {image.creator}
      </a>
      {" · "}
      <a
        href={image.licenseUrl}
        target="_blank"
        rel="noreferrer"
        aria-label={`${image.license} license for ${poi.name} photo`}
        className="underline decoration-stone-300 underline-offset-2 hover:text-sea"
      >
        {image.license}
      </a>
    </p>
  );
}

export function PoiCard({
  poi,
  description,
  showDescription = true,
  sourceUrls,
  onDetails,
}: PoiCardProps) {
  const sources = sourceUrls?.length ? sourceUrls : poi.source_urls;
  return (
    <article className="group flex min-w-0 flex-col overflow-hidden rounded-2xl border border-stone-200/80 bg-white shadow-sm transition duration-200 hover:-translate-y-0.5 hover:shadow-card focus-within:ring-2 focus-within:ring-sea/30">
      <div className="aspect-[16/10] overflow-hidden bg-teal-50">
        <PoiImage
          poi={poi}
          className="transition duration-500 group-hover:scale-[1.03]"
        />
      </div>
      <div className="flex flex-1 flex-col p-4">
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0">
            <p className="text-[11px] font-medium text-stone-500">{poi.area}</p>
            <h3 className="mt-1 font-display text-base font-bold leading-snug text-ink">
              {poi.name}
            </h3>
          </div>
          <span className="shrink-0 rounded-full bg-teal-50 px-2.5 py-1 text-[10px] font-semibold text-teal-800">
            {humanize(poi.category)}
          </span>
        </div>
        {showDescription && (
          <p className="mt-2 line-clamp-3 text-xs leading-5 text-stone-600">
            {description ?? poi.description}
          </p>
        )}
        <p className="mt-3 flex items-center gap-1.5 text-[11px] text-stone-500">
          <Clock3 aria-hidden="true" className="h-3.5 w-3.5 text-sea" />
          Suggested visit: {formatDuration(poi.suggested_visit_minutes)} · estimate
        </p>
        <div className="mt-auto pt-3">
          {getPoiImage(poi.id) && <PoiPhotoAttribution poi={poi} />}
          {sources.length > 0 && (
            <div className="mt-2 flex flex-wrap gap-x-3 gap-y-1">
              {sources.slice(0, 2).map((url) => (
                <a
                  key={url}
                  href={url}
                  target="_blank"
                  rel="noreferrer"
                  aria-label={`Background information about ${poi.name} from ${sourceName(url)}`}
                  className="inline-flex max-w-full items-center gap-1 truncate text-[11px] font-medium text-sea underline decoration-teal-200 underline-offset-2 hover:text-ink"
                >
                  {sourceName(url)} <ExternalLink aria-hidden="true" className="h-3 w-3 shrink-0" />
                </a>
              ))}
            </div>
          )}
          {poi.opening_hours_fixture === null && (
            <p className="mt-2 flex items-center gap-1 text-[10px] text-stone-500">
              <Info aria-hidden="true" className="h-3 w-3" />
              Opening hours not confirmed
            </p>
          )}
          <button
            type="button"
            onClick={() => onDetails(poi)}
            aria-label={`Open ${poi.name} details`}
            className="mt-3 inline-flex min-h-9 w-full items-center justify-between rounded-xl bg-[#F5F8F6] px-3 text-xs font-semibold text-ink transition hover:bg-teal-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sea"
          >
            View place details
            <ArrowUpRight aria-hidden="true" className="h-4 w-4 text-sea" />
          </button>
        </div>
      </div>
    </article>
  );
}
