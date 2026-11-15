import * as Dialog from "@radix-ui/react-dialog";
import { ExternalLink, MapPin, X } from "lucide-react";
import type { POI } from "../../lib/types";
import { formatDuration } from "../../lib/utils";
import { Badge } from "../ui/badge";
import { PoiPhotoAttribution } from "./poi-card";
import { PoiImage } from "./poi-image";

interface PoiDialogProps {
  poi: POI | null;
  onClose: () => void;
}

function familyLabel(value: POI["family_suitability_fixture"]): string {
  if (value === true) return "Marked suitable in the planning data; not a safety guarantee.";
  if (value === false) return "Not marked as suitable for child-oriented plans.";
  return "No family-suitability information available.";
}

function displayLabel(value: string): string {
  return value.replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toLocaleUpperCase());
}

function sourceName(url: string): string {
  try {
    return new URL(url).hostname.replace(/^www\./, "");
  } catch {
    return "Background information";
  }
}

export function PoiDialog({ poi, onClose }: PoiDialogProps) {
  return (
    <Dialog.Root open={poi !== null} onOpenChange={(open) => !open && onClose()}>
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 z-40 bg-ink/35 backdrop-blur-sm data-[state=open]:animate-in data-[state=closed]:animate-out" />
        <Dialog.Content className="fixed left-1/2 top-1/2 z-50 max-h-[85vh] w-[min(92vw,620px)] -translate-x-1/2 -translate-y-1/2 overflow-y-auto rounded-3xl border border-stone-200 bg-white p-6 shadow-2xl focus:outline-none sm:p-8">
          {poi && (
            <>
              <div className="mb-6 flex items-start justify-between gap-4">
                <div>
                  <Dialog.Title className="font-display text-2xl font-bold tracking-tight text-ink">
                    {poi.name}
                  </Dialog.Title>
                  <Dialog.Description className="mt-2 flex items-center gap-1.5 text-sm text-stone-500">
                    <MapPin className="h-4 w-4 text-sea" />
                    {poi.area}
                  </Dialog.Description>
                </div>
                <Dialog.Close asChild>
                  <button
                    className="rounded-full p-2 text-stone-400 hover:bg-stone-100 hover:text-ink"
                    aria-label="Close POI details"
                  >
                    <X className="h-5 w-5" />
                  </button>
                </Dialog.Close>
              </div>
              <div className="mb-2 aspect-[16/8] overflow-hidden rounded-2xl bg-teal-50">
                <PoiImage poi={poi} />
              </div>
              <div className="mb-5">
                <PoiPhotoAttribution poi={poi} />
              </div>
              <div className="mb-5 flex flex-wrap gap-2">
                <Badge tone="teal">{displayLabel(poi.category)}</Badge>
                <Badge>{poi.setting === "mixed" ? "Indoor & outdoor" : poi.setting}</Badge>
                {poi.tags.map((tag) => <Badge key={tag}>{displayLabel(tag)}</Badge>)}
              </div>
              <p className="mb-1 text-xs font-semibold uppercase tracking-wide text-stone-400">About this place</p>
              <p className="text-sm leading-7 text-stone-600">{poi.description}</p>
              <div className="mt-6 grid gap-3 rounded-2xl bg-sand/50 p-4 text-sm sm:grid-cols-2">
                <div>
                  <span className="text-xs font-semibold uppercase tracking-wide text-stone-400">Suggested visit</span>
                  <p className="mt-1 font-semibold text-ink">{formatDuration(poi.suggested_visit_minutes)} · estimate</p>
                </div>
                <div>
                  <span className="text-xs font-semibold uppercase tracking-wide text-stone-400">Family planning note</span>
                  <p className="mt-1 font-semibold text-ink">{familyLabel(poi.family_suitability_fixture)}</p>
                </div>
                <div className="sm:col-span-2">
                  <span className="text-xs font-semibold uppercase tracking-wide text-stone-400">Opening hours</span>
                  <p className="mt-1 text-stone-600">
                    {poi.opening_hours_fixture
                      ? "Planning schedule only; confirm current hours before visiting."
                      : "Hours are unknown; access is not confirmed."}
                  </p>
                </div>
              </div>
              {poi.source_urls.length > 0 && (
                <div className="mt-6">
                  <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-stone-500">Learn more</p>
                  <ul className="flex flex-wrap gap-2">
                    {poi.source_urls.map((url) => (
                      <li key={url}>
                        <a
                          href={url}
                          target="_blank"
                          rel="noreferrer"
                          aria-label={`Learn about ${poi.name} from ${sourceName(url)}`}
                          className="inline-flex min-h-9 items-center gap-1.5 rounded-xl border border-stone-200 px-3 text-xs font-medium text-sea transition hover:border-teal-200 hover:bg-teal-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sea"
                        >
                          {sourceName(url)} <ExternalLink className="h-3.5 w-3.5 shrink-0" />
                        </a>
                      </li>
                    ))}
                  </ul>
                  <p className="mt-3 text-xs leading-5 text-stone-500">Reference links provide background information, not current access or availability.</p>
                </div>
              )}
              <p className="mt-5 text-xs leading-5 text-stone-500">
                Visitor details are planning estimates, not verified live information. Please confirm current arrangements before your visit.
              </p>
            </>
          )}
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}
