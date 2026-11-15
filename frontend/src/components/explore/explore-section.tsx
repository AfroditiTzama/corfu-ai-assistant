import { useMemo, useState } from "react";
import { Compass, Search, SlidersHorizontal } from "lucide-react";
import { categories, pois } from "../../lib/dataset";
import type { POI } from "../../lib/types";
import { Badge } from "../ui/badge";
import { Button } from "../ui/button";
import { Card } from "../ui/card";
import { Input } from "../ui/input";
import { PoiCard } from "../poi/poi-card";

interface ExploreSectionProps {
  onSelectPoi: (poi: POI) => void;
}

function displayLabel(value: string): string {
  return value.replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toLocaleUpperCase());
}

export function ExploreSection({ onSelectPoi }: ExploreSectionProps) {
  const [query, setQuery] = useState("");
  const [category, setCategory] = useState("all");
  const [setting, setSetting] = useState("all");
  const filtered = useMemo(() => {
    const normalized = query.trim().toLocaleLowerCase();
    return pois.filter((poi) => {
      const matchesQuery =
        !normalized ||
        [poi.name, poi.area, poi.category, poi.description, ...poi.tags]
          .join(" ")
          .toLocaleLowerCase()
          .includes(normalized);
      return (
        matchesQuery &&
        (category === "all" || poi.category === category) &&
        (setting === "all" || poi.setting === setting)
      );
    });
  }, [category, query, setting]);

  return (
    <section id="explore" aria-label="Explore places in Corfu" className="scroll-mt-24 py-16 sm:py-20">
      <div className="mb-7 flex flex-wrap items-end justify-between gap-4">
        <div>
          <div className="mb-3 flex items-center gap-2 text-xs font-bold uppercase tracking-[0.18em] text-sea">
            <Compass className="h-4 w-4" /> The island, thoughtfully explored
          </div>
          <h2 className="font-display text-3xl font-bold tracking-tight text-ink sm:text-4xl">Explore Corfu</h2>
          <p className="mt-2 max-w-xl text-sm leading-6 text-stone-500">
            Find places to linger, learn and take in the island. Search by name, area or setting.
          </p>
        </div>
        <Badge tone="sand">{filtered.length} of {pois.length} places</Badge>
      </div>
      <Card className="p-4 sm:p-5">
        <div className="grid gap-3 md:grid-cols-[minmax(240px,1fr)_190px_170px]">
          <label className="relative">
            <Search className="absolute left-3.5 top-1/2 h-4 w-4 -translate-y-1/2 text-stone-400" />
            <Input
              aria-label="Search Corfu places"
              placeholder="Search names, areas, tags…"
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              className="pl-10"
            />
          </label>
          <label className="relative">
            <SlidersHorizontal className="pointer-events-none absolute left-3.5 top-1/2 h-4 w-4 -translate-y-1/2 text-stone-400" />
            <select
              aria-label="Filter by category"
              className="h-11 w-full appearance-none rounded-xl border border-stone-200 bg-white pl-10 pr-3 text-sm text-ink outline-none focus:border-sea focus:ring-2 focus:ring-sea/15"
              value={category}
              onChange={(event) => setCategory(event.target.value)}
            >
              <option value="all">All categories</option>
              {categories.map((item) => <option value={item} key={item}>{displayLabel(item)}</option>)}
            </select>
          </label>
          <label>
            <select
              aria-label="Filter by setting"
              className="h-11 w-full rounded-xl border border-stone-200 bg-white px-3 text-sm text-ink outline-none focus:border-sea focus:ring-2 focus:ring-sea/15"
              value={setting}
              onChange={(event) => setSetting(event.target.value)}
            >
              <option value="all">Any setting</option>
              <option value="indoor">Indoor</option>
              <option value="outdoor">Outdoor</option>
              <option value="mixed">Mixed</option>
            </select>
          </label>
        </div>
        {filtered.length ? (
          <div className="mt-5 grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
            {filtered.map((poi) => (
              <PoiCard
                key={poi.id}
                poi={poi}
                onDetails={onSelectPoi}
              />
            ))}
          </div>
        ) : (
          <div className="py-16 text-center">
            <p className="font-semibold text-ink">No places match these filters</p>
            <p className="mt-1 text-sm text-stone-500">Try another search or clear the selected category.</p>
            <Button className="mt-4" variant="secondary" size="sm" onClick={() => { setQuery(""); setCategory("all"); setSetting("all"); }}>
              Clear filters
            </Button>
          </div>
        )}
      </Card>
    </section>
  );
}
