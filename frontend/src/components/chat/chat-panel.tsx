import { useEffect, useRef, useState, type FormEvent } from "react";
import { ArrowUp, ArrowUpRight, Plus, RotateCcw, Trash2 } from "lucide-react";
import { pois } from "../../lib/dataset";
import type { ChatMessage, ChatRequest, Evidence, POI } from "../../lib/types";
import { BrandMark } from "../brand-mark";
import { Button } from "../ui/button";
import { Card } from "../ui/card";
import { PoiCard } from "../poi/poi-card";

const suggestions = [
  "What can I visit on a rainy afternoon?",
  "Show me Corfu's Venetian history",
  "Find places for a family day out",
];

const initialAssistantMessage: ChatMessage = {
  id: "initial-assistant-greeting",
  role: "assistant",
  content: "Hello and welcome to Corfu!\n\nI'm your AI travel guide. I can help you discover attractions, create personalized itineraries, and check the weather.\n\nWhat would you like to explore today?",
};

interface ChatPanelProps {
  messages: ChatMessage[];
  isThinking: boolean;
  hasConversation: boolean;
  onSend: (message: string) => void;
  onRetry: (request: ChatRequest) => void;
  onNewConversation: () => void;
  onDeleteConversation: () => void;
  onSelectPoi: (poi: POI) => void;
}

export function ChatPanel({
  messages,
  isThinking,
  hasConversation,
  onSend,
  onRetry,
  onNewConversation,
  onDeleteConversation,
  onSelectPoi,
}: ChatPanelProps) {
  const [draft, setDraft] = useState("");
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView?.({ behavior: "smooth", block: "end" });
  }, [messages, isThinking]);

  function submit(event: FormEvent) {
    event.preventDefault();
    const message = draft.trim();
    if (!message || isThinking) return;
    onSend(message);
    setDraft("");
  }

  return (
    <Card className="flex min-h-[620px] flex-col overflow-hidden lg:h-[min(690px,calc(100dvh-170px))] lg:min-h-[560px]">
      <div className="flex items-center justify-between border-b border-stone-100 px-5 py-4">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-ink text-white">
            <BrandMark className="h-8 w-8" />
          </div>
          <div>
            <p className="font-display text-sm font-bold text-ink">Your Corfu guide</p>
          </div>
        </div>
        <div className="flex items-center gap-1">
          {hasConversation && (
            <Button variant="ghost" size="sm" onClick={onDeleteConversation} aria-label="Delete conversation">
              <Trash2 className="h-4 w-4" /> <span className="hidden sm:inline">Delete</span>
            </Button>
          )}
          <Button variant="ghost" size="sm" onClick={onNewConversation} aria-label="Start a new conversation">
            <Plus className="h-4 w-4" /> New chat
          </Button>
        </div>
      </div>

      <div className="flex-1 space-y-5 overflow-y-auto px-4 py-5 sm:px-5">
        {messages.map((message) => (
          <MessageBubble
            key={message.id}
            message={message}
            onSelectPoi={onSelectPoi}
            onRetry={onRetry}
            onNewConversation={onNewConversation}
          />
        ))}
        {messages.length === 0 && (
          <section aria-label="Start a conversation" className="space-y-5">
            <MessageBubble
              message={initialAssistantMessage}
              onSelectPoi={onSelectPoi}
              onRetry={onRetry}
              onNewConversation={onNewConversation}
            />
            <div className="pl-11 sm:pl-12">
              <p className="mb-2.5 text-[11px] font-semibold uppercase tracking-wide text-stone-500">
                A few ideas
              </p>
              <div className="flex flex-col items-stretch gap-2 sm:flex-row sm:flex-wrap sm:items-center">
                {suggestions.map((prompt) => (
                  <button
                    key={prompt}
                    type="button"
                    disabled={isThinking}
                    onClick={() => onSend(prompt)}
                    className="rounded-xl border border-stone-200 bg-white px-3.5 py-2.5 text-left text-xs leading-5 text-stone-600 transition hover:border-teal-200 hover:bg-teal-50 hover:text-ink focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sea focus-visible:ring-offset-2 disabled:opacity-50 sm:rounded-full sm:py-2"
                  >
                    {prompt}
                  </button>
                ))}
              </div>
            </div>
          </section>
        )}
        {isThinking && (
          <div className="flex items-center gap-3" role="status" aria-label="Waiting for a response">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-ink text-white"><BrandMark className="h-6 w-6" /></div>
            <div className="flex items-center gap-1 rounded-2xl bg-stone-50 px-4 py-3">
              <span className="typing-dot" /><span className="typing-dot delay-150" /><span className="typing-dot delay-300" />
              <span className="ml-2 text-xs text-stone-400">Thinking…</span>
            </div>
          </div>
        )}
        <div ref={endRef} />
      </div>

      <div className="border-t border-stone-100 bg-[#FEFEFC] p-4 sm:p-5">
        <form onSubmit={submit} className="flex items-end gap-2 rounded-2xl border border-stone-200 bg-white p-2 shadow-sm transition focus-within:border-teal-300 focus-within:ring-2 focus-within:ring-sea/10">
          <textarea
            aria-label="Message your Corfu guide"
            value={draft}
            onChange={(event) => setDraft(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                event.currentTarget.form?.requestSubmit();
              }
            }}
            placeholder="Ask about places, itineraries, or the weather…"
            rows={1}
            className="max-h-28 min-h-10 flex-1 resize-none bg-transparent px-2 py-2.5 text-sm text-ink outline-none placeholder:text-stone-400"
          />
          <Button size="icon" type="submit" disabled={!draft.trim() || isThinking} aria-label="Send message" className="h-10 w-10 rounded-xl">
            <ArrowUp className="h-4 w-4" />
          </Button>
        </form>
      </div>
    </Card>
  );
}

function userFacingText(value: string): string {
  return value
    .replace(/\s*[\[(]?CF-\d{3}[\])]?/gi, "")
    .replace(/[ \t]+([,.;!?])/g, "$1")
    .replace(/;(?=\s|$)/g, ",")
    .replace(/,{2,}/g, ",")
    .replace(/\s{2,}/g, " ")
    .trim();
}

function sourceName(url: string): string {
  try {
    return new URL(url).hostname.replace(/^www\./, "");
  } catch {
    return "Background source";
  }
}

function MessageBubble({
  message,
  onSelectPoi,
  onRetry,
  onNewConversation,
}: {
  message: ChatMessage;
  onSelectPoi: (poi: POI) => void;
  onRetry: (request: ChatRequest) => void;
  onNewConversation: () => void;
}) {
  const user = message.role === "user";
  const result = message.result;
  const evidence = result?.evidence ?? [];
  const isRecommendation = result?.intent.intent === "poi_recommendation" && evidence.length > 0;
  const recommendedIds = result?.recommended_poi_ids;
  const selectedIds = recommendedIds === undefined
    ? evidence.map((item) => item.poi_id)
    : [...new Set(recommendedIds)];
  const evidenceById = new Map(evidence.map((item) => [item.poi_id, item]));
  const recommendedEvidence = selectedIds
    .map((poiId) => evidenceById.get(poiId))
    .filter((item): item is (typeof evidence)[number] => item !== undefined);
  const selectedIdSet = new Set(recommendedEvidence.map((item) => item.poi_id));
  const moreEvidence = evidence.filter((item) => !selectedIdSet.has(item.poi_id));
  const content = userFacingText(message.content);
  return (
    <div className={`flex gap-3 ${user ? "justify-end" : ""}`}>
      {!user && (
        <div className="mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-ink text-white">
          <BrandMark className="h-6 w-6" />
        </div>
      )}
      <div className={`max-w-[88%] ${user ? "rounded-2xl rounded-br-md bg-ink px-4 py-3 text-white" : "min-w-0 flex-1"}`}>
        {!user && <p className="mb-1 text-[11px] font-semibold text-sea">Corfu guide</p>}
        {content && (
          <p className={`whitespace-pre-wrap break-words text-sm leading-6 ${user ? "" : "text-stone-600"}`}>{content}</p>
        )}

        {message.error && (
          <div className="mt-3 rounded-xl border border-rose-100 bg-rose-50 p-3 text-xs leading-5 text-rose-800" role="alert">
            {message.error.conversationMissing && (
              <Button variant="secondary" size="sm" onClick={onNewConversation}>
                <Plus className="h-3.5 w-3.5" /> Start a new chat
              </Button>
            )}
            {message.request && !message.error.conversationMissing && (
              <Button
                variant="secondary"
                size="sm"
                className="mt-2"
                onClick={() => {
                  if (message.request) onRetry(message.request);
                }}
              >
                <RotateCcw className="h-3.5 w-3.5" /> Try again
              </Button>
            )}
          </div>
        )}

        {result && isRecommendation && (
          <>
            <section aria-label="Recommended for you" className="mt-3">
              <p className="mb-3 text-sm font-semibold text-ink">Recommended for you</p>
              <div className="grid gap-3 sm:grid-cols-2">
                {recommendedEvidence.map((item) => (
                  <RecommendationCard
                    key={item.poi_id}
                    item={item}
                    onSelectPoi={onSelectPoi}
                  />
                ))}
              </div>
            </section>
            {moreEvidence.length > 0 && (
              <section aria-label="More places to explore" className="mt-5">
                <p className="mb-3 text-sm font-semibold text-ink">More places to explore</p>
                <div className="grid gap-3 sm:grid-cols-2">
                  {moreEvidence.map((item) => (
                    <RecommendationCard
                      key={item.poi_id}
                      item={item}
                      onSelectPoi={onSelectPoi}
                    />
                  ))}
                </div>
              </section>
            )}
          </>
        )}

        {result && !isRecommendation && evidence.length > 0 && (
          <section aria-label="Sources and places mentioned" className="mt-3 space-y-2">
            {evidence.map((item) => {
              const poi = pois.find((candidate) => candidate.id === item.poi_id);
              return (
                <article key={item.poi_id} className="rounded-xl border border-stone-100 bg-white/80 p-3">
                  <div className="flex items-center justify-between gap-3">
                    <span className="text-sm font-semibold text-ink">{item.name}</span>
                    {poi && (
                      <button
                        type="button"
                        onClick={() => onSelectPoi(poi)}
                        aria-label={`Open ${item.name} details`}
                        className="shrink-0 rounded-lg p-1.5 text-sea hover:bg-teal-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sea"
                      >
                        <ArrowUpRight aria-hidden="true" className="h-4 w-4" />
                      </button>
                    )}
                  </div>
                  {item.source_urls.length > 0 && (
                    <ul className="mt-1 flex flex-wrap gap-x-3 gap-y-1">
                      {item.source_urls.map((url) => (
                        <li key={url}>
                          <a
                            href={url}
                            target="_blank"
                            rel="noreferrer"
                            aria-label={`Background information about ${item.name} from ${sourceName(url)}`}
                            className="inline-flex items-center gap-1 text-[11px] font-medium text-sea underline decoration-teal-200 underline-offset-2 hover:text-ink"
                          >
                            {sourceName(url)} <ArrowUpRight aria-hidden="true" className="h-3 w-3" />
                          </a>
                        </li>
                      ))}
                    </ul>
                  )}
                </article>
              );
            })}
          </section>
        )}

        {result && result.modification_applied === true && (
          <p className="mt-2 text-xs font-medium text-sea">Your itinerary has been updated.</p>
        )}
      </div>
      {user && (
        <div className="mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-sand text-xs font-bold text-amber-900">Y</div>
      )}
    </div>
  );
}

function RecommendationCard({
  item,
  onSelectPoi,
}: {
  item: Evidence;
  onSelectPoi: (poi: POI) => void;
}) {
  const poi = pois.find((candidate) => candidate.id === item.poi_id);
  if (!poi) {
    return (
      <article className="rounded-2xl border border-stone-200 bg-white p-4">
        <h3 className="font-display text-base font-bold text-ink">{item.name}</h3>
        <p className="mt-2 text-xs leading-5 text-stone-600">{userFacingText(item.description)}</p>
      </article>
    );
  }
  return (
    <PoiCard
      poi={poi}
      showDescription={false}
      sourceUrls={item.source_urls}
      onDetails={onSelectPoi}
    />
  );
}
