import { useEffect, useMemo, useRef, useState } from "react";
import { MapPin, ShieldCheck } from "lucide-react";
import { BrandMark } from "./components/brand-mark";
import { ChatPanel } from "./components/chat/chat-panel";
import { ExploreSection } from "./components/explore/explore-section";
import { ItineraryPanel } from "./components/itinerary/itinerary-panel";
import { PoiDialog } from "./components/poi/poi-dialog";
import { WeatherPanel } from "./components/weather/weather-panel";
import { Card } from "./components/ui/card";
import { assistantApi, AssistantApiError } from "./lib/api";
import type { ChatMessage, ChatRequest, OrchestrationResult, POI } from "./lib/types";

function newId(): string {
  if (globalThis.crypto?.randomUUID) return globalThis.crypto.randomUUID();
  return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (character) => {
    const random = Math.floor(Math.random() * 16);
    return (character === "x" ? random : (random & 0x3) | 0x8).toString(16);
  });
}

function isAbortError(error: unknown): boolean {
  return error instanceof Error && error.name === "AbortError";
}

function errorText(error: unknown): string {
  return error instanceof Error ? error.message : "The assistant request failed unexpectedly.";
}

export default function App() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isThinking, setIsThinking] = useState(false);
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [selectedPoi, setSelectedPoi] = useState<POI | null>(null);
  const busyRef = useRef(false);
  const requestControllerRef = useRef<AbortController | null>(null);

  useEffect(() => () => requestControllerRef.current?.abort(), []);

  const planningResult = useMemo(() => {
    const result = [...messages].reverse().find((message) =>
      message.result?.planning_result || message.result?.weather_replanning?.planning_result,
    );
    return result?.result?.planning_result ?? result?.result?.weather_replanning?.planning_result ?? null;
  }, [messages]);
  const latestPlanningMessage = useMemo(
    () => [...messages].reverse().find((message) =>
      message.result?.planning_result || message.result?.weather_replanning?.planning_result,
    ),
    [messages],
  );
  const weatherResult: OrchestrationResult | null = useMemo(() => {
    const result = [...messages].reverse().find((message) =>
      message.result && (
        message.result.weather_forecast ||
        message.result.weather_replanning ||
        message.result.weather_error ||
        message.result.intent.intent === "live_weather_query" ||
        message.result.intent.intent === "weather_based_replanning"
      ),
    );
    return result?.result ?? null;
  }, [messages]);
  const weatherReplanning = useMemo(() => {
    return latestPlanningMessage?.result?.weather_replanning ?? null;
  }, [latestPlanningMessage]);

  async function sendRequest(request: ChatRequest, addUserMessage: boolean): Promise<void> {
    if (busyRef.current) return;
    busyRef.current = true;
    setIsThinking(true);
    const controller = new AbortController();
    requestControllerRef.current = controller;
    if (addUserMessage) {
      setMessages((current) => [...current, {
        id: request.request_id,
        role: "user",
        content: request.message,
        request,
      }]);
    }

    try {
      const result = await assistantApi.chat(request, controller.signal);
      setConversationId(result.conversation_id);
      const planning = result.planning_result ?? result.weather_replanning?.planning_result;
      setMessages((current) => [...current, {
        id: newId(),
        role: "assistant",
        content: planning?.status === "infeasible"
          ? "The itinerary cannot meet all requested constraints. Review the validation details in the plan."
          : result.response,
        result,
      }]);
    } catch (error) {
      if (isAbortError(error)) return;
      const missingConversation =
        error instanceof AssistantApiError &&
        error.status === 404 &&
        request.conversation_id !== undefined;
      setMessages((current) => [...current, {
        id: newId(),
        role: "assistant",
        content: missingConversation
          ? "This stored conversation has expired or was lost after a backend restart."
          : errorText(error),
        request,
        error: {
          message: errorText(error),
          conversationMissing: missingConversation,
        },
      }]);
    } finally {
      if (requestControllerRef.current === controller) {
        requestControllerRef.current = null;
        busyRef.current = false;
        setIsThinking(false);
      }
    }
  }

  function sendMessage(message: string): void {
    if (busyRef.current) return;
    const request: ChatRequest = {
      message,
      request_id: newId(),
      ...(conversationId ? { conversation_id: conversationId } : {}),
    };
    void sendRequest(request, true);
  }

  function startNewConversation(): void {
    requestControllerRef.current?.abort();
    requestControllerRef.current = null;
    busyRef.current = false;
    setIsThinking(false);
    setConversationId(null);
    setMessages([]);
  }

  async function deleteConversation(): Promise<void> {
    if (!conversationId || busyRef.current) return;
    const id = conversationId;
    try {
      await assistantApi.deleteConversation(id);
      startNewConversation();
    } catch (error) {
      if (error instanceof AssistantApiError && error.status === 404) {
        startNewConversation();
        return;
      }
      setMessages((current) => [...current, {
        id: newId(),
        role: "assistant",
        content: `Could not delete this conversation: ${errorText(error)}`,
      }]);
    }
  }

  return (
    <div className="min-h-screen bg-[#F6F7F4] text-ink">
      <header className="sticky top-0 z-30 border-b border-stone-200/70 bg-[#F8F9F6]/90 backdrop-blur-xl">
        <div className="mx-auto flex h-16 max-w-[1440px] items-center justify-between px-4 sm:px-7 lg:px-10">
          <a href="#" className="flex items-center gap-2.5" aria-label="Corfu AI Travel Assistant home">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-ink text-[#F6F7F4]">
              <BrandMark className="h-8 w-8" />
            </div>
            <div>
              <span className="block font-display text-sm font-bold tracking-[0.12em] text-ink sm:text-base">CORFU</span>
              <span className="hidden text-[9px] font-semibold tracking-[0.16em] text-stone-500 sm:block">TRAVEL ASSISTANT</span>
            </div>
          </a>
          <nav className="hidden items-center gap-7 md:flex">
            <a href="#assistant" className="text-sm font-medium text-stone-500 transition hover:text-ink">Assistant</a>
            <a href="#explore" className="text-sm font-medium text-stone-500 transition hover:text-ink">Explore</a>
          </nav>
          <div className="flex items-center gap-2.5">
            <span className="hidden items-center gap-1.5 rounded-full border border-stone-200 bg-white px-3 py-1.5 text-xs font-semibold text-stone-600 sm:inline-flex">
              <MapPin className="h-3.5 w-3.5 text-sea" /> Corfu, Greece
            </span>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-[1440px] px-4 sm:px-7 lg:px-10">
        <section className="border-b border-stone-200/70 py-5 sm:py-6">
          <div className="flex flex-col gap-1.5 sm:flex-row sm:items-end sm:justify-between sm:gap-6">
            <h1 className="font-serif text-2xl font-medium tracking-tight text-ink sm:text-3xl">
              Explore Corfu
            </h1>
            <p className="max-w-xl text-sm leading-6 text-stone-600 sm:text-right">
              Discover attractions, plan itineraries and check the weather.
            </p>
          </div>
        </section>

        <section id="assistant" className="grid scroll-mt-24 gap-5 lg:grid-cols-[minmax(0,0.94fr)_minmax(0,1.06fr)]">
          <ChatPanel
            messages={messages}
            isThinking={isThinking}
            hasConversation={conversationId !== null}
            onSend={sendMessage}
            onRetry={(request) => { void sendRequest(request, false); }}
            onNewConversation={startNewConversation}
            onDeleteConversation={() => { void deleteConversation(); }}
            onSelectPoi={setSelectedPoi}
          />
          <div className="space-y-5">
            <ItineraryPanel planningResult={planningResult} weatherReplanning={weatherReplanning} />
            <WeatherPanel
              forecast={weatherResult?.weather_forecast ?? null}
              assessment={weatherResult?.rain_assessment ?? null}
              error={weatherResult?.weather_error ?? null}
              wasRequested={weatherResult !== null}
            />
          </div>
        </section>

        <ExploreSection onSelectPoi={setSelectedPoi} />

        <Card className="mb-12 flex flex-col gap-3 bg-ink p-5 text-white sm:flex-row sm:items-center sm:justify-between sm:p-6">
          <div className="flex items-start gap-3">
            <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-white/10 text-teal-200"><ShieldCheck className="h-5 w-5" /></div>
            <div>
              <p className="text-sm font-semibold">Know before you go</p>
              <p className="mt-1 max-w-2xl text-xs leading-5 text-white/70">Opening hours, family notes and visit durations are planning estimates, not verified visitor information. Walking times are approximate; check current details before setting out.</p>
            </div>
          </div>
        </Card>
      </main>
      <footer className="border-t border-stone-200/70 py-6 text-center text-[11px] text-stone-400">
        Corfu AI Travel Assistant <span className="mx-1.5">·</span> Discover the island at your own pace
      </footer>
      <PoiDialog poi={selectedPoi} onClose={() => setSelectedPoi(null)} />
    </div>
  );
}
