import type {
  ChatRequest,
  ConversationState,
  OrchestrationResult,
} from "./types";

export class AssistantApiError extends Error {
  constructor(
    message: string,
    readonly status?: number,
  ) {
    super(message);
    this.name = "AssistantApiError";
  }
}

export interface AssistantApi {
  health(signal?: AbortSignal): Promise<{ status: string }>;
  ready(signal?: AbortSignal): Promise<{ status: string }>;
  chat(request: ChatRequest, signal?: AbortSignal): Promise<OrchestrationResult>;
  getConversation(id: string, signal?: AbortSignal): Promise<ConversationState>;
  deleteConversation(id: string, signal?: AbortSignal): Promise<void>;
}

export class HttpAssistantApi implements AssistantApi {
  constructor(
    private readonly baseUrl: string,
    private readonly fetcher?: typeof fetch,
  ) {}

  health(signal?: AbortSignal): Promise<{ status: string }> {
    return this.request("/health", { signal });
  }

  ready(signal?: AbortSignal): Promise<{ status: string }> {
    return this.request("/ready", { signal });
  }

  chat(request: ChatRequest, signal?: AbortSignal): Promise<OrchestrationResult> {
    return this.request("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request),
      signal,
    });
  }

  getConversation(id: string, signal?: AbortSignal): Promise<ConversationState> {
    return this.request(`/conversations/${encodeURIComponent(id)}`, { signal });
  }

  async deleteConversation(id: string, signal?: AbortSignal): Promise<void> {
    await this.request<void>(`/conversations/${encodeURIComponent(id)}`, {
      method: "DELETE",
      signal,
      expectNoContent: true,
    });
  }

  private async request<T>(
    path: string,
    init: RequestInit & { expectNoContent?: boolean } = {},
  ): Promise<T> {
    const { expectNoContent, ...requestInit } = init;
    let response: Response;
    try {
      const requestUrl = this.baseUrl.startsWith("/")
        ? `${this.baseUrl.replace(/\/+$/, "")}${path}`
        : new URL(path, this.baseUrl);
      response = await (this.fetcher ?? globalThis.fetch)(requestUrl, requestInit);
    } catch (error) {
      if (error instanceof DOMException && error.name === "AbortError") throw error;
      throw new AssistantApiError("Could not reach the Corfu assistant. Check that the backend is running.");
    }

    if (!response.ok) {
      let detail = `Assistant request failed (${response.status}).`;
      if (response.status === 429) {
        detail = "The assistant is rate limited. Please wait a moment before trying again.";
      } else if (response.status >= 500) {
        detail = `The assistant service is unavailable (${response.status}). Please try again later.`;
      } else {
        try {
          const body: unknown = await response.json();
          if (
            body &&
            typeof body === "object" &&
            "detail" in body &&
            typeof body.detail === "string"
          ) {
            detail = body.detail;
          }
        } catch {
          // Keep the status-based message when the server does not return JSON.
        }
      }
      throw new AssistantApiError(detail, response.status);
    }
    if (expectNoContent) return undefined as T;
    try {
      return (await response.json()) as T;
    } catch {
      throw new AssistantApiError("The assistant returned an invalid response.");
    }
  }
}

export function createAssistantApi(): AssistantApi {
  const baseUrl =
    import.meta.env.VITE_API_BASE_URL?.trim() || "http://127.0.0.1:8000";
  return new HttpAssistantApi(baseUrl);
}

export const assistantApi = createAssistantApi();
