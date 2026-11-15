import { describe, expect, it, vi } from "vitest";
import { AssistantApiError, HttpAssistantApi } from "./api";

function response(body: unknown, status = 200): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    statusText: "response",
    json: async () => body,
  } as Response;
}

describe("HttpAssistantApi", () => {
  it("posts the typed chat request and parses the orchestration result", async () => {
    const fetcher = vi.fn().mockResolvedValue(response({ status: "completed" }));
    const api = new HttpAssistantApi("http://127.0.0.1:8000", fetcher);
    const request = {
      message: "Museum recommendations",
      request_id: "8ba6e65d-b715-4f15-9c95-351cc66d12cb",
    };

    await expect(api.chat(request)).resolves.toEqual({ status: "completed" });
    expect(fetcher).toHaveBeenCalledWith(
      new URL("http://127.0.0.1:8000/chat"),
      expect.objectContaining({
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(request),
      }),
    );
  });

  it("supports a same-origin API prefix for reverse-proxy deployments", async () => {
    const fetcher = vi.fn().mockResolvedValue(response({ status: "completed" }));
    const api = new HttpAssistantApi("/api", fetcher);

    await api.chat({
      message: "Plan an afternoon",
      request_id: "8ba6e65d-b715-4f15-9c95-351cc66d12cb",
    });

    expect(fetcher).toHaveBeenCalledWith(
      "/api/chat",
      expect.objectContaining({ method: "POST" }),
    );
  });

  it("reports safe HTTP errors including quota and upstream failures", async () => {
    const quotaApi = new HttpAssistantApi("http://api.test", vi.fn().mockResolvedValue(response({}, 429)));
    await expect(quotaApi.health()).rejects.toMatchObject({
      name: "AssistantApiError",
      status: 429,
      message: expect.stringContaining("rate limited"),
    });

    const unavailableApi = new HttpAssistantApi(
      "http://api.test",
      vi.fn().mockResolvedValue(response({ detail: "private provider data" }, 503)),
    );
    await expect(unavailableApi.ready()).rejects.toMatchObject({
      status: 503,
      message: expect.not.stringContaining("private provider data"),
    });
  });

  it("surfaces client-safe API detail and handles malformed response JSON", async () => {
    const invalidApi = new HttpAssistantApi(
      "http://api.test",
      vi.fn().mockResolvedValue(response({ detail: "message must not be blank" }, 422)),
    );
    await expect(invalidApi.chat({ message: "", request_id: "id" })).rejects.toThrow(
      "message must not be blank",
    );

    const malformedApi = new HttpAssistantApi("http://api.test", vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => { throw new SyntaxError("bad json"); },
    } as unknown as Response));
    await expect(malformedApi.health()).rejects.toThrow("invalid response");
  });

  it("converts network failures and preserves abort cancellation", async () => {
    const failedApi = new HttpAssistantApi(
      "http://api.test",
      vi.fn().mockRejectedValue(new TypeError("fetch failed")),
    );
    await expect(failedApi.health()).rejects.toBeInstanceOf(AssistantApiError);

    const abort = new DOMException("cancelled", "AbortError");
    const abortedApi = new HttpAssistantApi("http://api.test", vi.fn().mockRejectedValue(abort));
    await expect(abortedApi.health()).rejects.toBe(abort);
  });

  it("uses conversation endpoints and handles 204 deletion", async () => {
    const fetcher = vi.fn()
      .mockResolvedValueOnce(response({ conversation_id: "conversation" }))
      .mockResolvedValueOnce(response(null, 204));
    const api = new HttpAssistantApi("http://127.0.0.1:8000", fetcher);
    await expect(api.getConversation("conversation")).resolves.toEqual({
      conversation_id: "conversation",
    });
    await expect(api.deleteConversation("conversation")).resolves.toBeUndefined();
    expect(fetcher.mock.calls[1][1]).toMatchObject({ method: "DELETE" });
  });
});
