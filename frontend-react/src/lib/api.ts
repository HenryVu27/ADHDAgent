import type {
  ChatRequest,
  StreamEvent,
  SessionResponse,
  OutcomesResponse,
  KnowledgeDocument,
  OnboardingData,
  SessionListResponse,
  SessionDetailResponse,
  ObservabilityEvent,
  SessionsResponse,
  MessagesResponse,
  UploadResponse,
  UploadBlockedResponse,
} from "@/types"
import { getToken, signOut } from "@/lib/auth"

const BASE = "/api"

// SSE streaming must bypass the Vite dev proxy to avoid response buffering.
// In production (built React served by FastAPI), use the normal relative path.
const SSE_BASE = import.meta.env.DEV ? "http://localhost:8000/api" : "/api"

function authHeaders(): Record<string, string> {
  const token = getToken()
  return token ? { Authorization: `Bearer ${token}` } : {}
}

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: {
      "Content-Type": "application/json",
      ...authHeaders(),
      ...(options?.headers as Record<string, string> | undefined),
    },
    ...options,
  })
  if (res.status === 401) {
    signOut()
    window.location.href = "/signin"
    throw new Error("Session expired")
  }
  if (!res.ok) {
    const error = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(error.detail || `Request failed: ${res.status}`)
  }
  return res.json()
}

export const api = {
  async *chatStream(
    data: ChatRequest,
    signal: AbortSignal,
  ): AsyncGenerator<StreamEvent> {
    const res = await fetch(`${SSE_BASE}/chat/stream`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...authHeaders(),
      },
      body: JSON.stringify(data),
      signal,
    })
    if (res.status === 401) {
      signOut()
      window.location.href = "/signin"
      throw new Error("Session expired")
    }
    if (!res.ok) {
      const error = await res.json().catch(() => ({ detail: res.statusText }))
      throw new Error(error.detail || `Request failed: ${res.status}`)
    }

    const reader = res.body!.getReader()
    const decoder = new TextDecoder()
    let buffer = ""

    try {
      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })

        // Split on SSE message boundaries (\n\n)
        // Keep the trailing incomplete chunk in buffer
        const blocks = buffer.split("\n\n")
        buffer = blocks.pop() ?? ""

        let yieldCount = 0
        for (const block of blocks) {
          if (!block.trim()) continue
          let eventType = ""
          let dataStr = ""
          for (const line of block.split("\n")) {
            if (line.startsWith("event: ")) eventType = line.slice(7).trim()
            else if (line.startsWith("data: ")) dataStr = line.slice(6)
          }
          if (!eventType || !dataStr) continue
          try {
            const parsed = JSON.parse(dataStr)
            yield { type: eventType, ...parsed } as StreamEvent
            // Yield to the event loop every few token events so React can render
            // between batches instead of processing the entire HTTP chunk synchronously
            if (eventType === "token" && ++yieldCount % 3 === 0) {
              await new Promise(resolve => setTimeout(resolve, 0))
            }
          } catch {
            // malformed JSON — skip
          }
        }
      }
    } finally {
      reader.releaseLock()
    }
  },

  seedSession(sessionId: string, onboarding: OnboardingData, parentName?: string): Promise<{ status: string }> {
    return request("/session/seed", {
      method: "POST",
      body: JSON.stringify({
        session_id: sessionId,
        parent_name: parentName ?? "",
        child_name: onboarding.childName,
        child_age: onboarding.childAge,
        diagnosis_status: onboarding.diagnosisStatus,
        adhd_subtype: onboarding.adhdSubtype,
        challenges: onboarding.challenges,
        tried_strategies: onboarding.triedStrategies,
        goals: onboarding.goals,
      }),
    })
  },

  getSession(sessionId: string): Promise<SessionResponse> {
    return request(`/session/${sessionId}`)
  },

  getOutcomes(sessionId: string): Promise<OutcomesResponse> {
    return request(`/session/${sessionId}/outcomes`)
  },

  getHealth(): Promise<{ status: string; index_built: boolean }> {
    return request("/health")
  },

  getTopics(): Promise<{ topics: string[]; document_count: number }> {
    return request("/knowledge/topics")
  },

  getDocuments(): Promise<{ documents: KnowledgeDocument[] }> {
    return request("/knowledge/documents")
  },

  // Session management
  listSessions(): Promise<SessionsResponse> {
    return request("/sessions")
  },

  getSessionMessages(sessionId: string): Promise<MessagesResponse> {
    return request(`/session/${sessionId}/messages`)
  },

  // Observability
  getObservabilitySessions(): Promise<SessionListResponse> {
    return request("/observability/sessions")
  },

  getSessionDetail(sessionId: string): Promise<SessionDetailResponse> {
    return request(`/observability/sessions/${sessionId}`)
  },

  getSessionEvents(sessionId: string, category?: string): Promise<{ events: ObservabilityEvent[] }> {
    const params = category ? `?category=${category}` : ""
    return request(`/observability/sessions/${sessionId}/events${params}`)
  },

  analyzeSession(sessionId: string): Promise<{ status: string; turns_analyzed: number; total_flags: number }> {
    return request(`/observability/sessions/${sessionId}/analyze`, { method: "POST" })
  },

  async uploadFile(
    file: File,
    sessionId: string,
  ): Promise<UploadResponse | UploadBlockedResponse> {
    const formData = new FormData()
    formData.append("file", file)
    formData.append("session_id", sessionId)

    const res = await fetch(`${SSE_BASE}/upload`, {
      method: "POST",
      headers: authHeaders(),
      body: formData,
    })
    if (res.status === 401) {
      signOut()
      window.location.href = "/signin"
      throw new Error("Session expired")
    }
    if (!res.ok) {
      const error = await res.json().catch(() => ({ detail: res.statusText }))
      throw new Error(error.detail || `Upload failed: ${res.status}`)
    }
    return res.json()
  },

  deleteSession(sessionId: string): Promise<{ status: string }> {
    return request(`/session/${sessionId}`, { method: "DELETE" })
  },
}
