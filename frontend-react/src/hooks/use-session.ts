import { useState, useCallback, useEffect } from "react"
import type { SessionResponse } from "@/types"
import { api } from "@/lib/api"

export function useSession(sessionId: string) {
  const [session, setSession] = useState<SessionResponse | null>(null)
  const [isLoading, setIsLoading] = useState(false)

  // Reset session when switching to a new session
  useEffect(() => {
    setSession(null)
  }, [sessionId])

  const refresh = useCallback(async () => {
    setIsLoading(true)
    try {
      const data = await api.getSession(sessionId)
      setSession(data)
      return data
    } catch {
      // Session may not exist yet
      return null
    } finally {
      setIsLoading(false)
    }
  }, [sessionId])

  return { session, isLoading, refresh }
}
