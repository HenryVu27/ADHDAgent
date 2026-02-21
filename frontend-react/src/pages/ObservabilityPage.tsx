import { useState, useEffect, useCallback } from "react"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { SessionList } from "@/components/observability/SessionList"
import { SessionDetail } from "@/components/observability/SessionDetail"
import { api } from "@/lib/api"
import type { SessionOverview, SessionDetailResponse } from "@/types"

export function ObservabilityPage() {
  const [sessions, setSessions] = useState<SessionOverview[]>([])
  const [selectedSession, setSelectedSession] = useState<string | null>(null)
  const [sessionDetail, setSessionDetail] = useState<SessionDetailResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [analyzing, setAnalyzing] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const loadSessions = useCallback(async () => {
    try {
      setLoading(true)
      setError(null)
      const data = await api.getObservabilitySessions()
      setSessions(data.sessions)
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load sessions")
    } finally {
      setLoading(false)
    }
  }, [])

  const loadDetail = useCallback(async (sessionId: string) => {
    try {
      setLoading(true)
      setError(null)
      const data = await api.getSessionDetail(sessionId)
      setSessionDetail(data)
      setSelectedSession(sessionId)
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load session")
    } finally {
      setLoading(false)
    }
  }, [])

  const handleAnalyze = useCallback(async () => {
    if (!selectedSession) return
    try {
      setAnalyzing(true)
      await api.analyzeSession(selectedSession)
      await loadDetail(selectedSession)
    } catch (e) {
      setError(e instanceof Error ? e.message : "Analysis failed")
    } finally {
      setAnalyzing(false)
    }
  }, [selectedSession, loadDetail])

  useEffect(() => {
    loadSessions()
  }, [loadSessions])

  return (
    <div className="min-h-screen bg-background">
      <div className="max-w-6xl mx-auto px-6 py-8">
        <div className="flex items-center justify-between mb-6">
          <div>
            <h1 className="text-2xl font-semibold tracking-tight">Observability</h1>
            <p className="text-sm text-muted-foreground mt-1">
              Monitor AI conversation quality, traces, and events
            </p>
          </div>
          <Button variant="outline" size="sm" onClick={loadSessions}>
            Refresh
          </Button>
        </div>

        {error && (
          <Card className="p-4 mb-4 bg-red-50 text-red-700 text-sm">
            {error}
          </Card>
        )}

        {loading && !sessionDetail && (
          <Card className="p-8 text-center text-muted-foreground">Loading...</Card>
        )}

        {!loading && !selectedSession && (
          <SessionList sessions={sessions} onSelect={loadDetail} />
        )}

        {selectedSession && sessionDetail && (
          <SessionDetail
            data={sessionDetail}
            onBack={() => {
              setSelectedSession(null)
              setSessionDetail(null)
              loadSessions()
            }}
            onAnalyze={handleAnalyze}
            analyzing={analyzing}
          />
        )}
      </div>
    </div>
  )
}
