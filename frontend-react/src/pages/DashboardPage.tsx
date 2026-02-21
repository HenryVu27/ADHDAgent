import { useState, useEffect } from "react"
import { GreetingHeader } from "@/components/dashboard/GreetingHeader"
import { ActionCards } from "@/components/dashboard/ActionCards"
import { ActiveGoals } from "@/components/dashboard/ActiveGoals"
import { RecentSession } from "@/components/dashboard/RecentSession"
import { SessionHistory } from "@/components/dashboard/SessionHistory"
import { QuickTips } from "@/components/dashboard/QuickTips"
import { ProgressStats } from "@/components/dashboard/ProgressStats"
import { Separator } from "@/components/ui/separator"
import { api } from "@/lib/api"
import type { SessionListItem, ConversationPhase } from "@/types"

export function DashboardPage() {
  const [sessions, setSessions] = useState<SessionListItem[]>([])

  useEffect(() => {
    api.listSessions()
      .then((data) => setSessions(data.sessions))
      .catch(() => {})
  }, [])

  const recent = sessions.find((s) => s.turn_count > 0) || null

  return (
    <div className="space-y-8">
      <GreetingHeader />
      <ActionCards />

      {recent && (
        <RecentSession
          sessionId={recent.session_id}
          phase={recent.phase as ConversationPhase}
          turnCount={recent.turn_count}
        />
      )}

      {sessions.length > 1 && (
        <SessionHistory sessions={sessions} />
      )}

      <Separator />

      <div className="grid gap-8 lg:grid-cols-3">
        <div className="lg:col-span-2 space-y-8">
          <ActiveGoals goals={[]} />
          <QuickTips />
        </div>
        <div>
          <ProgressStats />
        </div>
      </div>
    </div>
  )
}
