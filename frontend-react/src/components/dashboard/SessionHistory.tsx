import { Link } from "react-router-dom"
import { History, ArrowRight } from "lucide-react"
import { Card, CardContent } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { setActiveSessionId } from "@/lib/auth"
import type { SessionListItem } from "@/types"

const phaseLabels: Record<string, string> = {
  intake: "Getting to Know You",
  strategy: "Exploring Strategies",
  progress: "Building a Plan",
  followup: "Following Up",
}

interface Props {
  sessions: SessionListItem[]
}

export function SessionHistory({ sessions }: Props) {
  // Skip the first (most recent) since RecentSession handles it
  const past = sessions.filter((s) => s.turn_count > 0).slice(1, 6)
  if (past.length === 0) return null

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2">
        <History className="h-4 w-4 text-muted-foreground" />
        <h2 className="font-semibold tracking-tight">Past Sessions</h2>
      </div>
      <div className="space-y-2">
        {past.map((s) => (
          <Card key={s.session_id}>
            <CardContent className="flex items-center justify-between p-3">
              <div className="flex items-center gap-3">
                <div>
                  <div className="flex items-center gap-2">
                    <Badge variant="secondary" className="text-xs">
                      {phaseLabels[s.phase] || s.phase}
                    </Badge>
                    <span className="text-xs text-muted-foreground">
                      {s.turn_count} messages
                    </span>
                  </div>
                  {s.created_at && (
                    <p className="mt-0.5 text-xs text-muted-foreground">
                      {new Date(s.created_at).toLocaleDateString(undefined, {
                        month: "short",
                        day: "numeric",
                        hour: "numeric",
                        minute: "2-digit",
                      })}
                    </p>
                  )}
                </div>
              </div>
              <Link
                to="/chat"
                onClick={() => setActiveSessionId(s.session_id)}
              >
                <Button size="sm" variant="ghost" className="gap-1.5 text-xs">
                  Resume
                  <ArrowRight className="h-3 w-3" />
                </Button>
              </Link>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  )
}
