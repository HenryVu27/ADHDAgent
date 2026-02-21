import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import type { SessionOverview } from "@/types"

interface SessionListProps {
  sessions: SessionOverview[]
  onSelect: (sessionId: string) => void
}

function flagBadgeColor(count: number) {
  if (count === 0) return "bg-emerald-100 text-emerald-700"
  if (count <= 2) return "bg-amber-100 text-amber-700"
  return "bg-red-100 text-red-700"
}

function qualityColor(score: number) {
  if (score > 0.8) return "text-emerald-600"
  if (score >= 0.5) return "text-amber-600"
  return "text-red-600"
}

export function SessionList({ sessions, onSelect }: SessionListProps) {
  if (sessions.length === 0) {
    return (
      <Card className="p-8 text-center text-muted-foreground">
        No sessions found. Start a conversation to see data here.
      </Card>
    )
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b text-left text-muted-foreground">
            <th className="pb-3 pr-4 font-medium">Session ID</th>
            <th className="pb-3 pr-4 font-medium">Turns</th>
            <th className="pb-3 pr-4 font-medium">Flags</th>
            <th className="pb-3 pr-4 font-medium">Quality</th>
            <th className="pb-3 pr-4 font-medium">Tools</th>
            <th className="pb-3 pr-4 font-medium">Blocked</th>
            <th className="pb-3 font-medium">Actions</th>
          </tr>
        </thead>
        <tbody>
          {sessions.map((s) => (
            <tr key={s.session_id} className="border-b last:border-0 hover:bg-muted/50 transition-colors">
              <td className="py-3 pr-4 font-mono text-xs">{s.session_id}</td>
              <td className="py-3 pr-4">{s.turn_count}</td>
              <td className="py-3 pr-4">
                <Badge variant="secondary" className={flagBadgeColor(s.total_flags)}>
                  {s.total_flags}
                </Badge>
              </td>
              <td className={`py-3 pr-4 font-medium ${qualityColor(s.avg_quality_score)}`}>
                {s.avg_quality_score.toFixed(2)}
              </td>
              <td className="py-3 pr-4">{s.tool_calls_count}</td>
              <td className="py-3 pr-4">{s.blocked_count}</td>
              <td className="py-3">
                <Button variant="ghost" size="sm" onClick={() => onSelect(s.session_id)}>
                  View
                </Button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
