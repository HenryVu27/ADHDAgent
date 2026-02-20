import { Link } from "react-router-dom"
import { Clock, ArrowRight } from "lucide-react"
import { Card, CardContent } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import type { ConversationPhase } from "@/types"

const phaseLabels: Record<ConversationPhase, string> = {
  intake: "Getting to Know You",
  strategy: "Exploring Strategies",
  progress: "Building a Plan",
  followup: "Following Up",
}

interface Props {
  phase: ConversationPhase | null
  turnCount: number
}

export function RecentSession({ phase, turnCount }: Props) {
  if (!phase || turnCount === 0) return null

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2">
        <Clock className="h-4 w-4 text-muted-foreground" />
        <h2 className="font-semibold tracking-tight">Recent Session</h2>
      </div>
      <Card>
        <CardContent className="flex items-center justify-between p-4">
          <div>
            <p className="text-sm font-medium">Last coaching session</p>
            <div className="mt-1 flex items-center gap-2">
              <Badge variant="secondary" className="text-xs">
                {phaseLabels[phase]}
              </Badge>
              <span className="text-xs text-muted-foreground">
                {turnCount} messages
              </span>
            </div>
          </div>
          <Link to="/chat">
            <Button size="sm" variant="outline" className="gap-2">
              Continue
              <ArrowRight className="h-3.5 w-3.5" />
            </Button>
          </Link>
        </CardContent>
      </Card>
    </div>
  )
}
