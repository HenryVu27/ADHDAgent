import { Link } from "react-router-dom"
import { Target, ArrowRight } from "lucide-react"
import { Card, CardContent } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import type { Goal } from "@/types"

interface Props {
  goals: Goal[]
}

export function ActiveGoals({ goals }: Props) {
  const activeGoals = goals.filter((g) => g.status === "active")

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2">
        <Target className="h-4 w-4 text-primary" />
        <h2 className="font-semibold tracking-tight">Active Goals</h2>
      </div>

      {activeGoals.length === 0 ? (
        <Card className="border-dashed">
          <CardContent className="flex flex-col items-center py-8 text-center">
            <Target className="mb-3 h-8 w-8 text-muted-foreground/40" />
            <p className="mb-1 text-sm font-medium">No goals yet</p>
            <p className="mb-4 text-xs text-muted-foreground">
              Chat with Ally to set your first goal.
            </p>
            <Link to="/chat">
              <Button size="sm" className="gap-2">
                Chat with Ally
                <ArrowRight className="h-3.5 w-3.5" />
              </Button>
            </Link>
          </CardContent>
        </Card>
      ) : (
        <div className="space-y-2">
          {activeGoals.map((goal, i) => (
            <Card key={i}>
              <CardContent className="flex items-center justify-between p-4">
                <div className="min-w-0 flex-1">
                  <p className="text-sm font-medium truncate">{goal.description}</p>
                  {goal.strategy_id && (
                    <p className="text-xs text-muted-foreground">
                      Strategy: {goal.strategy_id}
                    </p>
                  )}
                </div>
                <Badge variant="secondary" className="ml-2 shrink-0 bg-primary/10 text-primary">
                  {goal.status}
                </Badge>
              </CardContent>
            </Card>
          ))}
        </div>
      )}
    </div>
  )
}
