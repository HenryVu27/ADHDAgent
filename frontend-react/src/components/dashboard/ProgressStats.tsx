import { MessageCircle, Target, Flame } from "lucide-react"
import { Card, CardContent } from "@/components/ui/card"
import { getSessionStats } from "@/lib/auth"

export function ProgressStats() {
  const stats = getSessionStats()

  const items = [
    { label: "Sessions", value: stats.sessions, icon: MessageCircle, color: "text-primary" },
    { label: "Strategies", value: stats.strategies, icon: Target, color: "text-coach" },
    { label: "Day Streak", value: stats.streak, icon: Flame, color: "text-accent-foreground" },
  ]

  return (
    <div className="space-y-3">
      <h2 className="font-semibold tracking-tight">Your Progress</h2>
      <div className="grid grid-cols-3 gap-3">
        {items.map(({ label, value, icon: Icon, color }) => (
          <Card key={label}>
            <CardContent className="flex flex-col items-center p-4 text-center">
              <Icon className={`mb-2 h-5 w-5 ${color}`} />
              <span className="text-2xl font-bold tracking-tight">{value}</span>
              <span className="text-xs text-muted-foreground">{label}</span>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  )
}
