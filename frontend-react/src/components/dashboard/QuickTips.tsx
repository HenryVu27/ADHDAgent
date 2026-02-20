import { Lightbulb } from "lucide-react"
import { Card, CardContent } from "@/components/ui/card"

const tips = [
  {
    title: "Start with connection",
    body: "Before giving instructions, make eye contact and use your child's name. A 10-second connection moment increases cooperation.",
  },
  {
    title: "Praise the specific",
    body: 'Instead of "good job," try "I noticed you put your shoes on without being asked \u2014 that took initiative!"',
  },
  {
    title: "Transitions need warnings",
    body: "Give a 5-minute and 2-minute warning before activity changes. Visual timers make this even more effective.",
  },
]

export function QuickTips() {
  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2">
        <Lightbulb className="h-4 w-4 text-warm" />
        <h2 className="font-semibold tracking-tight">Quick Tips</h2>
      </div>
      <div className="grid gap-3 sm:grid-cols-3">
        {tips.map((tip) => (
          <Card
            key={tip.title}
            className="transition-shadow duration-200 hover:shadow-sm"
          >
            <CardContent className="p-4">
              <h3 className="mb-1 text-sm font-medium">{tip.title}</h3>
              <p className="text-xs text-muted-foreground leading-relaxed">{tip.body}</p>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  )
}
