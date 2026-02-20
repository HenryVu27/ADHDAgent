import { Check } from "lucide-react"
import type { ConversationPhase } from "@/types"

const phases: { key: ConversationPhase; label: string }[] = [
  { key: "intake", label: "Getting to Know You" },
  { key: "strategy", label: "Exploring Strategies" },
  { key: "progress", label: "Building a Plan" },
]

const phaseOrder: Record<ConversationPhase, number> = {
  intake: 0,
  strategy: 1,
  progress: 2,
  followup: 2,
}

interface Props {
  currentPhase: ConversationPhase
}

export function SessionProgress({ currentPhase }: Props) {
  const currentIndex = phaseOrder[currentPhase] ?? 0

  return (
    <div className="flex items-center gap-2">
      {phases.map((phase, i) => {
        const isCompleted = i < currentIndex
        const isCurrent = i === currentIndex

        return (
          <div key={phase.key} className="flex items-center gap-2">
            {i > 0 && (
              <div
                className={`h-px w-4 sm:w-8 ${
                  isCompleted ? "bg-success" : "bg-border"
                }`}
              />
            )}
            <div className="flex items-center gap-1.5">
              <div
                className={`flex h-6 w-6 shrink-0 items-center justify-center rounded-full text-xs font-medium transition-colors ${
                  isCompleted
                    ? "bg-success text-success-foreground"
                    : isCurrent
                      ? "bg-primary text-primary-foreground"
                      : "bg-muted text-muted-foreground"
                }`}
              >
                {isCompleted ? (
                  <Check className="h-3.5 w-3.5" />
                ) : (
                  i + 1
                )}
              </div>
              <span
                className={`hidden text-xs sm:inline ${
                  isCurrent
                    ? "font-medium text-foreground"
                    : "text-muted-foreground"
                }`}
              >
                {phase.label}
              </span>
            </div>
          </div>
        )
      })}
    </div>
  )
}
