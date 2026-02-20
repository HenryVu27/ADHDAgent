import type { ConversationPhase } from "@/types"

const phaseConfig: Record<ConversationPhase, { label: string; style: string }> = {
  intake: { label: "Getting to Know You", style: "bg-coach/15 text-coach" },
  strategy: { label: "Exploring Strategies", style: "bg-primary/10 text-primary" },
  progress: { label: "Tracking Progress", style: "bg-success/15 text-success" },
  followup: { label: "Following Up", style: "bg-accent/30 text-accent-foreground" },
}

interface Props {
  currentPhase: ConversationPhase
}

export function SessionProgress({ currentPhase }: Props) {
  const config = phaseConfig[currentPhase] ?? phaseConfig.intake

  return (
    <div className="flex items-center gap-2">
      <span className="text-sm font-semibold tracking-tight">Coaching Session</span>
      <span className={`rounded-full px-3 py-0.5 text-xs font-medium ${config.style}`}>
        {config.label}
      </span>
    </div>
  )
}
