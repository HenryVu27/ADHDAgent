import { Checkbox } from "@/components/ui/checkbox"
import { Label } from "@/components/ui/label"
import type { OnboardingData } from "@/types"

const challenges = [
  { id: "homework", label: "Homework struggles", desc: "Difficulty starting or completing assignments" },
  { id: "transitions", label: "Transitions", desc: "Meltdowns when switching activities" },
  { id: "emotions", label: "Emotional regulation", desc: "Big reactions, frustration, anger" },
  { id: "focus", label: "Focus & attention", desc: "Easily distracted, can't sit still" },
  { id: "social", label: "Social skills", desc: "Difficulty with peers, impulsive behavior" },
  { id: "routines", label: "Daily routines", desc: "Morning chaos, bedtime battles" },
  { id: "screen_time", label: "Screen time", desc: "Difficulty disengaging from screens" },
  { id: "self_esteem", label: "Self-esteem", desc: "Negative self-talk, feeling different" },
]

interface Props {
  data: OnboardingData
  onChange: (updates: Partial<OnboardingData>) => void
}

export function StepChallenges({ data, onChange }: Props) {
  const toggle = (id: string) => {
    const current = data.challenges
    const next = current.includes(id)
      ? current.filter((c) => c !== id)
      : [...current, id]
    onChange({ challenges: next })
  }

  return (
    <div className="space-y-6">
      <div className="text-center">
        <h2 className="text-xl font-semibold tracking-tight">What are the biggest challenges?</h2>
        <p className="mt-1 text-sm text-muted-foreground">
          Select all that apply. This helps us focus on what matters most.
        </p>
      </div>

      <div className="grid gap-3 sm:grid-cols-2">
        {challenges.map((c) => {
          const checked = data.challenges.includes(c.id)
          return (
            <button
              key={c.id}
              type="button"
              onClick={() => toggle(c.id)}
              className={`flex items-start gap-3 rounded-lg border p-4 text-left transition-colors ${
                checked
                  ? "border-primary bg-primary/5"
                  : "border-border hover:border-primary/50 hover:bg-muted/50"
              }`}
            >
              <Checkbox checked={checked} className="mt-0.5" />
              <div>
                <Label className="cursor-pointer font-medium">{c.label}</Label>
                <p className="text-xs text-muted-foreground">{c.desc}</p>
              </div>
            </button>
          )
        })}
      </div>
    </div>
  )
}
