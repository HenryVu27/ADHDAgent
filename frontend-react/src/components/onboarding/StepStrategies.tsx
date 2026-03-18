import { Checkbox } from "@/components/ui/checkbox"
import { Label } from "@/components/ui/label"
import type { OnboardingData } from "@/types"

const strategies = [
  { id: "visual_schedules", label: "Visual schedules" },
  { id: "timers", label: "Timers & countdowns" },
  { id: "reward_charts", label: "Reward/sticker charts" },
  { id: "calm_corner", label: "Calm-down corner" },
  { id: "specific_praise", label: "Specific praise" },
  { id: "chunking", label: "Breaking tasks into steps" },
  { id: "none", label: "Haven't tried any yet" },
]

interface Props {
  data: OnboardingData
  onChange: (updates: Partial<OnboardingData>) => void
}

export function StepStrategies({ data, onChange }: Props) {
  const toggle = (id: string) => {
    if (id === "none") {
      onChange({ triedStrategies: data.triedStrategies.includes("none") ? [] : ["none"] })
      return
    }
    const current = data.triedStrategies.filter((s) => s !== "none")
    const next = current.includes(id)
      ? current.filter((s) => s !== id)
      : [...current, id]
    onChange({ triedStrategies: next })
  }

  return (
    <div className="space-y-6">
      <div className="text-center">
        <h2 className="text-xl font-semibold tracking-tight">What have you tried so far?</h2>
        <p className="mt-1 text-sm text-muted-foreground">
          Select strategies you've already used — we'll build on what works.
        </p>
      </div>

      <div className="grid gap-3 sm:grid-cols-2">
        {strategies.map((s) => {
          const checked = data.triedStrategies.includes(s.id)
          return (
            <button
              key={s.id}
              type="button"
              onClick={() => toggle(s.id)}
              className={`flex items-center gap-3 rounded-lg border px-4 py-3 text-left transition-colors ${
                checked
                  ? "border-primary bg-primary/5"
                  : "border-border hover:border-primary/50 hover:bg-muted/50"
              }`}
            >
              <Checkbox checked={checked} />
              <Label className="cursor-pointer">{s.label}</Label>
            </button>
          )
        })}
      </div>
    </div>
  )
}
