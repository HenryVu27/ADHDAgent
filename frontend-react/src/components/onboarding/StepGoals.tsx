import { Checkbox } from "@/components/ui/checkbox"
import { Label } from "@/components/ui/label"
import type { OnboardingData } from "@/types"

const goalOptions = [
  "Smoother morning routines",
  "Less homework stress",
  "Better emotional regulation",
  "Calmer transitions",
  "Improved focus at home",
  "Stronger parent-child relationship",
  "Better sleep/bedtime routine",
  "More cooperation with tasks",
]

interface Props {
  data: OnboardingData
  onChange: (updates: Partial<OnboardingData>) => void
}

export function StepGoals({ data, onChange }: Props) {
  const toggle = (goal: string) => {
    const current = data.goals
    const next = current.includes(goal)
      ? current.filter((g) => g !== goal)
      : [...current, goal]
    onChange({ goals: next })
  }

  return (
    <div className="space-y-6">
      <div className="text-center">
        <h2 className="text-xl font-semibold tracking-tight">What are your goals?</h2>
        <p className="mt-1 text-sm text-muted-foreground">
          Pick the outcomes that matter most to your family.
        </p>
      </div>

      <div className="grid gap-3 sm:grid-cols-2">
        {goalOptions.map((goal) => {
          const checked = data.goals.includes(goal)
          return (
            <button
              key={goal}
              type="button"
              onClick={() => toggle(goal)}
              className={`flex items-center gap-3 rounded-lg border px-4 py-3 text-left transition-colors ${
                checked
                  ? "border-primary bg-primary/5"
                  : "border-border hover:border-primary/50 hover:bg-muted/50"
              }`}
            >
              <Checkbox checked={checked} />
              <Label className="cursor-pointer">{goal}</Label>
            </button>
          )
        })}
      </div>
    </div>
  )
}
