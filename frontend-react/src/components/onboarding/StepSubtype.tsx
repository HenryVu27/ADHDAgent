import type { OnboardingData } from "@/types"

const subtypes = [
  { id: "inattentive", label: "Inattentive", desc: "Difficulty focusing, easily distracted, forgetful" },
  { id: "hyperactive-impulsive", label: "Hyperactive-Impulsive", desc: "Restless, fidgety, acts without thinking" },
  { id: "combined", label: "Combined", desc: "Mix of inattentive and hyperactive-impulsive traits" },
  { id: "not_sure", label: "Not sure", desc: "We don't know the specific type" },
]

interface Props {
  data: OnboardingData
  onChange: (updates: Partial<OnboardingData>) => void
}

export function StepSubtype({ data, onChange }: Props) {
  return (
    <div className="space-y-6">
      <div className="text-center">
        <h2 className="text-xl font-semibold tracking-tight">What type of ADHD?</h2>
        <p className="mt-1 text-sm text-muted-foreground">
          Different types benefit from different strategies. It's okay if you're not sure.
        </p>
      </div>

      <div className="grid gap-3">
        {subtypes.map((sub) => (
          <button
            key={sub.id}
            type="button"
            onClick={() => onChange({ adhdSubtype: sub.id })}
            className={`rounded-lg border px-4 py-4 text-left transition-colors ${
              data.adhdSubtype === sub.id
                ? "border-primary bg-primary/5 text-primary"
                : "border-border hover:border-primary/50 hover:bg-muted/50"
            }`}
          >
            <span className="font-medium">{sub.label}</span>
            <p className="mt-0.5 text-xs text-muted-foreground">{sub.desc}</p>
          </button>
        ))}
      </div>
    </div>
  )
}
