import type { OnboardingData } from "@/types"

const options = [
  { id: "diagnosed", label: "Diagnosed", desc: "Formally diagnosed by a professional" },
  { id: "suspected", label: "Suspected", desc: "We think it might be ADHD but no formal diagnosis" },
  { id: "evaluating", label: "Being evaluated", desc: "Currently going through the evaluation process" },
  { id: "not_sure", label: "Not sure", desc: "We're still figuring things out" },
]

interface Props {
  data: OnboardingData
  onChange: (updates: Partial<OnboardingData>) => void
}

export function StepDiagnosis({ data, onChange }: Props) {
  return (
    <div className="space-y-6">
      <div className="text-center">
        <h2 className="text-xl font-semibold tracking-tight">What's the diagnosis status?</h2>
        <p className="mt-1 text-sm text-muted-foreground">
          This helps us tailor our coaching to where you are in your journey.
        </p>
      </div>

      <div className="grid gap-3">
        {options.map((opt) => (
          <button
            key={opt.id}
            type="button"
            onClick={() => onChange({ diagnosisStatus: opt.id })}
            className={`rounded-lg border px-4 py-4 text-left transition-colors ${
              data.diagnosisStatus === opt.id
                ? "border-primary bg-primary/5 text-primary"
                : "border-border hover:border-primary/50 hover:bg-muted/50"
            }`}
          >
            <span className="font-medium">{opt.label}</span>
            <p className="mt-0.5 text-xs text-muted-foreground">{opt.desc}</p>
          </button>
        ))}
      </div>
    </div>
  )
}
