import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import type { OnboardingData } from "@/types"

const ageRanges = [
  "3-5 (Preschool)",
  "6-8 (Early Elementary)",
  "9-11 (Late Elementary)",
  "12-14 (Middle School)",
  "15-17 (High School)",
]

interface Props {
  data: OnboardingData
  onChange: (updates: Partial<OnboardingData>) => void
}

export function StepChildInfo({ data, onChange }: Props) {
  return (
    <div className="space-y-6">
      <div className="text-center">
        <h2 className="text-xl font-semibold tracking-tight">Tell us about your child</h2>
        <p className="mt-1 text-sm text-muted-foreground">
          This helps us personalize strategies for your family.
        </p>
      </div>

      <div className="space-y-4">
        <div className="space-y-2">
          <Label htmlFor="childName">Child's name (or nickname)</Label>
          <Input
            id="childName"
            placeholder="e.g., Alex"
            value={data.childName}
            onChange={(e) => onChange({ childName: e.target.value })}
          />
        </div>

        <div className="space-y-2">
          <Label>Age range</Label>
          <div className="grid gap-2">
            {ageRanges.map((age) => (
              <button
                key={age}
                type="button"
                onClick={() => onChange({ childAge: age })}
                className={`rounded-lg border px-4 py-3 text-left text-sm transition-colors ${
                  data.childAge === age
                    ? "border-primary bg-primary/5 text-primary"
                    : "border-border hover:border-primary/50 hover:bg-muted/50"
                }`}
              >
                {age}
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}
