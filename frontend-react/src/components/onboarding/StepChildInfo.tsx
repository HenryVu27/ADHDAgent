import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import type { OnboardingData } from "@/types"

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
          <Label htmlFor="childAge">Child's age</Label>
          <Input
            id="childAge"
            type="number"
            min={1}
            max={18}
            placeholder="e.g., 7"
            value={data.childAge}
            onChange={(e) => onChange({ childAge: e.target.value })}
          />
        </div>
      </div>
    </div>
  )
}
