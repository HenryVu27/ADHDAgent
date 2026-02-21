import { useState } from "react"
import { motion, AnimatePresence } from "framer-motion"
import { ArrowLeft, ArrowRight, Check } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Progress } from "@/components/ui/progress"
import { StepChildInfo } from "./StepChildInfo"
import { StepChallenges } from "./StepChallenges"
import { StepStrategies } from "./StepStrategies"
import { StepGoals } from "./StepGoals"
import type { OnboardingData } from "@/types"

const STEPS = ["Child Info", "Challenges", "Strategies", "Goals"]

interface Props {
  onComplete: (data: OnboardingData) => void
}

export function OnboardingWizard({ onComplete }: Props) {
  const [step, setStep] = useState(0)
  const [data, setData] = useState<OnboardingData>({
    childName: "",
    childAge: "",
    challenges: [],
    triedStrategies: [],
    goals: [],
  })

  const update = (updates: Partial<OnboardingData>) => {
    setData((prev) => ({ ...prev, ...updates }))
  }

  const canProceed = () => {
    switch (step) {
      case 0:
        return data.childAge !== ""
      case 1:
        return data.challenges.length > 0
      case 2:
        return data.triedStrategies.length > 0
      case 3:
        return data.goals.length > 0
      default:
        return false
    }
  }

  const handleNext = () => {
    if (step < STEPS.length - 1) {
      setStep(step + 1)
    } else {
      onComplete(data)
    }
  }

  const progress = ((step + 1) / STEPS.length) * 100

  return (
    <div className="mx-auto max-w-lg space-y-8">
      {/* Progress */}
      <div className="space-y-3">
        <div className="flex items-center justify-between text-sm text-muted-foreground">
          <span>Step {step + 1} of {STEPS.length}</span>
          <span>{STEPS[step]}</span>
        </div>
        <Progress value={progress} className="h-2" />
      </div>

      {/* Step Content */}
      <AnimatePresence mode="wait">
        <motion.div
          key={step}
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          exit={{ opacity: 0, x: -20 }}
          transition={{ duration: 0.25 }}
        >
          {step === 0 && <StepChildInfo data={data} onChange={update} />}
          {step === 1 && <StepChallenges data={data} onChange={update} />}
          {step === 2 && <StepStrategies data={data} onChange={update} />}
          {step === 3 && <StepGoals data={data} onChange={update} />}
        </motion.div>
      </AnimatePresence>

      {/* Navigation */}
      <div className="flex items-center justify-between">
        <Button
          variant="ghost"
          onClick={() => setStep(step - 1)}
          disabled={step === 0}
          className="gap-2"
        >
          <ArrowLeft className="h-4 w-4" />
          Back
        </Button>
        <Button onClick={handleNext} disabled={!canProceed()} className="gap-2">
          {step === STEPS.length - 1 ? (
            <>
              <Check className="h-4 w-4" />
              Finish Setup
            </>
          ) : (
            <>
              Next
              <ArrowRight className="h-4 w-4" />
            </>
          )}
        </Button>
      </div>
    </div>
  )
}
