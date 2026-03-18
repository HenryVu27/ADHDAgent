import { useNavigate, Navigate } from "react-router-dom"
import { Sprout } from "lucide-react"
import { toast } from "sonner"
import { OnboardingWizard } from "@/components/onboarding/OnboardingWizard"
import { useAuth } from "@/hooks/use-auth"
import type { OnboardingData } from "@/types"

export function OnboardingPage() {
  const { isAuthenticated, hasOnboarded, saveOnboarding } = useAuth()
  const navigate = useNavigate()

  if (!isAuthenticated) {
    return <Navigate to="/" replace />
  }

  if (hasOnboarded) {
    return <Navigate to="/dashboard" replace />
  }

  const handleComplete = (data: OnboardingData) => {
    saveOnboarding(data)
    toast.success("You're all set! Ally is ready to help your family.")
    navigate("/dashboard")
  }

  return (
    <div className="flex min-h-screen flex-col items-center bg-background px-4 py-12">
      <div className="mb-8 flex items-center gap-3">
        <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-coach shadow-md">
          <Sprout className="h-6 w-6 text-coach-foreground" />
        </div>
      </div>
      <OnboardingWizard onComplete={handleComplete} />
    </div>
  )
}
