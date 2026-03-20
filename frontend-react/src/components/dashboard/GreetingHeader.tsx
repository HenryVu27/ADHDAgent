import { useAuth } from "@/hooks/use-auth"

function getGreeting(): string {
  const hour = new Date().getHours()
  if (hour < 12) return "Good morning"
  if (hour < 17) return "Good afternoon"
  return "Good evening"
}

export function GreetingHeader() {
  const { user, getOnboarding } = useAuth()
  const firstName = user?.name?.split(" ")[0] ?? "there"
  const onboarding = getOnboarding()
  const childName = onboarding?.childName

  return (
    <div className="rounded-2xl bg-gradient-to-r from-primary/8 via-accent/8 to-transparent px-6 py-5">
      <h1 className="text-2xl font-bold tracking-tight md:text-3xl">
        {getGreeting()}, {firstName}!
      </h1>
      <p className="text-muted-foreground">
        {childName
          ? `How are things going with ${childName}? Let's keep making progress.`
          : "Here's your coaching hub. What would you like to focus on today?"}
      </p>
    </div>
  )
}
