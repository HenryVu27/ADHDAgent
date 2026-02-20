import { GreetingHeader } from "@/components/dashboard/GreetingHeader"
import { ActionCards } from "@/components/dashboard/ActionCards"
import { ActiveGoals } from "@/components/dashboard/ActiveGoals"
import { RecentSession } from "@/components/dashboard/RecentSession"
import { QuickTips } from "@/components/dashboard/QuickTips"
import { ProgressStats } from "@/components/dashboard/ProgressStats"
import { Separator } from "@/components/ui/separator"
import { getSessionStats } from "@/lib/auth"

export function DashboardPage() {
  const stats = getSessionStats()

  return (
    <div className="space-y-8">
      <GreetingHeader />
      <ActionCards />

      {stats.sessions > 0 && (
        <RecentSession phase={null} turnCount={0} />
      )}

      <Separator />

      <div className="grid gap-8 lg:grid-cols-3">
        <div className="lg:col-span-2 space-y-8">
          <ActiveGoals goals={[]} />
          <QuickTips />
        </div>
        <div>
          <ProgressStats />
        </div>
      </div>
    </div>
  )
}
