import { User, Calendar, Target, Heart, Brain } from "lucide-react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Separator } from "@/components/ui/separator"
import { useAuth } from "@/hooks/use-auth"
import { getSessionStats } from "@/lib/auth"

export function ProfilePage() {
  const { user, getOnboarding } = useAuth()
  const onboarding = getOnboarding()
  const stats = getSessionStats()

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Profile</h1>
        <p className="text-muted-foreground">Your account and family information.</p>
      </div>

      <div className="grid gap-6 md:grid-cols-2">
        {/* Account Info */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-base">
              <User className="h-4 w-4 text-primary" />
              Account
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="flex justify-between text-sm">
              <span className="text-muted-foreground">Name</span>
              <span className="font-medium">{user?.name}</span>
            </div>
            <div className="flex justify-between text-sm">
              <span className="text-muted-foreground">Email</span>
              <span className="font-medium">{user?.email}</span>
            </div>
            <div className="flex justify-between text-sm">
              <span className="text-muted-foreground">Joined</span>
              <span className="font-medium">
                {user?.createdAt
                  ? new Date(user.createdAt).toLocaleDateString()
                  : "\u2014"}
              </span>
            </div>
          </CardContent>
        </Card>

        {/* Stats */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-base">
              <Target className="h-4 w-4 text-coach" />
              Activity
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="flex justify-between text-sm">
              <span className="text-muted-foreground">Coaching sessions</span>
              <span className="font-medium">{stats.sessions}</span>
            </div>
            <div className="flex justify-between text-sm">
              <span className="text-muted-foreground">Strategies explored</span>
              <span className="font-medium">{stats.strategies}</span>
            </div>
            <div className="flex justify-between text-sm">
              <span className="text-muted-foreground">Day streak</span>
              <span className="font-medium">{stats.streak}</span>
            </div>
          </CardContent>
        </Card>
      </div>

      {onboarding && (
        <>
          <Separator />
          <div className="grid gap-6 md:grid-cols-2">
            {/* Child Info */}
            <Card className="border-t-[3px] border-t-coach">
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-base">
                  <Heart className="h-4 w-4 text-primary" />
                  {onboarding.childName
                    ? `${onboarding.childName}'s Profile`
                    : "Child Information"}
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                {onboarding.childName && (
                  <div className="flex justify-between text-sm">
                    <span className="text-muted-foreground">Name</span>
                    <span className="font-medium">{onboarding.childName}</span>
                  </div>
                )}
                <div className="flex justify-between text-sm">
                  <span className="text-muted-foreground">Age range</span>
                  <span className="font-medium">{onboarding.childAge}</span>
                </div>
              </CardContent>
            </Card>

            {/* Challenges & Goals */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-base">
                  <Brain className="h-4 w-4 text-accent-foreground" />
                  Focus Areas
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                {onboarding.challenges.length > 0 && (
                  <div>
                    <p className="mb-2 text-xs font-medium text-muted-foreground uppercase tracking-wider">
                      Challenges
                    </p>
                    <div className="flex flex-wrap gap-1.5">
                      {onboarding.challenges.map((c) => (
                        <Badge key={c} variant="secondary" className="rounded-full bg-primary/10 text-primary">
                          {c.replace(/_/g, " ")}
                        </Badge>
                      ))}
                    </div>
                  </div>
                )}
                {onboarding.goals.length > 0 && (
                  <div>
                    <p className="mb-2 text-xs font-medium text-muted-foreground uppercase tracking-wider">
                      Goals
                    </p>
                    <div className="flex flex-wrap gap-1.5">
                      {onboarding.goals.map((g) => (
                        <Badge key={g} variant="outline" className="rounded-full">
                          {g}
                        </Badge>
                      ))}
                    </div>
                  </div>
                )}
                {onboarding.triedStrategies.length > 0 && (
                  <div>
                    <p className="mb-2 text-xs font-medium text-muted-foreground uppercase tracking-wider">
                      Strategies Tried
                    </p>
                    <div className="flex flex-wrap gap-1.5">
                      {onboarding.triedStrategies.map((s) => (
                        <Badge key={s} variant="secondary" className="rounded-full bg-coach/10 text-coach">
                          {s.replace(/_/g, " ")}
                        </Badge>
                      ))}
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>

            {/* Calendar placeholder */}
            <Card className="md:col-span-2">
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-base">
                  <Calendar className="h-4 w-4 text-muted-foreground" />
                  Coaching History
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="flex h-24 items-center justify-center rounded-lg border border-dashed border-border text-sm text-muted-foreground">
                  Coaching session history will appear here as you use the app.
                </div>
              </CardContent>
            </Card>
          </div>
        </>
      )}
    </div>
  )
}
