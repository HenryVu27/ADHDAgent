import { useState, useEffect } from "react"
import { User, Calendar, Target, Heart, Brain, Sparkles } from "lucide-react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Separator } from "@/components/ui/separator"
import { useAuth } from "@/hooks/use-auth"
import { getSessionStats, getActiveSessionId } from "@/lib/auth"
import { api } from "@/lib/api"
import type { FamilyProfile, SessionListItem } from "@/types"

export function ProfilePage() {
  const { user, getOnboarding } = useAuth()
  const onboarding = getOnboarding()
  const stats = getSessionStats()
  const [backendProfile, setBackendProfile] = useState<FamilyProfile | null>(null)
  const [sessions, setSessions] = useState<SessionListItem[]>([])

  useEffect(() => {
    // Fetch family profile from the active session
    const activeId = getActiveSessionId()
    if (activeId) {
      api.getSession(activeId)
        .then((data) => setBackendProfile(data.family_profile))
        .catch(() => {})
    }

    // Fetch session history
    api.listSessions()
      .then((data) => setSessions(data.sessions.filter((s) => s.turn_count > 0)))
      .catch(() => {})
  }, [])

  // Merge onboarding + backend extracted facts
  const hasBackendData = backendProfile && (
    backendProfile.child_name ||
    backendProfile.child_age ||
    backendProfile.challenge_areas.length > 0 ||
    backendProfile.attempted_strategies.length > 0 ||
    backendProfile.good_day_description ||
    backendProfile.hardest_situations.length > 0
  )

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

      {/* Backend-extracted facts — "What Ally Has Learned" */}
      {hasBackendData && (
        <>
          <Separator />
          <div className="space-y-3">
            <div className="flex items-center gap-2">
              <Sparkles className="h-4 w-4 text-coach" />
              <h2 className="font-semibold tracking-tight">What Ally Has Learned</h2>
            </div>
            <p className="text-sm text-muted-foreground">
              Facts extracted from your conversations with Ally.
            </p>
            <div className="grid gap-6 md:grid-cols-2">
              {/* Child details from backend */}
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2 text-base">
                    <Heart className="h-4 w-4 text-primary" />
                    {backendProfile?.child_name
                      ? `About ${backendProfile.child_name}`
                      : "Child Information"}
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  {backendProfile?.child_name && (
                    <div className="flex justify-between text-sm">
                      <span className="text-muted-foreground">Name</span>
                      <span className="font-medium">{backendProfile.child_name}</span>
                    </div>
                  )}
                  {backendProfile?.child_age && (
                    <div className="flex justify-between text-sm">
                      <span className="text-muted-foreground">Age</span>
                      <span className="font-medium">{backendProfile.child_age}</span>
                    </div>
                  )}
                  {backendProfile?.diagnosis_status && (
                    <div className="flex justify-between text-sm">
                      <span className="text-muted-foreground">Diagnosis</span>
                      <span className="font-medium">{backendProfile.diagnosis_status}</span>
                    </div>
                  )}
                  {backendProfile?.good_day_description && (
                    <div className="text-sm">
                      <span className="text-muted-foreground">A good day looks like</span>
                      <p className="mt-1 font-medium leading-relaxed">
                        {backendProfile.good_day_description}
                      </p>
                    </div>
                  )}
                </CardContent>
              </Card>

              {/* Challenges & strategies from backend */}
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2 text-base">
                    <Brain className="h-4 w-4 text-accent-foreground" />
                    Insights
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  {backendProfile && backendProfile.challenge_areas.length > 0 && (
                    <div>
                      <p className="mb-2 text-xs font-medium text-muted-foreground uppercase tracking-wider">
                        Challenge Areas
                      </p>
                      <div className="flex flex-wrap gap-1.5">
                        {backendProfile.challenge_areas.map((c) => (
                          <Badge key={c} variant="secondary" className="rounded-full bg-primary/10 text-primary">
                            {c.replace(/_/g, " ")}
                          </Badge>
                        ))}
                      </div>
                    </div>
                  )}
                  {backendProfile && backendProfile.attempted_strategies.length > 0 && (
                    <div>
                      <p className="mb-2 text-xs font-medium text-muted-foreground uppercase tracking-wider">
                        Strategies Discussed
                      </p>
                      <div className="flex flex-wrap gap-1.5">
                        {backendProfile.attempted_strategies.map((s) => (
                          <Badge key={s} variant="secondary" className="rounded-full bg-coach/10 text-coach">
                            {s.replace(/_/g, " ")}
                          </Badge>
                        ))}
                      </div>
                    </div>
                  )}
                  {backendProfile && backendProfile.hardest_situations.length > 0 && (
                    <div>
                      <p className="mb-2 text-xs font-medium text-muted-foreground uppercase tracking-wider">
                        Hardest Situations
                      </p>
                      <div className="flex flex-wrap gap-1.5">
                        {backendProfile.hardest_situations.map((h) => (
                          <Badge key={h} variant="outline" className="rounded-full">
                            {h.replace(/_/g, " ")}
                          </Badge>
                        ))}
                      </div>
                    </div>
                  )}
                </CardContent>
              </Card>
            </div>
          </div>
        </>
      )}

      {onboarding && (
        <>
          <Separator />
          <div className="grid gap-6 md:grid-cols-2">
            {/* Child Info from onboarding */}
            <Card>
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

            {/* Challenges & Goals from onboarding */}
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

            {/* Coaching History */}
            <Card className="md:col-span-2">
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-base">
                  <Calendar className="h-4 w-4 text-muted-foreground" />
                  Coaching History
                </CardTitle>
              </CardHeader>
              <CardContent>
                {sessions.length > 0 ? (
                  <div className="space-y-2">
                    {sessions.slice(0, 10).map((s) => (
                      <div
                        key={s.session_id}
                        className="flex items-center justify-between rounded-lg border border-border/50 px-3 py-2"
                      >
                        <div className="flex items-center gap-3">
                          <Badge variant="secondary" className="text-xs">
                            {s.phase}
                          </Badge>
                          <span className="text-sm text-muted-foreground">
                            {s.turn_count} messages
                          </span>
                        </div>
                        {s.created_at && (
                          <span className="text-xs text-muted-foreground">
                            {new Date(s.created_at).toLocaleDateString(undefined, {
                              month: "short",
                              day: "numeric",
                            })}
                          </span>
                        )}
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="flex h-24 items-center justify-center rounded-lg border border-dashed border-border text-sm text-muted-foreground">
                    Coaching session history will appear here as you use the app.
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        </>
      )}
    </div>
  )
}
