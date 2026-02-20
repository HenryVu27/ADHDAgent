import { Link } from "react-router-dom"
import { MessageCircle, BookOpen, ArrowRight } from "lucide-react"
import { Card, CardContent } from "@/components/ui/card"
import { Button } from "@/components/ui/button"

export function ActionCards() {
  return (
    <div className="grid gap-4 sm:grid-cols-2">
      <Link to="/chat" className="group">
        <Card className="h-full transition-all duration-200 hover:shadow-md hover:-translate-y-0.5">
          <CardContent className="flex flex-col gap-4 p-6">
            <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-coach/15">
              <MessageCircle className="h-6 w-6 text-coach" />
            </div>
            <div>
              <h3 className="mb-1 text-lg font-semibold tracking-tight">Chat with Ally</h3>
              <p className="text-sm text-muted-foreground leading-relaxed">
                Continue your coaching session and get personalized strategies for your family.
              </p>
            </div>
            <Button variant="ghost" className="w-fit gap-2 p-0 text-primary">
              Start a conversation
              <ArrowRight className="h-4 w-4 transition-transform group-hover:translate-x-1" />
            </Button>
          </CardContent>
        </Card>
      </Link>

      <Link to="/resources" className="group">
        <Card className="h-full transition-all duration-200 hover:shadow-md hover:-translate-y-0.5">
          <CardContent className="flex flex-col gap-4 p-6">
            <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-accent/30">
              <BookOpen className="h-6 w-6 text-accent-foreground" />
            </div>
            <div>
              <h3 className="mb-1 text-lg font-semibold tracking-tight">Resource Library</h3>
              <p className="text-sm text-muted-foreground leading-relaxed">
                Browse evidence-based strategies, ADHD facts, and parenting guidance.
              </p>
            </div>
            <Button variant="ghost" className="w-fit gap-2 p-0 text-primary">
              Explore resources
              <ArrowRight className="h-4 w-4 transition-transform group-hover:translate-x-1" />
            </Button>
          </CardContent>
        </Card>
      </Link>
    </div>
  )
}
