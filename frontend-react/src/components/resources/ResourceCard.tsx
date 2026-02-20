import { Badge } from "@/components/ui/badge"
import { Card, CardContent } from "@/components/ui/card"
import { BookOpen, Lightbulb, Heart } from "lucide-react"
import type { KnowledgeDocument } from "@/types"

function getDocIcon(doc: KnowledgeDocument) {
  if (doc.steps) return Lightbulb
  if (doc.tags.includes("adhd_facts") || doc.tags.includes("executive_function")) return BookOpen
  return Heart
}

function getDocCategory(doc: KnowledgeDocument): string {
  if (doc.steps) return "Strategy"
  if (doc.tags.some((t) => ["adhd_facts", "executive_function", "neuroscience"].includes(t)))
    return "Fact"
  return "Guidance"
}

function getCategoryStyle(category: string) {
  switch (category) {
    case "Strategy":
      return { icon: "bg-accent/30 text-accent-foreground", badge: "bg-accent/20 text-accent-foreground" }
    case "Fact":
      return { icon: "bg-coach/15 text-coach", badge: "bg-coach/15 text-coach" }
    default:
      return { icon: "bg-primary/10 text-primary", badge: "bg-primary/10 text-primary" }
  }
}

interface Props {
  document: KnowledgeDocument
  onClick: () => void
}

export function ResourceCard({ document: doc, onClick }: Props) {
  const Icon = getDocIcon(doc)
  const category = getDocCategory(doc)
  const style = getCategoryStyle(category)

  return (
    <Card
      className="cursor-pointer transition-all duration-200 hover:shadow-md hover:-translate-y-0.5"
      onClick={onClick}
    >
      <CardContent className="p-5">
        <div className="mb-3 flex items-start justify-between">
          <div className={`flex h-10 w-10 items-center justify-center rounded-lg ${style.icon}`}>
            <Icon className="h-5 w-5" />
          </div>
          <Badge variant="secondary" className={`text-xs ${style.badge}`}>
            {category}
          </Badge>
        </div>
        <h3 className="mb-1 font-semibold tracking-tight">{doc.name}</h3>
        <p className="mb-3 text-sm text-muted-foreground leading-relaxed line-clamp-2">
          {doc.description}
        </p>
        <div className="flex flex-wrap gap-1">
          {doc.tags.slice(0, 4).map((tag) => (
            <Badge key={tag} variant="outline" className="text-[11px] px-1.5 py-0 rounded-full">
              {tag.replace(/_/g, " ")}
            </Badge>
          ))}
          {doc.tags.length > 4 && (
            <Badge variant="outline" className="text-[11px] px-1.5 py-0 rounded-full">
              +{doc.tags.length - 4}
            </Badge>
          )}
        </div>
        {doc.source && (
          <p className="mt-2 text-[11px] text-muted-foreground">Source: {doc.source}</p>
        )}
      </CardContent>
    </Card>
  )
}
