import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from "@/components/ui/dialog"
import { Badge } from "@/components/ui/badge"
import { Separator } from "@/components/ui/separator"
import type { KnowledgeDocument } from "@/types"

interface Props {
  document: KnowledgeDocument | null
  open: boolean
  onClose: () => void
}

export function ResourceDetail({ document: doc, open, onClose }: Props) {
  if (!doc) return null

  return (
    <Dialog open={open} onOpenChange={(v) => !v && onClose()}>
      <DialogContent className="max-h-[80vh] max-w-lg overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="tracking-tight">{doc.name}</DialogTitle>
          <DialogDescription className="leading-relaxed">
            {doc.description}
          </DialogDescription>
        </DialogHeader>

        <div className="flex flex-wrap gap-1.5">
          {doc.tags.map((tag) => (
            <Badge key={tag} variant="secondary" className="text-xs">
              {tag.replace(/_/g, " ")}
            </Badge>
          ))}
        </div>

        <Separator />

        {doc.steps && doc.steps.length > 0 && (
          <div>
            <h4 className="mb-3 font-semibold text-sm">Steps</h4>
            <ol className="space-y-2">
              {doc.steps.map((step, i) => (
                <li key={i} className="flex gap-3 text-sm leading-relaxed">
                  <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-coach/15 text-xs font-medium text-coach">
                    {i + 1}
                  </span>
                  <span className="text-muted-foreground">{step}</span>
                </li>
              ))}
            </ol>
          </div>
        )}

        {doc.key_points && doc.key_points.length > 0 && (
          <div>
            <h4 className="mb-3 font-semibold text-sm">Key Points</h4>
            <ul className="space-y-2">
              {doc.key_points.map((point, i) => (
                <li key={i} className="flex gap-2 text-sm leading-relaxed">
                  <span className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-coach" />
                  <span className="text-muted-foreground">{point}</span>
                </li>
              ))}
            </ul>
          </div>
        )}

        {doc.evidence_level && (
          <>
            <Separator />
            <div className="flex items-center gap-2 text-xs text-muted-foreground">
              <span>Evidence level:</span>
              <Badge variant="outline" className="text-xs">
                {doc.evidence_level}
              </Badge>
            </div>
          </>
        )}

        {doc.source && (
          <p className="text-xs text-muted-foreground">Source: {doc.source}</p>
        )}
      </DialogContent>
    </Dialog>
  )
}
