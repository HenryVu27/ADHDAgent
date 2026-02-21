import { useState, useMemo } from "react"
import { useQuery } from "@tanstack/react-query"
import { motion } from "framer-motion"
import { BookOpen, Loader2 } from "lucide-react"
import { api } from "@/lib/api"
import { ResourceFilters } from "@/components/resources/ResourceFilters"
import { ResourceCard } from "@/components/resources/ResourceCard"
import { ResourceDetail } from "@/components/resources/ResourceDetail"
import type { KnowledgeDocument } from "@/types"

function getDocCategory(doc: KnowledgeDocument): string {
  if (doc.steps) return "strategies"
  if (doc.tags.some((t) => ["adhd_facts", "executive_function", "neuroscience"].includes(t)))
    return "facts"
  return "guidance"
}

export function ResourcesPage() {
  const [category, setCategory] = useState("all")
  const [search, setSearch] = useState("")
  const [selectedDoc, setSelectedDoc] = useState<KnowledgeDocument | null>(null)

  const { data, isLoading } = useQuery({
    queryKey: ["knowledge-documents"],
    queryFn: () => api.getDocuments(),
  })

  const documents = data?.documents ?? []

  const filtered = useMemo(() => {
    let result = documents
    if (category !== "all") {
      result = result.filter((doc) => getDocCategory(doc) === category)
    }
    if (search) {
      const q = search.toLowerCase()
      result = result.filter(
        (doc) =>
          doc.name.toLowerCase().includes(q) ||
          doc.description.toLowerCase().includes(q) ||
          doc.tags.some((t) => t.toLowerCase().includes(q))
      )
    }
    return result
  }, [documents, category, search])

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Resource Library</h1>
        <p className="text-muted-foreground">
          Browse evidence-based strategies, ADHD facts, and parenting guidance.
        </p>
      </div>

      <ResourceFilters
        category={category}
        onCategoryChange={setCategory}
        search={search}
        onSearchChange={setSearch}
      />

      {isLoading ? (
        <div className="flex h-40 items-center justify-center">
          <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
        </div>
      ) : filtered.length === 0 ? (
        <div className="flex h-40 flex-col items-center justify-center text-center">
          <BookOpen className="mb-2 h-8 w-8 text-muted-foreground/50" />
          <p className="text-sm text-muted-foreground">
            {search ? "No resources match your search." : "No resources available."}
          </p>
        </div>
      ) : (
        <motion.div
          initial="hidden"
          animate="show"
          variants={{
            hidden: {},
            show: { transition: { staggerChildren: 0.05 } },
          }}
          className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3"
        >
          {filtered.map((doc) => (
            <motion.div
              key={doc.id}
              variants={{
                hidden: { opacity: 0, y: 12 },
                show: { opacity: 1, y: 0 },
              }}
            >
              <ResourceCard document={doc} onClick={() => setSelectedDoc(doc)} />
            </motion.div>
          ))}
        </motion.div>
      )}

      <ResourceDetail
        document={selectedDoc}
        open={!!selectedDoc}
        onClose={() => setSelectedDoc(null)}
      />
    </div>
  )
}
