import { useRef, useState, useCallback, useEffect } from "react"

const STEP_NORMAL = 12          // chars per frame at 60fps (~720 chars/sec)
const BATCH_THRESHOLD = 30      // re-render every N chars advanced
const CATCHUP_THRESHOLD = 100   // if this far behind, catch up faster

interface TypewriterControls {
  displayedText: string
  reset: (newText: string) => void
}

/**
 * Adaptive typewriter for streaming LLM responses.
 *
 * Decouples token receipt from display via requestAnimationFrame:
 *   - fullRef accumulates received tokens without re-renders
 *   - rAF loop advances shownRef at STEP_NORMAL chars/frame (~720 chars/sec)
 *   - When buffer is CATCHUP_THRESHOLD+ chars ahead, step = ceil(behind/10)
 *   - React state only updates every BATCH_THRESHOLD chars (~20fps)
 *
 * onComplete: called once when the animation catches up to the full text
 * (shownRef >= fullRef.length and fullRef is non-empty). Used by the done
 * handler to delay finalization until the typewriter finishes.
 */
export function useTypewriter(
  fullText: string,
  onComplete?: () => void,
): TypewriterControls {
  const fullRef = useRef("")
  const shownRef = useRef(0)
  const lastRenderedRef = useRef(0)
  const rafRef = useRef<number | null>(null)
  const [displayedText, setDisplayedText] = useState("")
  const onCompleteRef = useRef(onComplete)
  const firedCompleteRef = useRef(false)

  // Keep callback ref fresh without re-creating tick
  useEffect(() => {
    onCompleteRef.current = onComplete
  }, [onComplete])

  // Keep fullRef in sync with incoming fullText
  useEffect(() => {
    fullRef.current = fullText
    // New text arrived — animation is no longer complete
    if (fullText.length > shownRef.current) {
      firedCompleteRef.current = false
    }
    scheduleFrame()
  }, [fullText]) // eslint-disable-line react-hooks/exhaustive-deps

  const scheduleFrame = useCallback(() => {
    if (rafRef.current !== null) return  // already scheduled
    if (shownRef.current >= fullRef.current.length) return
    rafRef.current = requestAnimationFrame(tick)
  }, [])

  const tick = useCallback(() => {
    rafRef.current = null
    const full = fullRef.current
    const shown = shownRef.current

    if (shown >= full.length) {
      if (full.length > 0 && !firedCompleteRef.current) {
        firedCompleteRef.current = true
        onCompleteRef.current?.()
      }
      return
    }

    const behind = full.length - shown
    const step = behind > CATCHUP_THRESHOLD ? Math.ceil(behind / 10) : STEP_NORMAL
    shownRef.current = Math.min(shown + step, full.length)

    const advanced = shownRef.current - lastRenderedRef.current
    if (advanced >= BATCH_THRESHOLD || shownRef.current === full.length) {
      setDisplayedText(full.slice(0, shownRef.current))
      lastRenderedRef.current = shownRef.current
    }

    if (shownRef.current < full.length) {
      rafRef.current = requestAnimationFrame(tick)
    } else if (!firedCompleteRef.current) {
      firedCompleteRef.current = true
      onCompleteRef.current?.()
    }
  }, [])

  const reset = useCallback((newText: string) => {
    if (rafRef.current !== null) {
      cancelAnimationFrame(rafRef.current)
      rafRef.current = null
    }
    fullRef.current = newText
    shownRef.current = 0
    lastRenderedRef.current = 0
    firedCompleteRef.current = false
    setDisplayedText("")
    scheduleFrame()
  }, [scheduleFrame])

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current)
    }
  }, [])

  return { displayedText, reset }
}
