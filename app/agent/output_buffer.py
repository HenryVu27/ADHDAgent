"""Token buffer for output gate pre-screening.

Accumulates streamed tokens until a configurable character threshold,
then signals readiness for output gate classification before flushing
tokens to the client.
"""


class OutputBuffer:
    """Accumulates streamed tokens for output gate pre-screening."""

    def __init__(self, buffer_size: int = 200):
        self._buffer: list[str] = []
        self._char_count: int = 0
        self._buffer_size: int = buffer_size
        self._flushed: bool = False

    @property
    def buffered_text(self) -> str:
        return "".join(self._buffer)

    @property
    def is_full(self) -> bool:
        return self._char_count >= self._buffer_size

    @property
    def flushed(self) -> bool:
        return self._flushed

    def add(self, token: str) -> None:
        """Add a token to the buffer."""
        self._buffer.append(token)
        self._char_count += len(token)

    def flush(self) -> str:
        """Return all buffered text and mark as flushed."""
        text = self.buffered_text
        self._flushed = True
        return text

    def should_passthrough(self, token: str) -> bool:
        """After flush, tokens pass through directly."""
        return self._flushed
