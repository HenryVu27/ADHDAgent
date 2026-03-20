import asyncio


class RateLimiter:
    """Async rate limiter using semaphore with timed release."""

    def __init__(self, rps: float):
        self._semaphore = asyncio.Semaphore(max(1, int(rps)))
        self._interval = 1.0 / rps

    async def acquire(self):
        await self._semaphore.acquire()
        loop = asyncio.get_running_loop()
        loop.call_later(self._interval, self._semaphore.release)
