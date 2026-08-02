export type RuntimeObserverOptions<S, E> = {
  signal: AbortSignal
  fetchSnapshot: (signal: AbortSignal) => Promise<{ state: S; lastEventId: string | null }>
  stream: (cursor: { lastEventId: string | null }, signal: AbortSignal) => AsyncIterable<E>
  reduce: (state: S, event: E) => S
  onState: (state: S) => void
  isStaleCursor: (error: unknown) => boolean
  now?: () => number
  sleep?: (milliseconds: number, signal: AbortSignal) => Promise<void>
  deadlineMs?: number
}

function abortableSleep(milliseconds: number, signal: AbortSignal): Promise<void> {
  if (signal.aborted) return Promise.resolve()

  return new Promise<void>((resolve) => {
    const finish = () => {
      globalThis.clearTimeout(timer)
      signal.removeEventListener('abort', finish)
      resolve()
    }
    const timer = globalThis.setTimeout(finish, milliseconds)
    signal.addEventListener('abort', finish, { once: true })
  })
}

export async function observeOrdinaryChatRuntime<S, E>(options: RuntimeObserverOptions<S, E>): Promise<void> {
  if (options.signal.aborted) return

  const now = options.now ?? Date.now
  const deadline = now() + (options.deadlineMs ?? 120_000)
  const sleep = options.sleep ?? abortableSleep
  let snapshot = await options.fetchSnapshot(options.signal)
  if (options.signal.aborted) return

  let state = snapshot.state
  let lastEventId = snapshot.lastEventId
  for (let attempt = 0; !options.signal.aborted && now() < deadline; attempt += 1) {
    options.onState(state)
    try {
      for await (const event of options.stream({ lastEventId }, options.signal)) {
        if (options.signal.aborted) return
        state = options.reduce(state, event)
        lastEventId = (event as { event_id?: string }).event_id ?? lastEventId
        options.onState(state)
      }
    } catch (error) {
      if (options.signal.aborted) return
      if (!options.isStaleCursor(error)) throw error
      snapshot = await options.fetchSnapshot(options.signal)
      if (options.signal.aborted) return
      state = snapshot.state
      lastEventId = snapshot.lastEventId
    }
    if (!options.signal.aborted && now() < deadline) {
      await sleep(Math.min(3000, 400 * (attempt + 1)), options.signal)
    }
  }
}
