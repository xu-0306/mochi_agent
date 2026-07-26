export interface ParsedSseFrame {
  eventName: string | null
  data: string | null
}

export function parseSseFrame(frame: string): ParsedSseFrame {
  const lines = frame.split(/\r?\n/)
  const eventName =
    lines
      .find((line) => line.startsWith('event:'))
      ?.slice('event:'.length)
      .trim() ?? null
  const data =
    lines
      .filter((line) => line.startsWith('data:'))
      .map((line) => line.slice(5).trim())
      .join('\n')
      .trim() || null

  return { eventName, data }
}
