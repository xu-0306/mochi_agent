import assert from 'node:assert/strict'
import fs from 'node:fs/promises'
import path from 'node:path'

const source = await fs.readFile(
  path.join(process.cwd(), 'src/components/chat/ChatInput.tsx'),
  'utf8'
)

assert.match(
  source,
  /function extractClipboardFiles\(data: DataTransfer \| null\): File\[\]/,
  'ChatInput should extract file payloads from clipboard paste events'
)

assert.match(
  source,
  /function normalizeUploadFile\(file: File, fallbackIndex: number\): File/,
  'ChatInput should normalize pasted screenshots that arrive without a filename'
)

assert.match(
  source,
  /const handlePaste = React\.useCallback\(\(event: React\.ClipboardEvent<HTMLTextAreaElement>\) => \{/,
  'ChatInput should wire a paste handler for clipboard attachments'
)

assert.match(
  source,
  /const clipboardFiles = extractClipboardFiles\(event\.clipboardData\)/,
  'ChatInput paste handler should read files from clipboardData'
)

assert.match(
  source,
  /event\.preventDefault\(\)\s*void handleAttachFiles\(clipboardFiles\)/,
  'ChatInput should route pasted clipboard files through the existing attachment upload flow'
)

assert.match(
  source,
  /onPaste=\{handlePaste\}/,
  'ChatInput textarea should subscribe to the paste handler'
)

console.log('ok')
