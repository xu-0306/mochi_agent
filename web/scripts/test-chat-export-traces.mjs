import assert from 'node:assert/strict'
import path from 'node:path'
import { pathToFileURL } from 'node:url'

const moduleUrl = pathToFileURL(
  path.join(process.cwd(), 'src/lib/chat-p2.ts')
).href

const { buildChatExport } = await import(moduleUrl)

function makeMessage(id, type, content, overrides = {}) {
  return {
    id,
    type,
    content,
    timestamp: new Date('2026-07-08T08:00:00.000Z'),
    ...overrides,
  }
}

const traceEvents = [
  {
    type: 'subagent_thinking',
    status: 'running',
    content: 'Trying arXiv before falling back to web search.',
    summary: 'Search strategy update',
    metadata: { source: 'runtime_progress' },
    createdAt: '2026-07-08T08:00:01.500Z',
    roleId: 'researcher',
  },
  {
    type: 'subagent_tool_result',
    status: 'success',
    summary: 'arxiv_search returned 5 results.',
    metadata: {
      tool_name: 'arxiv_search',
      call_id: 'call-arxiv-1',
      guarded_length: 702,
      result: { title: 'DCNN-Match', published: '2021-09-06T20:16:27Z' },
    },
    createdAt: '2026-07-08T08:00:04.000Z',
    roleId: 'researcher',
  },
]

const messages = [
  makeMessage('user-1', 'user', 'Find related papers.'),
  makeMessage('assistant-1', 'assistant', 'I found a few useful papers.', {
    reasoningSteps: [
      {
        id: 'reasoning-1',
        type: 'thinking',
        content: 'Try another retrieval path for paper metadata.',
        timestamp: new Date('2026-07-08T08:00:01.000Z'),
        source: 'model_summary',
      },
      {
        id: 'tool-1',
        type: 'tool_result',
        content: '',
        timestamp: new Date('2026-07-08T08:00:02.000Z'),
        toolName: 'semantic_scholar_search',
        toolCallId: 'call-semantic-1',
        toolError: 'HTTP 429 after 3 attempts',
        toolMeta: {
          status_code: 429,
          last_tool_name: 'semantic_scholar_search',
        },
        status: 'error',
      },
      {
        id: 'tool-2',
        type: 'tool_result',
        content: '',
        timestamp: new Date('2026-07-08T08:00:03.000Z'),
        toolName: 'web_search',
        toolCallId: 'call-web-1',
        toolResult: {
          query: 'small multimodal LLM medical image fine-tuning methods 2024',
          results: [{ title: 'PEFoMed', url: 'https://arxiv.org/abs/2401.02797' }],
        },
        toolMeta: { provider: 'tavily', status_code: 200 },
        status: 'success',
      },
    ],
  }),
]

const markdownDefault = buildChatExport(messages, 'markdown')
assert.match(markdownDefault, /## User\nFind related papers\./)
assert.match(markdownDefault, /## Assistant\nI found a few useful papers\./)
assert.doesNotMatch(markdownDefault, /Reasoning and Tool Trace/)
assert.doesNotMatch(markdownDefault, /semantic_scholar_search/)

const markdownWithTrace = buildChatExport(messages, 'markdown', { includeReasoning: true, traceEvents })
assert.match(markdownWithTrace, /### Reasoning and Tool Trace/)
assert.match(markdownWithTrace, /#### Reasoning summary/)
assert.match(markdownWithTrace, /semantic_scholar_search failed/)
assert.match(markdownWithTrace, /Call ID: `call-semantic-1`/)
assert.match(markdownWithTrace, /HTTP 429 after 3 attempts/)
assert.match(markdownWithTrace, /web_search success/)
assert.match(markdownWithTrace, /PEFoMed/)
assert.match(markdownWithTrace, /## Execution Trace/)
assert.match(markdownWithTrace, /#### Thinking/)
assert.match(markdownWithTrace, /arxiv_search returned/)
assert.match(markdownWithTrace, /Call ID: `call-arxiv-1`/)
assert.match(markdownWithTrace, /DCNN-Match/)

const jsonDefault = JSON.parse(buildChatExport(messages, 'json'))
assert.equal(jsonDefault.length, 2)
assert.equal(jsonDefault[1].reasoning_steps, undefined)

const jsonWithTrace = JSON.parse(buildChatExport(messages, 'json', { includeReasoning: true, traceEvents }))
assert.equal(jsonWithTrace.messages[1].reasoning_steps.length, 3)
assert.equal(jsonWithTrace.messages[1].reasoning_steps[1].tool_name, 'semantic_scholar_search')
assert.equal(jsonWithTrace.messages[1].reasoning_steps[1].tool_error, 'HTTP 429 after 3 attempts')
assert.equal(jsonWithTrace.messages[1].reasoning_steps[2].tool_result.results[0].title, 'PEFoMed')
assert.equal('tool_args' in jsonWithTrace.messages[1].reasoning_steps[1], false)
assert.equal(jsonWithTrace.execution_trace.length, 2)
assert.equal(jsonWithTrace.execution_trace[1].tool_name, 'arxiv_search')
assert.equal(jsonWithTrace.execution_trace[1].call_id, 'call-arxiv-1')
assert.equal(jsonWithTrace.execution_trace[1].metadata.result.title, 'DCNN-Match')
