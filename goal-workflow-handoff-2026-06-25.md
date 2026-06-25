# Goal / Workflow Handoff

Date: 2026-06-25

## 1. Product direction locked in

This round implements the following UX decision:

- `Goal` is the user-facing long-running task concept.
- `Workflow` is no longer a parallel top-level concept. It is an execution mode of a goal: `execution_mode = "workflow"`.
- `/goal <request>` prepares a chat-first single-agent long task.
- `/workflow <request>` prepares a workflow-style long task.
- A goal does not have to be a workflow.
- Single-agent goals should stay in the main chat lane.
- Workflow UI stays available as advanced controls, not as the primary entry path.
- No extra dedicated goal page should be introduced unless product direction changes later.

## 2. What is already implemented

### 2.1 Slash-command routing in chat

Primary file: `web/src/app/page.tsx`

- `/goal` parser: `web/src/app/page.tsx:493`
- `/goal status|pause|resume|stop`: `web/src/app/page.tsx:509`
- Goal confirmation keywords (`start`, `go ahead`, `proceed`, `yes`, `run it`): `web/src/app/page.tsx:524`
- Proposal creation / revision branch: `web/src/app/page.tsx:2752`
- Goal create + start flow: `web/src/app/page.tsx:2807`
- Goal lifecycle commands: `web/src/app/page.tsx:2918`

### 2.2 Session goal state persistence

Primary files:

- Session workflow PATCH API: `web/src/lib/api.ts:2243`
- Session goal PATCH API: `web/src/lib/api.ts:2257`
- Local `persistSessionGoalState`: `web/src/app/page.tsx:1882`
- Local `persistWorkflowState`: `web/src/app/page.tsx:1922`

Session detail now supports:

- `workflow`
- `goal`
- `security_override`

Related contract area: `web/src/lib/api.ts:1929`

### 2.3 Goal proposal and summary cards

Primary files:

- Goal card builder: `web/src/app/page.tsx:599`
- Goal proposal builder: `web/src/app/page.tsx:2465`
- Goal card tone changes: `web/src/components/chat/GoalCard.tsx:39`
- Superseded badge rendering: `web/src/components/chat/GoalCard.tsx:200`

Implemented card states:

- proposal
- revised proposal
- started
- lifecycle status summary
- superseded older card instances for the same goal

### 2.4 Workflow UI suppression for single-agent goals

Primary file: `web/src/components/chat/WorkflowPanel.tsx`

- Goal execution mode lookup: `web/src/components/chat/WorkflowPanel.tsx:423`
- Suppression switch: `web/src/components/chat/WorkflowPanel.tsx:432`
- Suppressed UI block: `web/src/components/chat/WorkflowPanel.tsx:626`

Current behavior:

- If the current session goal is `single_agent`, workflow-native controls are intentionally moved out of the main path.
- The panel explains that `/workflow <request>` is the explicit path for workflow preparation.
- Existing workflow settings remain stored in the session, but they are not treated as active in this state.

### 2.5 Task/projection gating

Primary files:

- Task panel workflow-native context gating: `web/src/components/chat/TaskPanel.tsx:1235`
- Task panel workflow section visibility: `web/src/components/chat/TaskPanel.tsx:718`
- Projection gating via `allowWorkflowNativeContent`: `web/src/lib/chat-projections.ts:114`
- Goal card superseded marking: `web/src/lib/chat-projections.ts:251`

Current behavior:

- Single-agent goals do not inject workflow-native progress/completion content into the chat stream.
- Workflow-native content is still allowed when there is no current goal and the session is still bound to a workflow run.
- Delegated subagent task cards still remain visible.

## 3. Verification status

Confirmed in the current working tree:

- `npm.cmd --prefix web run type-check` passed
- `npm.cmd exec eslint -- src/app/page.tsx src/components/chat/GoalCard.tsx src/components/chat/TaskPanel.tsx src/components/chat/WorkflowPanel.tsx src/lib/api.ts src/lib/chat-projections.ts` passed from `web/`

Not rerun in this pass:

- backend `pytest`
- manual end-to-end UI flows
- automated tests for slash routing and session round-trip

## 4. Files changed

Modified:

- `web/src/app/page.tsx`
- `web/src/components/chat/GoalCard.tsx`
- `web/src/components/chat/TaskPanel.tsx`
- `web/src/components/chat/WorkflowPanel.tsx`
- `web/src/lib/api.ts`
- `web/src/lib/chat-projections.ts`

Untracked and should not be deleted:

- `Untitled-2026-06-19-1605.excalidraw`

## 5. Main risks and unfinished work

### 5.1 `handleSend` is now the highest-risk area

Key entry: `web/src/app/page.tsx:2563`

This block now contains:

- goal proposal logic
- workflow proposal logic
- proposal revision logic
- confirmation logic
- active goal lifecycle command handling
- normal chat send handling

It works, but it is the most likely place for regressions. The next agent should consider splitting command routing into helpers after tests exist.

### 5.2 Targeted test coverage is still missing

Best next tests:

- `/goal` proposal routing
- `/workflow` proposal routing
- confirmation updates the correct `goal` and `workflow` session state
- single-agent goal suppresses workflow-native projections
- superseded goal cards only leave the latest card active

### 5.3 Backend session goal contract still needs one more pass

Frontend now sends `{ goal }` through `PATCH /sessions/:id`.

Relevant entry:

- `web/src/lib/api.ts:2257`

The next agent should confirm:

- backend schema fully accepts the current `goal` payload shape
- draft sessions and persisted sessions behave the same way
- session reload restores `goal.default_route` as expected

### 5.4 Subagent delegation attempts did not produce usable output

Prior attempts to split the work across subagents failed due to provider capacity / model allowlist constraints. This handoff is based on the actual working tree only.

## 6. Suggested reading order for the next agent

1. `web/src/app/page.tsx:2465-2996`
2. `web/src/components/chat/WorkflowPanel.tsx:423-652`
3. `web/src/components/chat/TaskPanel.tsx:1225-1313`
4. `web/src/lib/chat-projections.ts:107-277`
5. `web/src/lib/api.ts:1929-2269`

## 7. Suggested next implementation order

1. Add focused tests first.
2. Refactor `handleSend` goal/workflow command routing into smaller helpers.
3. Validate backend session goal round-trip.
4. Only after that, consider richer agent-driven orchestration for model selection, role suggestion, and run duration planning.

## 8. Guardrails for the next agent

- Do not reintroduce a heavy separate goal page unless product direction changes.
- Do not turn goal and workflow back into two parallel top-level UX concepts.
- Do not delete `Untitled-2026-06-19-1605.excalidraw`.
- If `handleSend` is going to be changed heavily, add tests first.
