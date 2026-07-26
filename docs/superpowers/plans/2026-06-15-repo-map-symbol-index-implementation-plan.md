# Repo Map / Symbol Index Implementation Plan

Date: 2026-06-15

## Context

Tasks 1-4 restored reliable workspace read exposure, resumable large file reads, diagnostics surfacing, and discovery-first tool exposure. The next usability gap is repo navigation in medium and large workspaces: the agent still relies too heavily on `glob_search`, `grep_search`, and repeated `file_read` calls to figure out where to look.

Mature products generally add a lightweight indexing or code-intelligence layer here:
- Aider uses a repository map to expose important files, classes, and functions.
- Claude Code documents project-specific indexing for large codebases.
- `cc-haha` already has mature workspace tree / preview / diff UX plus LSP capability plumbing, but does not expose a simple agent-facing `repo_map` / `read_symbol` style API.

Goal: add a minimal but complete read-only repo navigation layer that improves file targeting without replacing normal reads or overcommitting to a full LSP platform.

## Product Goal

Add two bounded workspace tools:
- `repo_map()`
- `read_symbol(path="...", symbol="...")`

These tools should:
- stay inside existing workspace boundaries
- be read-only
- help the model choose the right files and symbol regions faster
- work without requiring external services or heavyweight setup
- remain compatible with current `tool_search` / workspace tool exposure behavior

## Non-Goals

- Do not build a full LSP orchestration layer in this wave.
- Do not require tree-sitter, ctags, or external binaries as hard dependencies.
- Do not replace `file_read`, `glob_search`, or `grep_search`.
- Do not attempt cross-repository symbol intelligence.
- Do not overfit to one language only.

## Architecture

Implement a small in-process repo indexing helper with two tool surfaces:

1. `repo_map`
- walks the effective workspace root
- returns a bounded list of files plus extracted top-level symbols
- supports light filtering / path prefix narrowing
- is optimized for navigation, not full-text output

2. `read_symbol`
- resolves a file inside the effective workspace root
- extracts top-level symbols from that file
- returns the requested symbol block or bounded fallback region
- preserves existing line-based readability so the agent can continue with `file_read` if needed

Indexing approach:
- Python: use `ast` for robust top-level function / class extraction with line ranges
- JS / TS / JSX / TSX: use bounded regex-based top-level declaration extraction
- Other common text files: no deep parsing; return file entry with no symbols

This keeps the first version useful and deterministic without introducing external parser dependencies.

## Subagent Split

Use fresh worker subagents with disjoint ownership.

### Worker A: Core Repo Index / Tools
Owns:
- `mochi/tools/repo_map.py` (new)
- any small helper module created for symbol extraction, if needed
- `tests/test_repo_map.py` (new)

Responsibilities:
- implement `RepoMapTool`
- implement `ReadSymbolTool`
- implement bounded symbol extraction helpers
- add direct tool tests

### Worker B: Registry / Exposure / Prompt Integration
Owns:
- `mochi/tools/registry_factory.py`
- `mochi/agents/tool_exposure.py`
- `mochi/agents/prompt_builder.py`
- `tests/test_tool_exposure.py`
- `tests/test_prompt_builder.py`

Responsibilities:
- register the new tools
- place them in the workspace tool group
- keep current workspace baseline behavior intact
- improve ranking / guidance so the agent can use repo navigation when helpful without replacing normal file reads

### Controller
Owns:
- final integration
- review loop
- targeted verification
- any small conflict-resolution edits if Worker A/B interfaces meet awkwardly

## Task Breakdown

### Task 1: Add Core Repo Navigation Tools

Files:
- Create: `mochi/tools/repo_map.py`
- Create: `tests/test_repo_map.py`

Requirements:
- `repo_map()` returns a bounded navigation-oriented summary for the effective workspace
- include:
  - relative file path
  - detected language / kind when available
  - extracted top-level symbols with line ranges when available
- skip noisy directories such as:
  - `.git`
  - `.mochi`
  - `node_modules`
  - `__pycache__`
  - `.pytest_cache`
- keep output bounded by:
  - max files
  - max symbols per file
  - max characters in formatted tool output

- `read_symbol(path, symbol)`:
  - validates workspace scope using the same boundary model as file tools
  - finds the requested symbol in the file
  - returns the bounded symbol block with line numbers
  - for Python use `ast` line ranges when available
  - for regex-parsed languages use the start line plus next-symbol boundary fallback
  - returns a useful error when the symbol is not found

Tests:
- Python file with classes/functions appears in `repo_map`
- JS/TS-style declarations appear in `repo_map`
- `read_symbol` returns correct Python block
- `read_symbol` returns correct regex-based block
- blocked / missing paths fail safely

### Task 2: Register And Surface The New Tools

Files:
- Modify: `mochi/tools/registry_factory.py`
- Modify: `mochi/agents/tool_exposure.py`
- Modify: `mochi/agents/prompt_builder.py`
- Modify: `tests/test_tool_exposure.py`
- Modify: `tests/test_prompt_builder.py`

Requirements:
- register `repo_map` and `read_symbol` as workspace read-only tools
- place them in the workspace tool group
- do not remove current workspace baseline:
  - `file_read`
  - `glob_search`
  - `grep_search`
  - specialized readers
- ranking should make `repo_map` and `read_symbol` discoverable for:
  - large repo navigation
  - symbol lookup
  - definition / class / function inspection
- prompt guidance should mention:
  - use `repo_map` to orient in larger repos when needed
  - use `read_symbol` for targeted symbol inspection
  - continue using normal read tools for concrete file content

Tests:
- workspace tool grouping includes the new tools
- repo-navigation intent can expose or prioritize them without suppressing current baseline
- prompt builder text includes the new guidance

### Task 3: Verification

Run at minimum:

```powershell
pytest tests/test_repo_map.py tests/test_tool_exposure.py tests/test_prompt_builder.py -v
```

Recommended focused integration check:

```powershell
pytest tests/test_engine_phase2.py -k "tool_exposure or planner" -v
```

## Expected End State

- Mochi gains a practical repo navigation layer for medium/large workspaces.
- The agent can ask for a repo map before blindly grepping or reading many files.
- The agent can target a symbol definition directly without requiring a full file read first.
- The implementation stays read-only, local, and dependency-light.
- This wave leaves room for a future LSP-backed or tree-sitter-backed upgrade without forcing it now.
