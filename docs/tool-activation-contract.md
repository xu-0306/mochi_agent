# Mochi Tool Activation Contract

## Purpose

Mochi separates tool discovery from tool authority:

- A tool can be **discoverable** in the catalog without being callable in the current turn.
- A tool is **callable** only when the exposure plan and runtime policy make it available to the active registry view.
- `tool_search` is a discovery mechanism. It does not activate tools or mutate the active registry.
- A workspace write request creates a write obligation. A successful answer requires an observable mutation event, not only a model claim.

## Runtime contract

The normal flow is:

```text
route intent -> exposure plan -> callable registry view
    -> optional policy-gated activation request
    -> tool execution and security checks
    -> mutation verification before final success
```

For a searched tool, these fields describe availability:

```python
{
    "name": "file_write",
    "callable_this_turn": False,
    "activation_required": True,
    "activation_reason": (
        "Tool is discoverable but not exposed as callable in this turn."
    ),
    "activation_request": {
        "tool_name": "file_write",
        "required_intent": "workspace_write",
        "policy_check": "required",
    },
}
```

The activation request is descriptive. The runtime must still evaluate routed intent, execution profile, tool mode, allowlist, denylist, approval policy, workspace scope, and path protections before promotion. A search result must never be treated as authority.

## Write obligation

When the routed intent is `workspace_write`, the runtime carries a file-mutation obligation independently of which write tools are exposed.

The ReAct loop may finish successfully only after a file mutation result provides observable evidence such as:

- `error is None`;
- `file_changes` metadata;
- `bytes_written` or equivalent mutation metadata;
- a successful tool output.

If no write tool is callable, or execution is denied, the loop returns a structured blocker such as `file_artifact_not_mutated`. It must not accept a normal “saved” claim as proof of persistence.

## Diagnostic taxonomy

Use the metadata layer to locate the failure:

| Layer | Diagnostic fields | Meaning |
| --- | --- | --- |
| Routing | `intent_route.intent`, `confidence`, `source`, `rationale` | Which user-intent route was selected and why |
| Exposure | `exposed_tools`, `discoverable_tool_names`, exposure `diagnostics` | What was callable versus searchable this turn |
| Activation | `runtime_category=tool_activation`, `error_type=tool_activation_denied`, `requested_tool`, `reason` | Whether a promotion request was accepted or denied |
| Policy / approval | `approval_kind`, `requires_approval`, `recoverability` | Which policy or approval decision stopped execution |
| Mutation execution | `runtime_category=permission` or `deliverable_guard`, `error_type`, `file_changes`, `bytes_written` | Whether path security or the required side effect failed |

Common examples include:

- `mutation_tool_not_callable`: a required mutation tool was not in the callable view;
- `tool_activation_denied`: activation was rejected by routing, profile, policy, approval, or workspace checks;
- `file_path_denied`: the file path failed workspace/protected-path checks;
- `file_artifact_not_mutated`: the turn ended without verified mutation.

## Debugging checklist

1. Check the routed intent and its source before interpreting keywords.
2. Compare `exposed_tools` with `discoverable_tool_names`.
3. For a hidden tool, inspect `callable_this_turn`, `activation_required`, and `activation_reason`.
4. If activation was requested, inspect `requested_tool`, `reason`, `approval_kind`, and `recoverability`.
5. For a write request, verify mutation metadata and the final blocker/success event.
6. Do not “fix” a discovery result by directly adding a tool to the active registry.

These fields are diagnostics, not bypasses. Existing execution profiles, allowlists, denylists, approval rules, workspace scope, and file path protections remain authoritative.
