# Mochi Windows sandbox broker

This directory defines the versioned native boundary used by Mochi's Windows sandbox backend. The current executable is deliberately a fail-closed scaffold: it implements only the nonce-bound `hello` probe and reports every enforcement capability as false. It does not execute commands.

`sandbox.mode = required` therefore blocks execution on Windows until the broker implements and passes real AppContainer filesystem/network isolation, restricted-token and Job Object lifetime control, crash-safe per-run ACL ownership, recovery, and adversarial tests. Do not change capability flags to true based on configuration or mocked protocol responses.

## Build

From an MSVC Developer PowerShell with the Windows SDK installed:

```powershell
cmake -S native/mochi-sandbox-windows -B build/mochi-sandbox-windows
cmake --build build/mochi-sandbox-windows --config Release
```

The release executable is written to `mochi/runtime/sandbox/bin/` so Hatch can include it in Windows wheels. The Python backend still verifies protocol version, nonce, backend identity, version, and the complete observed capability set before use.

## Protocol

Frames are UTF-8 JSON objects terminated by one newline and capped at 1 MiB. `protocol.schema.json` documents the exact outer frame. Python performs stricter message-specific validation and binds `run` to the digest-bearing `SandboxPlan`; no command is accepted as a shell-constructed broker argument.

## Debugging and release gate

Run `mochi-sandbox-windows.exe --probe --nonce 0123456789abcdef` to inspect the handshake. A production helper must keep reporting `available=false` until real OS tests prove filesystem, process, and network containment together. A successfully compiled scaffold is not an available sandbox.
