# Domain Docs

How the engineering skills should consume this repo's domain documentation when exploring the codebase.

## This repo has multiple contexts

The repo is a single uv workspace holding one shared core and two domain packs. Contexts are declared in [`GLOSSARY-MAP.md`](../../GLOSSARY-MAP.md) at the repo root:

| Context | Glossary | ADRs |
| --- | --- | --- |
| Shared core (pipeline vocabulary) | `src/core/GLOSSARY.md` | `src/core/docs/adr/` |
| Rotating equipment fault diagnosis | `src/rotating/GLOSSARY.md` | `src/rotating/docs/adr/` |
| Pulmonary disease diagnosis | `src/pulmonary/GLOSSARY.md` | `src/pulmonary/docs/adr/` |

System-wide decisions live in `docs/adr/` at the repo root. Read the map first, then the glossary of whichever context the task touches.

## Before exploring, read these

- **`GLOSSARY-MAP.md`** at the repo root, then the relevant context's `GLOSSARY.md`.
- **`docs/adr/`**: read ADRs that touch the area you're about to work in.
- In a domain pack, also check `src/<context>/docs/adr/` for context-scoped decisions.

If any of these files don't exist, **proceed silently**. Don't flag their absence; don't suggest creating them upfront. The `/domain-modeling` skill creates them lazily when terms or decisions actually get resolved.

## Context rules

- **Never assume a term means the same thing across contexts.** `故障`, `症状` / `征兆`, and `诊断` are deliberately defined differently in each domain. Read the glossary for the context you are in.
- **A term used by both domains** belongs in `src/core/GLOSSARY.md` only if it means exactly the same thing in both. Otherwise define it separately in each context glossary.
- **Match the language**: these glossaries are written in Chinese, and new terms should be too.

## Use the glossary's vocabulary

When your output names a domain concept (in an issue title, a refactor proposal, a hypothesis, a test name), use the term as defined in the relevant `GLOSSARY.md`. Don't drift to synonyms the glossary explicitly avoids.

If the concept you need isn't in the glossary yet, that's a signal: either you're inventing language the project doesn't use (reconsider) or there's a real gap (note it for `/domain-modeling`).

## Flag ADR conflicts

If your output contradicts an existing ADR, surface it explicitly rather than silently overriding:

> _Contradicts ADR-0004 (concept + evidence dual node layer), but worth reopening because…_
