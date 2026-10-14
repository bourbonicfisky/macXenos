# Developer documentation

Engineering records for macXenos. For building from source, see
[Building macXenos](../building.md). For the app itself, see the
[user guide](../user_guide.md).

This directory mixes native-Metal project records with documentation
inherited from Xenia. It is intentionally not a flat list of equally
authoritative inputs. Agents must also follow
[the documentation-specific instructions](../AGENTS.md).

## Native Metal: start here

| Need | Read first | Role |
| --- | --- | --- |
| Current capability or missing-gap question | [Metal current state](metal_current_state.md) | Compact, commit-pinned routing summary |
| Delivery, evidence, and reference policy | [Fable Metal recovery strategy](fable_recovery_strategy.md) | Standing policy; volatile branch and task state lives elsewhere |
| How OpenSpec work is run | [Delivery strategy](delivery_strategy.md) | Standing lane discipline (at most two active changes) |
| Which reference repository to consult | [Reference repository policy](reference_repository_policy.md) | Canary, Edge, XeniOS and XenDroid roles |
| Architecture research and backlog provenance | [Metal architecture gap roadmap](metal_architecture_gap_roadmap.md) | Dated research ledger, not live status authority |
| Exact foundation lineage and integration evidence | [Metal foundation integration manifest](metal_foundation_integration_manifest.md) | Frozen integration record |
| Ordered D3D12 draw comparison | [D3D12 IssueDraw parity ledger](metal_d3d12_issue_draw_parity_ledger.md) | Dated parity audit |
| How games are tested live, and which ones | [Live game tests](live_game_tests.md) | Scripted routes and the validation sweep |
| Current live observations | [Metal live compatibility matrix](metal_live_compatibility_matrix.md) | Observation record, not broad completion evidence |
| Presentation and gamma | [Metal presentation/gamma status](metal_presentation_gamma_status.md) | Current delivered scopes and remaining boundaries |
| Promoted behavior contracts | `../../openspec/specs/metal-*/spec.md` | Standing requirements |

Decision documents, censuses, negative experiments, and title investigations
(the other `metal_*.md` files here and everything in `evidence/`) are
valuable for their named question and pinned revision. They must not be
treated as current global status unless the current-state page links to them
for that specific claim.

Background reading: [AMD vs Apple and the challenges of Xbox emulation](metal-explainer.md)
explains why the same shader can pick different texture detail levels on
Apple and AMD GPUs.

## App interface

The macXenos front end (start screen, settings, profiles and the in-game
dashboard) is the Blades page in
[`src/xenia/app/blades_ui`](../../src/xenia/app/blades_ui/README.md), hosted
in a WKWebView over the game. The ImGui settings dialogs it replaced are
gone; add new user-facing settings there, not as dialogs.

## Moonshots

Ambitious features recorded for one day, with what is already known:
[moonshots.md](moonshots.md) (custom soundtracks, original Xbox games through
XeFu, frame generation through MetalFX, exact EDRAM output merging).

## Release notes

Published release notes are in [`../releases/`](../releases/).

## Deprecated archive

Closed decisions, pre-gates, censuses, audits, investigations, and session
handoffs live under [`archive/`](archive/README.md), sorted by kind
(`decisions/`, `evidence/`, `closed-investigations/`, `handoffs/`). Every
file there is deprecated for current guidance. Do not ingest the archive
recursively; open an exact record only when a current decision, spec, source
comment, or test links to it.

## Inherited Xenia documentation

[`xenia/`](xenia/) holds `cpu.md`, `cpu_todo.md`, `gpu.md`, `kernel.md`,
`instruction_tracing.md`, `style_guide.md` and the PowerPC manuals in
`ppc/`. They come from the Xenia lineage and are useful background, but they
may describe Canary, Windows, Linux, Vulkan, or older source layouts rather
than this macOS Metal fork.

## Build inputs kept outside this directory

`docs/game_configs/` (built-in per-title configs copied into the app) and
`docs/acknowledgements/` (third-party notices shown in the app) are read by
the build. Don't move them without updating the build.
