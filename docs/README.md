# Project documentation

This directory mixes inherited Xenia documentation with native-Metal project
records. It is intentionally not a flat list of equally authoritative inputs.

## Native Metal: start here

| Need | Read first | Role |
| --- | --- | --- |
| Current capability or missing-gap question | [Metal current state](metal_current_state.md) | Compact, commit-pinned routing summary |
| Delivery, evidence, and reference policy | [Fable Metal recovery strategy](fable_recovery_strategy.md) | Standing policy; operational snapshots inside may be historical |
| Architecture research and backlog provenance | [Metal architecture gap roadmap](metal_architecture_gap_roadmap.md) | Dated research ledger, not live status authority |
| Exact foundation lineage and integration evidence | [Metal foundation integration manifest](metal_foundation_integration_manifest.md) | Frozen integration record |
| Ordered D3D12 draw comparison | [D3D12 IssueDraw parity ledger](metal_d3d12_issue_draw_parity_ledger.md) | Dated parity audit |
| Current live observations | [Metal live compatibility matrix](metal_live_compatibility_matrix.md) | Observation record, not broad completion evidence |
| Promoted behavior contracts | `../openspec/specs/metal-*/spec.md` | Standing requirements |

Decision documents, censuses, negative experiments, and title investigations are
valuable for their named question and pinned revision. They must not be treated
as current global status unless the current-state page links to them for that
specific claim.

Agents must also follow [the documentation-specific instructions](AGENTS.md).

## Inherited general documentation

`building.md`, `cpu.md`, `cpu_todo.md`, `gpu.md`, `kernel.md`,
`instruction_tracing.md`, `style_guide.md`, and `ppc/` originate from the Xenia
lineage. They are useful background but may describe Canary, Windows, Linux,
Vulkan, or older source layouts rather than this macOS Metal fork.
