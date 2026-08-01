# D3D12 SM5.1 tessellation bytecode (checked-in artifacts)

Prebuilt DXBC blobs consumed by the Metal backend's tessellation pipeline leg
(`port-xenios-tessellation`): 10 hull shaders
(`{discrete,continuous}_{triangle,quad}_{1cp,3cp,4cp}_hs.h`,
`adaptive_{triangle,quad}_hs.h`) and the 2 tessellation vertex shaders
(`tessellation_{indexed,adaptive}_vs.h`). The Metal chain feeds them through
dxilconv → Metal Shader Converter exactly like guest DXBC.

## Provenance

- Original artifacts were upstream Xenia's checked-in
  `src/xenia/gpu/shaders/bytecode/d3d12_5_1/` tree (BSD-3-Clause, (c) Xenia
  Developers), removed upstream by `f2ac39cfd` ("[Build] Generate shaders on
  build instead of having them checked in"). That path is now gitignored as a
  build-output directory (`.gitignore:66`), which is why these live here in a
  tracked metal-owned directory — the same relocation XeniOS made, and for the
  same reason.
- These copies were taken from the XeniOS fork
  (`XeniOS/src/xenia/gpu/metal/d3d12_5_1_bytecode/` @ `87b176a07`, same BSD-3
  lineage), which regenerated them against the CURRENT shared
  `xe_system_cbuffer` layout (`xe_textures_resolution_scaled` at offset 208 —
  matches our `dxbc_shader_translator.h:507`). The pre-`f2ac39cfd` historical
  blobs in our own git history differ only in the RDEF reflection chunk (older
  cbuffer member naming) and were deliberately NOT used.

## Regeneration

Sources are the shared HLSL in `src/xenia/gpu/shaders/`
(`*.hs.hlsl`, `tessellation_indexed.vs.hlsl`, `tessellation_adaptive.vs.hlsl`).
Compile with FXC (SM5.1, Windows or wine) via the existing wrapper:

```
python3 tools/build/compile_shader_dxbc.py \
    src/xenia/gpu/shaders/discrete_triangle_3cp.hs.hlsl \
    src/xenia/gpu/metal/d3d12_5_1_bytecode/discrete_triangle_3cp_hs.h
```

The wrapper derives the stage/profile from the `.hs.`/`.vs.` infix in the
source name and the C identifier from the filename; it needs FXC (`FXC_PATH`,
Windows SDK, or wine on macOS/Linux).

Any regeneration after a system-cbuffer layout change MUST rebuild all 12
together (the RDEF chunk embeds the layout) and re-run the tessellation
conformance suite.
