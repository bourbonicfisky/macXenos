# AMD vs Apple and the challenges of Xbox emulation

### The short answer

Both GPUs run the bit-identical shader, but they compute one fixed-function step — interpolating a varying across the triangle — differently. AMD's interpolation is plane-exact (a straight line stays a straight line to the last bit); Apple's carries tiny per-pixel wobble. The shader then takes a derivative of that interpolated value to pick a texture mip, and the derivative amplifies Apple's wobble into a multi-mip error.



---

### Why it must be the interpolator

The lattice is purely texture-LOD selection (forcing both GPUs to the same mip collapses the cross-machine difference to ~0). The LOD comes from the screen-space gradient of the reconstructed UV o = A / W, where A = uv/w and W = 1/w are noperspective (screen-linear) varyings. Mode-4 decomposed the ballooning gradient: ~97% comes from the numerator A, ~3% from W. And the division that produces o is IEEE-deterministic — same inputs give the same bits on any GPU. So if AMD and Apple disagree on identical SPIR-V, the disagreement cannot be in the shader math; it can only be in how the hardware interpolated A before the shader ever saw it.

What actually differs

A varying across one covered 2x2 quad (should be a straight line):

```
  AMD (plane-exact):        Apple (wobbly):
   A                          A
   │      ●                   │      ●
   │    ●   ← exactly linear  │   ●    ← ±ulp jitter per pixel
   │  ●                       │ ●  ●
   └──────── x                └──────── x
   dA/dx = constant, correct   dA/dx = scatters quad-to-quad
```

- AMD (immediate-mode, RDNA): evaluates the varying from the triangle's plane equation at high precision, so a linear varying reconstructs exactly. Its finite difference dA/dx is the true, constant gradient. Clean LOD.
- Apple (TBDR, PowerVR lineage): interpolates with a different method/precision, so a "linear" varying isn't perfectly linear pixel-to-pixel. The value is close enough that the sample location looks fine — but the finite difference A(x+1) − A(x) is a small difference of large-ish numbers, so the ulp-level wobble becomes a large relative error. dA/dx then scatters by several mips from quad to quad.

▎ Fact vs. inference: that the divergence lives in fixed-function A interpolation is measured (identical SPIR-V + deterministic division + the 97% numerator split). Why Apple's interpolator isn't plane-exact — precision, barycentric scheme, TBDR tiling — is the plausible mechanism; we haven't dumped Apple's interpolator internals to pin the exact reason.

Why it shows up as dark strokes (and only here)

Two things have to line up, which is why you don't see this on most surfaces:

1. The derivative amplifies it. The wobble is invisible in the value (sample p the shader differentiates it for LOD. Xenia's determinism stack routes LODthrough exactly this reconstructed o, exposing the wobble.
2. SF4 baked ink-hatch detail into the texture's mip chain (high mips = dark).samples those dark high mips at scattered triangle-edge quads → dark strokestracing the mesh = the lattice. A texture with smooth mips would just look faintly blurry/sharp instead of drawing a wireframe.

The irony (and why the fix is what it is)

Xenia's whole "manual perspective" determinism stack assumes noperspective interpolation is plane-exact — which is true on AMD, and is why AMD is the clean reference. Apple quietly violates that assumption. That's why the fix can't be more shader math on the ivative trick still eats the wobbled A) — it has to compute the gradientstraight from the triangle's three vertices (geometry), bypassing hardware interpolation entirely. Which is exactly the Route 1 we built — and exactly why MoltenVK's missing per-vertex read is such an annoying wall.