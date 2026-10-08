import type { NativeState, Renderer } from "../native";

/**
 * Whether settings that only one renderer reads belong on screen: the
 * renderer is running, or the `gpu` setting picks it for the next launch.
 * Choosing Vulkan in Graphics therefore shows its options before the restart.
 */
export function rendererShown(state: NativeState, renderer: Exclude<Renderer, "none">) {
  return state.renderer === renderer || state.selectedRenderer === renderer;
}
