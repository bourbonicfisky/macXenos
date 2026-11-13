import type { BladeDef } from "../ui/blade";
import type { MenuItem } from "../ui/blade-menu";
import { setCvar, type NativeState } from "../native";
import {
  Check,
  NextLaunchNote,
  Options,
  ResetGroup,
  Slider,
  Value,
  type Option
} from "../ui/settings-controls";
import { Aside } from "./parts";
import { t } from "../i18n";
import {
  ConsoleCheck,
  ConsoleLockNote,
  ConsoleOptions,
  ConsoleValue
} from "../ui/console-controls";
import { rendererShown } from "./visibility";

const RENDERERS: Option[] = [
  { value: "metal", label: "Metal", short: "Metal", recommended: true },
  { value: "vulkan", label: "Vulkan (MoltenVK)", short: "Vulkan" }
];

// The emulated console's refresh: VSync on is 60 Hz, the beat games pace
// themselves to; off fires it as fast as it can (graphics_system.cc).
const REFRESH: Option[] = [
  { value: true, get label() { return t("graphics.60Hz"); }, get short() { return t("graphics.60Hz"); }, recommended: true },
  { value: false, get label() { return t("graphics.unlocked"); }, get short() { return t("graphics.unlocked"); } }
];

const SCALES: Option[] = [
  { value: 1, label: "1x — 720p", short: "1x" },
  { value: 2, label: "2x — 1440p", short: "2x" },
  { value: 3, label: "3x — 2160p", short: "3x" }
];

const UPSCALER_LABELS: Record<string, Option> = {
  bilinear: { value: "bilinear", get label() { return t("graphics.bilinear"); }, get short() { return t("graphics.bilinear"); } },
  cas: { value: "cas", label: "AMD CAS", short: "CAS" },
  fsr: { value: "fsr", label: "AMD FSR 1.0", short: "FSR" },
  metalfx: { value: "metalfx", label: "Apple MetalFX", short: "MetalFX" }
};

const ANISOTROPY: Option[] = [
  { value: -1, get label() { return t("graphics.asTheGameAsks"); }, get short() { return t("graphics.game"); } },
  { value: 0, get label() { return t("graphics.qos.off"); }, get short() { return t("graphics.qos.off"); } },
  { value: 2, label: "2x", short: "2x" },
  { value: 3, label: "4x", short: "4x" },
  { value: 4, label: "8x", short: "8x" },
  { value: 5, label: "16x", short: "16x" }
];

const WINDOW_SIZING: Option[] = [
  { value: "scaled", get label() { return t("graphics.gameSize"); }, get short() { return t("graphics.gameSize"); } },
  { value: "actual", get label() { return t("graphics.exactPixels"); }, get short() { return t("graphics.exact"); } },
  { value: "off", get label() { return t("graphics.keepMySize"); }, get short() { return t("graphics.keep"); } }
];

export function graphics(state: NativeState): BladeDef {
  const upscalerValue = String(state.cvars.postprocess_scaling_and_sharpening?.value ?? "");
  const upscaler = upscalerValue === "" ? "bilinear" : upscalerValue;
  const upscalers = state.upscalers.map((id) => UPSCALER_LABELS[id]).filter(Boolean);
  // "any" picks Metal on a Mac, so it reads as Metal.
  const renderer = String(state.cvars.gpu?.value ?? "any") === "vulkan" ? "vulkan" : "metal";
  const vsync = state.cvars.vsync?.value !== false;
  const limit = Number(state.cvars.framerate_limit?.value ?? 0);
  const limitUncapped = state.cvars.limit_uncapped_games?.value === true;
  const scaleX = Number(state.cvars.draw_resolution_scale_x?.value ?? 1);
  const scaleY = Number(state.cvars.draw_resolution_scale_y?.value ?? 1);

  const menu: MenuItem[] = [
    {
      id: "renderer",
      label: t("graphics.renderer"),
      glyph: "monitor",
      value: <Value name="gpu" options={RENDERERS} current={renderer} />,
      heading: t("graphics.renderer"),
      detail: (
        <>
          <p>{t("graphics.renderer.detail")}</p>
          <Options name="gpu" title={t("graphics.renderer")} options={RENDERERS} current={renderer} />
          <NextLaunchNote names={["gpu"]} />
        </>
      )
    },
    {
      id: "refresh",
      label: t("graphics.refreshRate"),
      glyph: "speed",
      value: <>{!vsync ? t("graphics.unlocked") : limit > 0 ? t("graphics.hertz", { count: limit }) : t("graphics.60Hz")}</>,
      heading: t("graphics.refreshRate"),
      detail: (
        <>
          <p>{t("graphics.refresh.detail")}</p>
          <Options
            name="vsync"
            title={t("graphics.refreshRate")}
            options={REFRESH}
            current={vsync && limit === 0 ? true : vsync ? -1 : false}
            onChoose={(value) => {
              setCvar("vsync", value);
              setCvar("framerate_limit", 0);
            }}
          />
          <Aside>{t("graphics.refresh.unlockedNote")}</Aside>
          <NextLaunchNote names={["vsync", "framerate_limit"]} />
        </>
      )
    },
    {
      id: "uncapped",
      label: t("graphics.uncapped.label"),
      glyph: "speed",
      value: (
        <>
          {limitUncapped ? t("graphics.uncapped.cap") : t("graphics.uncapped.unlimited")}
        </>
      ),
      heading: t("graphics.uncapped.label"),
      detail: (
        <>
          <p>{t("graphics.uncapped.body")}</p>
          <Options
            name="limit_uncapped_games"
            title={t("graphics.uncapped.label")}
            options={[
              { value: false, label: t("graphics.uncapped.unlimited") },
              { value: true, label: t("graphics.uncapped.cap") }
            ]}
            current={limitUncapped}
          />
        </>
      )
    },
    {
      id: "resolution",
      label: t("graphics.internalResolution"),
      glyph: "resolution",
      value: <>{scaleX === scaleY ? `${scaleX}x` : `${scaleX}x${scaleY}`}</>,
      heading: t("graphics.internalResolution"),
      detail: (
        <>
          <p>{t("graphics.resolution.detail")}</p>
          <Options
            name="draw_resolution_scale_x"
            title={t("graphics.internalResolution")}
            options={SCALES}
            dropdown
            current={scaleX === scaleY ? scaleX : -1}
            onChoose={(value) => {
              setCvar("draw_resolution_scale_x", value);
              setCvar("draw_resolution_scale_y", value);
            }}
          />
          <Aside>{t("graphics.resolution.cost")}</Aside>
          <Aside>{t("graphics.resolution.outputNote")}</Aside>
          <NextLaunchNote names={["draw_resolution_scale_x", "draw_resolution_scale_y"]} />
        </>
      )
    },
    {
      id: "video",
      label: t("graphics.videoOutput"),
      glyph: "monitor",
      value: <ConsoleValue id="resolution" />,
      heading: t("graphics.videoOutput"),
      detail: (
        <>
          <p>{t("graphics.video.detail")}</p>
          <p className="pref-group">{t("graphics.resolution")}</p>
          <ConsoleOptions id="resolution" />
          <ConsoleCheck id="widescreen" />
          <p className="pref-group">{t("graphics.region")}</p>
          <ConsoleOptions id="av_region" />
          <Aside>{t("graphics.video.note")}</Aside>
          <ConsoleLockNote />
        </>
      )
    },
    {
      id: "upscaling",
      label: t("graphics.upscaling"),
      glyph: "eye",
      value: <>{UPSCALER_LABELS[upscaler]?.short ?? upscaler}</>,
      heading: t("graphics.upscalingAndSharpening"),
      detail: (
        <>
          <p>{t("graphics.upscaling.detail")}</p>
          <Options
            name="postprocess_scaling_and_sharpening"
            title={t("graphics.upscaling")}
            options={upscalers}
            current={upscaler}
          />
          {upscaler === "fsr" ? (
            <>
              <p className="pref-group">{t("graphics.fsrSharpnessHint")}</p>
              <Slider
                name="postprocess_ffx_fsr_sharpness_reduction"
                title={t("graphics.fsrSharpnessReduction")}
                min={0}
                max={2}
                step={0.05}
              />
            </>
          ) : null}
          {upscaler === "fsr" || upscaler === "cas" ? (
            <>
              <p className="pref-group">{t("graphics.casExtraSharpness")}</p>
              <Slider
                name="postprocess_ffx_cas_additional_sharpness"
                title={t("graphics.casExtraSharpness")}
                min={0}
                max={1}
                step={0.05}
              />
            </>
          ) : null}
        </>
      )
    },
    {
      id: "fps",
      label: t("graphics.fpsCounter"),
      glyph: "speed",
      value: <Value name="show_fps" />,
      heading: t("graphics.fpsCounter"),
      detail: (
        <>
          <p>{t("graphics.fps.detail")}</p>
          <Options
            name="show_fps"
            title={t("graphics.fpsCounter")}
            options={[
              { value: false, label: t("graphics.qos.off") },
              { value: true, label: t("graphics.unsetInputs.on") }
            ]}
          />
          <Aside>{t("graphics.fps.note")}</Aside>
        </>
      )
    },
    {
      id: "screen",
      label: t("graphics.screen"),
      glyph: "monitor",
      heading: t("graphics.screen"),
      detail: (
        <>
          <p>{t("graphics.screen.detail")}</p>
          <Check name="present_letterbox" label={t("graphics.screen.letterbox")} />
          <Check name="postprocess_dither" label={t("graphics.screen.dither")} />
          <Check name="fullscreen" label={t("graphics.startInFullScreen")} />
          <p className="pref-group">{t("graphics.windowSize.heading")}</p>
          <Options name="window_guest_resolution_mode" title={t("graphics.windowSize")} options={WINDOW_SIZING} />
          <NextLaunchNote names={["window_guest_resolution_mode"]} />
        </>
      )
    }
  ];

  if (rendererShown(state, "vulkan")) {
    menu.push({
      id: "filtering",
      label: t("graphics.textureFiltering"),
      glyph: "layers",
      value: <Value name="anisotropic_override" options={ANISOTROPY} />,
      heading: t("graphics.textureFilteringVulkanOnly"),
      detail: (
        <>
          <p>{t("graphics.filtering.detail")}</p>
          <Options name="anisotropic_override" title={t("graphics.textureFiltering")} options={ANISOTROPY} />
          <NextLaunchNote names={["anisotropic_override"]} />
        </>
      )
    });
  }

  if (rendererShown(state, "metal")) {
    const unsetInputs: Option[] = [
      { value: "auto", label: t("graphics.unsetInputs.auto"), recommended: true },
      { value: "on", label: t("graphics.unsetInputs.on") },
      { value: "off", label: t("graphics.unsetInputs.off") }
    ];
    menu.push({
      id: "unset-inputs",
      label: t("graphics.unsetInputs.label"),
      glyph: "layers",
      value: <Value name="metal_unwritten_interpolators" options={unsetInputs} />,
      heading: t("graphics.unsetInputs.label"),
      detail: (
        <>
          <p>{t("graphics.unsetInputs.body")}</p>
          <Options
            name="metal_unwritten_interpolators"
            title={t("graphics.unsetInputs.label")}
            options={unsetInputs}
          />
          <Aside>{t("graphics.unsetInputs.aside")}</Aside>
        </>
      )
    });
  }

  // Guest-thread QoS (also under CPU). A Graphics row because it is the
  // single biggest frame-rate switch on Apple Silicon. Default "all" since
  // 6a14e43bc (M4 Pro Mafia II 35.6 -> 150.6, GTA IV 39.8 -> 60); 2026-10-15
  // M4 mini A/Bs: JC2 29.8 -> 27.5, RE5 13.7 -> 12.3; M1 Max: no loser.
  const qosPolicies: Option[] = [
    { value: "off", label: t("graphics.qos.off"), short: t("graphics.qos.off") },
    { value: "priority", label: t("graphics.qos.priority"), short: t("graphics.qos.priorityShort") },
    { value: "all", label: t("graphics.qos.all"), short: t("graphics.qos.allShort"), recommended: true }
  ];
  menu.push({
    id: "performance-cores",
    label: t("graphics.qos.label"),
    glyph: "chip",
    value: <Value name="guest_thread_qos" options={qosPolicies} />,
    heading: t("graphics.qos.label"),
    detail: (
      <>
        <p>{t("graphics.qos.body")}</p>
        <Options name="guest_thread_qos" title={t("graphics.qos.label")} options={qosPolicies} />
        <Aside>{t("graphics.qos.aside")}</Aside>
        <NextLaunchNote names={["guest_thread_qos"]} />
      </>
    )
  });

  menu.push({
    id: "troubleshooting",
    label: t("graphics.troubleshooting"),
    glyph: "wrench",
    heading: t("graphics.switches"),
    detail: (
      <>
        <p>{t("graphics.troubleshooting.detail")}</p>
        {rendererShown(state, "metal") ? (
          <>
            <p className="pref-group">{t("graphics.metalRenderer")}</p>
            <Check name="metal_msaa_clear_fold" label={t("graphics.foldMultisampledClears")} />
            <Check name="metal_stencil_export" label={t("graphics.singlePassDepthStencil")} />
            <Check name="metal_resolve_span_coalesce" label={t("graphics.coalesceResolveRows")} />
            <Check name="metal_raster_sync_skip" label={t("graphics.skipUnchangedSyncs")} />
            <Check name="metal_raster_mark_skip" label={t("graphics.skipRedundantTileMarking")} />
            <Check name="metal_fsr_half_precision" label={t("graphics.fsrInHalfPrecision")} />
            <p className="pref-group">{t("graphics.experimental.heading")}</p>
            <p className="m-0">{t("graphics.experimental.rasterExportsBody")}</p>
            <p className="pref-group">{t("graphics.experimental.rasterExports")}</p>
            <Options
              name="metal_memexport_raster_mode"
              title={t("graphics.experimental.rasterExports")}
              options={[
                { value: "auto", label: t("graphics.experimental.auto"), recommended: true },
                { value: "on", label: t("graphics.experimental.on") },
                { value: "off", label: t("graphics.experimental.off") }
              ]}
            />
            <Aside>{t("graphics.experimental.rasterExportsAside")}</Aside>
            <p className="pref-group">{t("graphics.serialize.heading")}</p>
            <p className="m-0">{t("graphics.serialize.body")}</p>
            <Check name="metal_completion_wait_exports" label={t("graphics.serialize.label")} />
            <Aside>{t("graphics.serialize.aside")}</Aside>
          </>
        ) : null}
        {rendererShown(state, "vulkan") ? (
          <>
            <p className="pref-group">{t("graphics.vulkanRenderer")}</p>
            <Check name="mvk_config_defaults" label={t("graphics.moltenVkDefaults")} />
          </>
        ) : null}
        <NextLaunchNote
          names={[
            "metal_msaa_clear_fold",
            "metal_stencil_export",
            "metal_resolve_span_coalesce",
            "metal_raster_sync_skip",
            "metal_raster_mark_skip",
            "metal_fsr_half_precision",
            "metal_memexport_raster_mode",
            "metal_completion_wait_exports",
            "mvk_config_defaults"
          ]}
        />
        <ResetGroup
          title={t("graphics.switches")}
          names={[
            "metal_msaa_clear_fold",
            "metal_stencil_export",
            "metal_resolve_span_coalesce",
            "metal_raster_sync_skip",
            "metal_raster_mark_skip",
            "metal_fsr_half_precision",
            "metal_memexport_raster_mode",
            "metal_completion_wait_exports",
            "mvk_config_defaults"
          ]}
        />
      </>
    )
  });

  return { id: "graphics", label: t("graphics.tab"), title: t("graphics.title"), tone: "teal", menu };
}
