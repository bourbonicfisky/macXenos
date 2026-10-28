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
  { value: true, label: "60 Hz", short: "60 Hz", recommended: true },
  { value: false, label: "Unlocked", short: "Unlocked" }
];

const SCALES: Option[] = [
  { value: 1, label: "1x — 720p", short: "1x" },
  { value: 2, label: "2x — 1440p", short: "2x" },
  { value: 3, label: "3x — 2160p", short: "3x" }
];

const UPSCALER_LABELS: Record<string, Option> = {
  bilinear: { value: "bilinear", label: "Bilinear", short: "Bilinear" },
  cas: { value: "cas", label: "AMD CAS", short: "CAS" },
  fsr: { value: "fsr", label: "AMD FSR 1.0", short: "FSR" },
  metalfx: { value: "metalfx", label: "Apple MetalFX", short: "MetalFX" }
};

const ANISOTROPY: Option[] = [
  { value: -1, label: "As the game asks", short: "Game" },
  { value: 0, label: "Off", short: "Off" },
  { value: 2, label: "2x", short: "2x" },
  { value: 3, label: "4x", short: "4x" },
  { value: 4, label: "8x", short: "8x" },
  { value: 5, label: "16x", short: "16x" }
];

const WINDOW_SIZING: Option[] = [
  { value: "scaled", label: "Game size", short: "Game size" },
  { value: "actual", label: "Exact pixels", short: "Exact" },
  { value: "off", label: "Keep my size", short: "Keep" }
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
      label: "Renderer",
      glyph: "monitor",
      value: <Value name="gpu" options={RENDERERS} current={renderer} />,
      heading: "Renderer",
      detail: (
        <>
          <p>
            Metal draws every frame natively on your Mac&rsquo;s GPU and is the
            fastest. Vulkan runs through MoltenVK and is kept for comparison
            when a game looks wrong.
          </p>
          <Options name="gpu" title="Renderer" options={RENDERERS} current={renderer} />
          <NextLaunchNote names={["gpu"]} />
        </>
      )
    },
    {
      id: "refresh",
      label: "Refresh rate",
      glyph: "speed",
      value: <>{!vsync ? "Unlocked" : limit > 0 ? `${limit} Hz` : "60 Hz"}</>,
      heading: "Refresh rate",
      detail: (
        <>
          <p>
            The Xbox 360 refreshed 60 times a second, and games set their own
            frame rate from that beat, usually 30 or 60. Keep it at 60 Hz, as
            the console ran.
          </p>
          <Options
            name="vsync"
            title="Refresh rate"
            options={REFRESH}
            current={vsync && limit === 0 ? true : vsync ? -1 : false}
            onChoose={(value) => {
              setCvar("vsync", value);
              setCvar("framerate_limit", 0);
            }}
          />
          <Aside>
            Unlocked removes the beat. Games that time themselves by it run as
            fast as they can, which for most means too fast.
          </Aside>
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
      label: "Internal resolution",
      glyph: "resolution",
      value: <>{scaleX === scaleY ? `${scaleX}x` : `${scaleX}x${scaleY}`}</>,
      heading: "Internal resolution",
      detail: (
        <>
          <p>
            Draws the game itself at a higher resolution, so edges, text and
            distant detail are genuinely sharper rather than stretched. Most
            games draw at 720p.
          </p>
          <Options
            name="draw_resolution_scale_x"
            title="Internal resolution"
            options={SCALES}
            dropdown
            current={scaleX === scaleY ? scaleX : -1}
            onChoose={(value) => {
              setCvar("draw_resolution_scale_x", value);
              setCvar("draw_resolution_scale_y", value);
            }}
          />
          <Aside>
            Costs 4x the GPU work at 2x and 9x at 3x; a base M4 holding 30 fps
            at 1x manages about 10&ndash;25 at 2x.
          </Aside>
          <Aside>
            Video output, below, is different: it sets the resolution the
            console reports, not how much detail games draw.
          </Aside>
          <NextLaunchNote names={["draw_resolution_scale_x", "draw_resolution_scale_y"]} />
        </>
      )
    },
    {
      id: "video",
      label: "Video output",
      glyph: "monitor",
      value: <ConsoleValue id="resolution" />,
      heading: "Video output",
      detail: (
        <>
          <p>
            The TV the console thinks it is connected to. Games are told this
            resolution and shape, and the window opens at that size.
          </p>
          <p className="pref-group">Resolution</p>
          <ConsoleOptions id="resolution" />
          <ConsoleCheck id="widescreen" />
          <p className="pref-group">Region</p>
          <ConsoleOptions id="av_region" />
          <Aside>
            This doesn&rsquo;t make games sharper: most still draw at 720p and
            the console stretches the picture to fit. For more detail, raise
            Internal resolution above.
          </Aside>
          <ConsoleLockNote />
        </>
      )
    },
    {
      id: "upscaling",
      label: "Upscaling",
      glyph: "eye",
      value: <>{UPSCALER_LABELS[upscaler]?.short ?? upscaler}</>,
      heading: "Upscaling and sharpening",
      detail: (
        <>
          <p>
            How the finished picture is stretched to fit your window. On a 1080p
            screen they all cost almost nothing, so choose by look. On Retina,
            4K and 5K screens, MetalFX and FSR are the clearest, and MetalFX is
            lighter on older Macs.
          </p>
          <Options
            name="postprocess_scaling_and_sharpening"
            title="Upscaling"
            options={upscalers}
            current={upscaler}
          />
          {upscaler === "fsr" ? (
            <>
              <p className="pref-group">FSR sharpness reduction (lower is sharper)</p>
              <Slider
                name="postprocess_ffx_fsr_sharpness_reduction"
                title="FSR sharpness reduction"
                min={0}
                max={2}
                step={0.05}
              />
            </>
          ) : null}
          {upscaler === "fsr" || upscaler === "cas" ? (
            <>
              <p className="pref-group">CAS extra sharpness</p>
              <Slider
                name="postprocess_ffx_cas_additional_sharpness"
                title="CAS extra sharpness"
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
      label: "FPS counter",
      glyph: "speed",
      value: <Value name="show_fps" />,
      heading: "FPS counter",
      detail: (
        <>
          <p>
            Shows how many frames a second the game draws, in the top right
            corner while it runs. Most games aim for 30 or 60.
          </p>
          <Options
            name="show_fps"
            title="FPS counter"
            options={[
              { value: false, label: "Off" },
              { value: true, label: "On" }
            ]}
          />
          <Aside>It stays hidden under the dashboard.</Aside>
        </>
      )
    },
    {
      id: "screen",
      label: "Screen",
      glyph: "monitor",
      heading: "Screen",
      detail: (
        <>
          <p>
            Letterboxing keeps the game&rsquo;s shape, with bars where your
            screen is wider or taller. Dithering smooths banding in dark
            gradients.
          </p>
          <Check name="present_letterbox" label="Letterbox to keep the aspect ratio" />
          <Check name="postprocess_dither" label="Dither to smooth gradients" />
          <Check name="fullscreen" label="Start in full screen" />
          <p className="pref-group">Window size when a game starts</p>
          <Options name="window_guest_resolution_mode" title="Window size" options={WINDOW_SIZING} />
          <NextLaunchNote names={["window_guest_resolution_mode"]} />
        </>
      )
    }
  ];

  if (rendererShown(state, "vulkan")) {
    menu.push({
      id: "filtering",
      label: "Texture filtering",
      glyph: "layers",
      value: <Value name="anisotropic_override" options={ANISOTROPY} />,
      heading: "Texture filtering — Vulkan only",
      detail: (
        <>
          <p>
            Keeps floors and roads sharp at a slant. The Metal renderer follows
            each game&rsquo;s own setting, so this only applies with Vulkan.
          </p>
          <Options name="anisotropic_override" title="Texture filtering" options={ANISOTROPY} />
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
    label: "Troubleshooting",
    glyph: "wrench",
    heading: "Renderer switches",
    detail: (
      <>
        <p>
          The defaults are right for every game. Turn one off only to narrow down
          a rendering problem, and back on after.
        </p>
        {rendererShown(state, "metal") ? (
          <>
            <p className="pref-group">Metal renderer</p>
            <Check name="metal_msaa_clear_fold" label="Fold multisampled clears" />
            <Check name="metal_stencil_export" label="Single-pass depth and stencil rebuild" />
            <Check name="metal_resolve_span_coalesce" label="Coalesce resolve rows" />
            <Check name="metal_raster_sync_skip" label="Skip unchanged render-target syncs" />
            <Check name="metal_raster_mark_skip" label="Skip redundant tile marking" />
            <Check name="metal_fsr_half_precision" label="FSR in half precision" />
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
            <Check name="metal_serialize_submissions" label={t("graphics.serialize.label")} />
            <Aside>{t("graphics.serialize.aside")}</Aside>
          </>
        ) : null}
        {rendererShown(state, "vulkan") ? (
          <>
            <p className="pref-group">Vulkan renderer</p>
            <Check name="mvk_config_defaults" label="Apply curated MoltenVK defaults" />
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
            "metal_serialize_submissions",
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
            "metal_serialize_submissions",
            "mvk_config_defaults"
          ]}
        />
      </>
    )
  });

  return { id: "graphics", label: "graphics", title: "Graphics", tone: "teal", menu };
}
