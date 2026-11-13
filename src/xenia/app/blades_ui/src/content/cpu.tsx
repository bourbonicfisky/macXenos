import { t } from "../i18n";
import type { BladeDef } from "../ui/blade";
import { send, useNative, type NativeState } from "../native";
import {
  Check,
  NextLaunchNote,
  Options,
  UseRecommended,
  Value,
  announceSetting,
  type Option
} from "../ui/settings-controls";
import { Aside } from "./parts";

const QOS: Option[] = [
  { value: "off", get label() { return t("graphics.qos.off"); }, get short() { return t("graphics.qos.off"); } },
  { value: "main", get label() { return t("cpu.mainThread"); }, get short() { return t("cpu.main"); } },
  { value: "priority", get label() { return t("graphics.qos.priority"); }, get short() { return t("graphics.qos.priorityShort"); } },
  { value: "all", get label() { return t("graphics.qos.all"); }, get short() { return t("graphics.qos.allShort"); }, recommended: true }
];

const QOS_CLASS: Option[] = [
  { value: "user_initiated", get label() { return t("cpu.userInitiated"); }, get short() { return t("cpu.initiated"); } },
  { value: "user_interactive", get label() { return t("cpu.userInteractive"); }, get short() { return t("cpu.interactive"); } }
];

const SPEEDS = [
  { value: 0.5, get label() { return t("cpu.half"); } },
  { value: 1, get label() { return t("pref.volume.normal"); } },
  { value: 2, get label() { return t("cpu.double"); } }
];

/** Game speed changes the clock of the game that is running, right away. */
function Speed() {
  const { timeScalar, titleOpen } = useNative();
  return (
    <div role="radiogroup" aria-label={t("cpu.gameSpeed")} className="mt-[0.3em] flex flex-wrap gap-[0.5em]">
      {SPEEDS.map((speed) => (
        <button
          key={speed.value}
          type="button"
          role="radio"
          aria-checked={Math.abs(timeScalar - speed.value) < 1e-6}
          aria-disabled={!titleOpen || undefined}
          className="pref-option rounded-row inline-flex min-h-[2.2em] cursor-pointer items-center px-[0.85em]"
          onClick={() => {
            if (!titleOpen || Math.abs(timeScalar - speed.value) < 1e-6) return;
            send({ op: "action", id: "timeScalar", value: speed.value });
            announceSetting(t("cpu.gameSpeed"), speed.label);
          }}
        >
          {speed.label}
        </button>
      ))}
    </div>
  );
}

function SpeedValue() {
  const { timeScalar } = useNative();
  return <>{timeScalar === 1 ? t("pref.volume.normal") : `${timeScalar}x`}</>;
}

export function cpu(_state: NativeState): BladeDef {
  return {
    id: "cpu",
    label: t("cpu.tab"),
    title: "CPU",
    tone: "amber",
    menu: [
      {
        id: "speed",
        label: t("cpu.gameSpeed"),
        glyph: "speed",
        value: <SpeedValue />,
        heading: t("cpu.gameSpeed"),
        detail: (
          <>
            <p>{t("cpu.speed.detail")}</p>
            <Speed />
            <Aside>{t("cpu.speed.note")}</Aside>
          </>
        )
      },
      {
        id: "scheduling",
        label: t("cpu.threadScheduling"),
        glyph: "chip",
        value: <Value name="guest_thread_qos" options={QOS} />,
        heading: t("graphics.qos.label"),
        detail: (
          <>
            <p>{t("cpu.scheduling.detail")}</p>
            <Options name="guest_thread_qos" title={t("cpu.threadScheduling")} options={QOS} />
            <p className="pref-group">{t("cpu.priority")}</p>
            <p className="menu-note m-0 text-[0.78em]">{t("cpu.priority.note")}</p>
            <Options name="guest_thread_qos_class" title={t("cpu.threadPriority")} options={QOS_CLASS} />
            <NextLaunchNote names={["guest_thread_qos", "guest_thread_qos_class"]} />
          </>
        )
      },
      {
        id: "threads",
        label: t("cpu.gameThreads"),
        glyph: "layers",
        heading: t("cpu.gameThreads"),
        detail: (
          <>
            <p>{t("cpu.threads.detail")}</p>
            <Check name="ignore_thread_affinities" label={t("cpu.ignoreThreadPlacement")} recommended />
            <Check name="ignore_thread_priorities" label={t("cpu.ignoreThreadPriorities")} recommended />
            <UseRecommended
              names={["ignore_thread_affinities", "ignore_thread_priorities"]}
              title={t("cpu.gameThreads")}
            />
            <NextLaunchNote names={["ignore_thread_affinities", "ignore_thread_priorities"]} />
          </>
        )
      },
      {
        id: "compatibility",
        label: t("cpu.compatibility"),
        glyph: "wrench",
        heading: t("cpu.compatibilityFixes"),
        detail: (
          <>
            <p>{t("cpu.compatibility.detail")}</p>
            <Check name="ignore_trap_instructions" label={t("cpu.skipDebugTraps")} recommended />
            <Check
              name="disable_context_promotion"
              label={t("cpu.disableContextPromotion")}
              recommended
            />
            <Check name="disable_prefetch_and_cachecontrol" label={t("cpu.skipCacheHints")} recommended />
            <Check name="clock_no_scaling" label={t("cpu.useHostClock")} recommended />
            <Check name="clock_source_raw" label={t("cpu.readTheHardwareTimer")} recommended />
            <UseRecommended
              names={[
                "ignore_trap_instructions",
                "disable_context_promotion",
                "disable_prefetch_and_cachecontrol",
                "clock_no_scaling",
                "clock_source_raw"
              ]}
              title={t("cpu.compatibility")}
            />
            <NextLaunchNote
              names={[
                "ignore_trap_instructions",
                "disable_context_promotion",
                "clock_no_scaling",
                "clock_source_raw"
              ]}
            />
          </>
        )
      },
      {
        id: "patches",
        label: t("cpu.patchesAndPlugins"),
        glyph: "blocks",
        value: <Value name="apply_patches" />,
        heading: t("cpu.patchesAndPlugins"),
        detail: (
          <>
            <p>{t("cpu.patches.detail")}</p>
            <Check name="apply_patches" label={t("cpu.applyGamePatches")} />
            <Check name="allow_plugins" label={t("cpu.loadPluginsAndTrainers")} />
            <NextLaunchNote names={["apply_patches", "allow_plugins"]} />
          </>
        )
      }
    ]
  };
}
