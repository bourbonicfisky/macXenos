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
  { value: "off", label: "Off", short: "Off" },
  { value: "main", label: "Main thread", short: "Main" },
  { value: "priority", label: "High-priority threads", short: "Priority" },
  { value: "all", label: "Every thread", short: "All" }
];

const QOS_CLASS: Option[] = [
  { value: "user_initiated", label: "User initiated", short: "Initiated" },
  { value: "user_interactive", label: "User interactive", short: "Interactive" }
];

const SPEEDS = [
  { value: 0.5, label: "Half" },
  { value: 1, label: "Normal" },
  { value: 2, label: "Double" }
];

/** Game speed changes the clock of the game that is running, right away. */
function Speed() {
  const { timeScalar, titleOpen } = useNative();
  return (
    <div role="radiogroup" aria-label="Game speed" className="mt-[0.3em] flex flex-wrap gap-[0.5em]">
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
            announceSetting("Game speed", speed.label);
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
  return <>{timeScalar === 1 ? "Normal" : `${timeScalar}x`}</>;
}

export function cpu(_state: NativeState): BladeDef {
  return {
    id: "cpu",
    label: "cpu",
    title: "CPU",
    tone: "amber",
    menu: [
      {
        id: "speed",
        label: "Game speed",
        glyph: "speed",
        value: <SpeedValue />,
        heading: "Game speed",
        detail: (
          <>
            <p>
              Runs the game&rsquo;s clock slower or faster. Most games keep
              their own time, so this is for testing more than playing.
            </p>
            <Speed />
            <Aside>Only while a game is running. It goes back to normal when the game closes.</Aside>
          </>
        )
      },
      {
        id: "scheduling",
        label: "Thread scheduling",
        glyph: "chip",
        value: <Value name="guest_thread_qos" options={QOS} />,
        heading: "Performance cores",
        detail: (
          <>
            <p>
              Asks macOS to keep the game&rsquo;s threads on performance cores.
              Off lets macOS decide, which suits most games.
            </p>
            <Options name="guest_thread_qos" title="Thread scheduling" options={QOS} />
            <p className="pref-group">Priority</p>
            <p className="menu-note m-0 text-[0.78em]">Interactive is the highest macOS gives.</p>
            <Options name="guest_thread_qos_class" title="Thread priority" options={QOS_CLASS} />
            <NextLaunchNote names={["guest_thread_qos", "guest_thread_qos_class"]} />
          </>
        )
      },
      {
        id: "threads",
        label: "Game threads",
        glyph: "layers",
        heading: "Game threads",
        detail: (
          <>
            <p>
              Games tell the 360 which of its six hardware threads to use and
              how urgent each is. Ignoring that can help a game that stalls.
            </p>
            <Check name="ignore_thread_affinities" label="Ignore thread placement" recommended />
            <Check name="ignore_thread_priorities" label="Ignore thread priorities" recommended />
            <UseRecommended
              names={["ignore_thread_affinities", "ignore_thread_priorities"]}
              title="Game threads"
            />
            <NextLaunchNote names={["ignore_thread_affinities", "ignore_thread_priorities"]} />
          </>
        )
      },
      {
        id: "compatibility",
        label: "Compatibility",
        glyph: "wrench",
        heading: "Compatibility fixes",
        detail: (
          <>
            <p>
              Fixes a few games need. Leave them as they are unless a
              game&rsquo;s page says otherwise.
            </p>
            <Check name="ignore_trap_instructions" label="Skip debug traps" recommended />
            <Check
              name="disable_context_promotion"
              label="Turn off context promotion (some sports games)"
              recommended
            />
            <Check name="disable_prefetch_and_cachecontrol" label="Skip cache hints" recommended />
            <Check name="clock_no_scaling" label="Use the Mac's clock directly" recommended />
            <Check name="clock_source_raw" label="Read the hardware timer" recommended />
            <UseRecommended
              names={[
                "ignore_trap_instructions",
                "disable_context_promotion",
                "disable_prefetch_and_cachecontrol",
                "clock_no_scaling",
                "clock_source_raw"
              ]}
              title="Compatibility"
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
        label: "Patches and plugins",
        glyph: "blocks",
        value: <Value name="apply_patches" />,
        heading: "Patches and plugins",
        detail: (
          <>
            <p>
              Game patches fix bugs and unlock frame rates; plugins and trainers
              are loaded from each game&rsquo;s plugins folder.
            </p>
            <Check name="apply_patches" label="Apply game patches" />
            <Check name="allow_plugins" label="Load plugins and trainers" />
            <NextLaunchNote names={["apply_patches", "allow_plugins"]} />
          </>
        )
      }
    ]
  };
}
