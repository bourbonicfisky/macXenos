import type { BladeDef } from "../ui/blade";
import { send, useNative, type NativeState } from "../native";
import { NextLaunchNote, Options, Slider, Toggle, Value, type Option } from "../ui/settings-controls";
import { Aside } from "./parts";

// "any" plays through SDL on a Mac, so it reads as the Mac's output.
const OUTPUTS: Option[] = [
  { value: "sdl", label: "Your Mac", short: "Your Mac", recommended: true },
  { value: "nop", label: "None", short: "None" }
];

/** The console's music player, while a game is using it. */
function MusicPlayerControls() {
  const { musicPlayer, titleOpen } = useNative();
  if (!musicPlayer.available) return null;
  if (!titleOpen || musicPlayer.state === "idle") {
    return <Aside>Nothing is playing. A game that uses the player starts it.</Aside>;
  }
  const playing = musicPlayer.state === "playing";
  return (
    <>
      <p className="pref-group">{playing ? "Playing" : "Paused"}</p>
      <div className="flex flex-wrap gap-[0.5em]">
        <button
          type="button"
          className="pref-option rounded-row inline-flex min-h-[2.2em] cursor-pointer items-center px-[0.85em]"
          onClick={() => send({ op: "music", id: "previous" })}
        >
          Previous
        </button>
        <button
          type="button"
          className="pref-option rounded-row inline-flex min-h-[2.2em] cursor-pointer items-center px-[0.85em]"
          onClick={() => send({ op: "music", id: playing ? "pause" : "resume" })}
        >
          {playing ? "Pause" : "Play"}
        </button>
        <button
          type="button"
          className="pref-option rounded-row inline-flex min-h-[2.2em] cursor-pointer items-center px-[0.85em]"
          onClick={() => send({ op: "music", id: "next" })}
        >
          Next
        </button>
      </div>
      <p className="pref-group">Volume</p>
      <div className="flex flex-wrap gap-[0.5em]" role="radiogroup" aria-label="Music volume">
        {[0.25, 0.5, 0.75, 1].map((level) => (
          <button
            key={level}
            type="button"
            role="radio"
            aria-checked={Math.abs(musicPlayer.volume - level) < 0.05}
            className="pref-option rounded-row inline-flex min-h-[2.2em] cursor-pointer items-center px-[0.85em]"
            onClick={() => send({ op: "music", id: "volume", value: level })}
          >
            {Math.round(level * 100)}%
          </button>
        ))}
      </div>
    </>
  );
}

const DECODERS: Option[] = [
  { value: "new", label: "New", short: "New" },
  { value: "old", label: "Previous", short: "Previous" },
  { value: "master", label: "Original", short: "Original" },
  { value: "fake", label: "Silent", short: "Silent" }
];

export function audio(state: NativeState): BladeDef {
  const output = String(state.cvars.apu?.value ?? "any") === "nop" ? "nop" : "sdl";
  return {
    id: "audio",
    label: "audio",
    title: "Audio",
    tone: "blue",
    menu: [
      {
        id: "sound",
        label: "Game sound",
        glyph: "sound",
        value: <Value name="mute" options={[{ value: false, label: "On" }, { value: true, label: "Muted" }]} />,
        heading: "Game sound",
        detail: (
          <>
            <p>Mute every game&rsquo;s sound without touching your Mac&rsquo;s volume.</p>
            <Toggle name="mute" title="Game sound" off="On" on="Muted" />
          </>
        )
      },
      {
        id: "output",
        label: "Audio output",
        glyph: "plug",
        value: <Value name="apu" options={OUTPUTS} current={output} />,
        heading: "Audio output",
        detail: (
          <>
            <p>
              Games play through your Mac&rsquo;s current output. None runs them
              without any sound, for troubleshooting.
            </p>
            <Options name="apu" title="Audio output" options={OUTPUTS} current={output} />
            <NextLaunchNote names={["apu"]} />
          </>
        )
      },
      {
        id: "latency",
        label: "Latency",
        glyph: "clock",
        value: <Value name="apu_max_queued_frames" format={(v) => `${v} frames`} />,
        heading: "Audio latency",
        detail: (
          <>
            <p>
              How much sound is queued ahead. Fewer frames reach your ears
              sooner; too few crackle when a game is busy.
            </p>
            <Slider
              name="apu_max_queued_frames"
              title="Audio latency"
              min={4}
              max={64}
              step={1}
              format={(v) => `${Math.round(v)}`}
            />
            <NextLaunchNote names={["apu_max_queued_frames"]} />
          </>
        )
      },
      {
        id: "decoder",
        label: "Audio decoder",
        glyph: "chip",
        value: <Value name="xma_decoder" options={DECODERS} />,
        heading: "Audio decoder",
        detail: (
          <>
            <p>
              Games store sound as XMA. The new decoder is the most stable; try
              a previous one if a game&rsquo;s sound is broken.
            </p>
            <Options name="xma_decoder" title="Audio decoder" options={DECODERS} />
            <p className="pref-group">Decode on its own thread</p>
            <Toggle name="use_dedicated_xma_thread" title="Decode on its own thread" />
            <Aside>Its own thread is faster; turning it off can fix stutter in a few games.</Aside>
            <NextLaunchNote names={["xma_decoder", "use_dedicated_xma_thread"]} />
          </>
        )
      },
      {
        id: "music",
        label: "Music player",
        glyph: "play",
        value: <Value name="enable_xmp" />,
        heading: "Music player",
        detail: (
          <>
            <p>
              The console&rsquo;s music player. Some games, such as the
              Project Gotham Racing games, play their own soundtrack through
              it; with it off, those games have no music.
            </p>
            <Toggle name="enable_xmp" title="Music player" />
            <NextLaunchNote names={["enable_xmp"]} />
            <MusicPlayerControls />
          </>
        )
      }
    ]
  };
}
