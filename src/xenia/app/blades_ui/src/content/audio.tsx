import type { BladeDef } from "../ui/blade";
import { send, useNative, type NativeState } from "../native";
import { NextLaunchNote, Options, Slider, Toggle, Value, type Option } from "../ui/settings-controls";
import { Aside } from "./parts";
import { t } from "../i18n";

// "any" plays through SDL on a Mac, so it reads as the Mac's output.
const OUTPUTS: Option[] = [
  { value: "sdl", get label() { return t("audio.yourMac"); }, get short() { return t("audio.yourMac"); }, recommended: true },
  { value: "nop", get label() { return t("home.gameFolders.none"); }, get short() { return t("home.gameFolders.none"); } }
];

/** The console's music player, while a game is using it. */
function MusicPlayerControls() {
  const { musicPlayer, titleOpen } = useNative();
  if (!musicPlayer.available) return null;
  if (!titleOpen || musicPlayer.state === "idle") {
    return <Aside>{t("audio.music.empty")}</Aside>;
  }
  const playing = musicPlayer.state === "playing";
  return (
    <>
      <p className="pref-group">{playing ? t("audio.playing") : t("audio.paused")}</p>
      <div className="flex flex-wrap gap-[0.5em]">
        <button
          type="button"
          className="pref-option rounded-row inline-flex min-h-[2.2em] cursor-pointer items-center px-[0.85em]"
          onClick={() => send({ op: "music", id: "previous" })}
        >
          {t("audio.previous")}
        </button>
        <button
          type="button"
          className="pref-option rounded-row inline-flex min-h-[2.2em] cursor-pointer items-center px-[0.85em]"
          onClick={() => send({ op: "music", id: playing ? "pause" : "resume" })}
        >
          {playing ? t("audio.pause") : t("audio.play")}
        </button>
        <button
          type="button"
          className="pref-option rounded-row inline-flex min-h-[2.2em] cursor-pointer items-center px-[0.85em]"
          onClick={() => send({ op: "music", id: "next" })}
        >
          {t("audio.next")}
        </button>
      </div>
      <p className="pref-group">{t("volume.label")}</p>
      <div className="flex flex-wrap gap-[0.5em]" role="radiogroup" aria-label={t("audio.musicVolume")}>
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
  { value: "new", get label() { return t("audio.new"); }, get short() { return t("audio.new"); } },
  { value: "old", get label() { return t("audio.previous"); }, get short() { return t("audio.previous"); } },
  { value: "master", get label() { return t("audio.original"); }, get short() { return t("audio.original"); } },
  { value: "fake", get label() { return t("audio.silent"); }, get short() { return t("audio.silent"); } }
];

export function audio(state: NativeState): BladeDef {
  const output = String(state.cvars.apu?.value ?? "any") === "nop" ? "nop" : "sdl";
  return {
    id: "audio",
    label: t("audio.tab"),
    title: t("advanced.audio"),
    tone: "blue",
    menu: [
      {
        id: "volume",
        label: t("volume.label"),
        glyph: "sound",
        value: <Value name="volume" format={(v) => t("volume.value", { percent: Number(v) })} />,
        heading: t("volume.label"),
        detail: (
          <>
            <p>{t("volume.detail")}</p>
            <Slider
              name="volume"
              title={t("volume.label")}
              min={0}
              max={100}
              step={5}
              format={(v) => t("volume.value", { percent: Math.round(v) })}
            />
          </>
        )
      },
      {
        id: "sound",
        label: t("audio.gameSound"),
        glyph: "sound",
        value: <Value name="mute" options={[{ value: false, label: t("graphics.unsetInputs.on") }, { value: true, label: t("audio.muted") }]} />,
        heading: t("audio.gameSound"),
        detail: (
          <>
            <p>{t("audio.sound.detail")}</p>
            <Toggle name="mute" title={t("audio.gameSound")} off={t("graphics.unsetInputs.on")} on={t("audio.muted")} />
          </>
        )
      },
      {
        id: "output",
        label: t("audio.output"),
        glyph: "plug",
        value: <Value name="apu" options={OUTPUTS} current={output} />,
        heading: t("audio.output"),
        detail: (
          <>
            <p>{t("audio.output.detail")}</p>
            <Options name="apu" title={t("audio.output")} options={OUTPUTS} current={output} />
            <NextLaunchNote names={["apu"]} />
          </>
        )
      },
      {
        id: "latency",
        label: t("audio.latency"),
        glyph: "clock",
        value: <Value name="apu_max_queued_frames" format={(v) => t("audio.frames", { count: Number(v) })} />,
        heading: t("audio.latency.heading"),
        detail: (
          <>
            <p>{t("audio.latency.detail")}</p>
            <Slider
              name="apu_max_queued_frames"
              title={t("audio.latency.heading")}
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
        label: t("audio.decoder"),
        glyph: "chip",
        value: <Value name="xma_decoder" options={DECODERS} />,
        heading: t("audio.decoder"),
        detail: (
          <>
            <p>{t("audio.decoder.detail")}</p>
            <Options name="xma_decoder" title={t("audio.decoder")} options={DECODERS} />
            <p className="pref-group">{t("audio.decodeOnItsOwnThread")}</p>
            <Toggle name="use_dedicated_xma_thread" title={t("audio.decodeOnItsOwnThread")} />
            <Aside>{t("audio.decoder.threadNote")}</Aside>
            <NextLaunchNote names={["xma_decoder", "use_dedicated_xma_thread"]} />
          </>
        )
      },
      {
        id: "music",
        label: t("audio.musicPlayer"),
        glyph: "play",
        value: <Value name="enable_xmp" />,
        heading: t("audio.musicPlayer"),
        detail: (
          <>
            <p>{t("audio.music.detail")}</p>
            <Toggle name="enable_xmp" title={t("audio.musicPlayer")} />
            <NextLaunchNote names={["enable_xmp"]} />
            <MusicPlayerControls />
          </>
        )
      }
    ]
  };
}
