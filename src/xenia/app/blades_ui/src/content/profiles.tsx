import { useEffect, useState } from "react";
import type { BladeDef } from "../ui/blade";
import type { MenuItem } from "../ui/blade-menu";
import { send, useNative, type NativeState, type Profile } from "../native";
import { showToast } from "../ui/toast";
import { Aside, PaneAction } from "./parts";
import { Gamercard } from "../ui/gamercard";
import { GamerPicturesPane, GamerPicturesValue } from "../ui/gamer-pictures";
import { t } from "../i18n";

/* ==========================================================================
   Profiles: the console's gamer profiles. Each holds its own saves and
   achievements; signing one in as a player is what a game sees. A profile
   signed in here stays signed in at the next launch.
   ========================================================================== */

/** The console's own rule (ProfileManager::IsGamertagValid). */
const GAMERTAG = /^[A-Za-z][A-Za-z0-9]*( [A-Za-z0-9]+)*$/;
export const validGamertag = (name: string) => name.length >= 1 && name.length <= 15 && GAMERTAG.test(name);

const SLOTS = [-1, 0, 1, 2, 3];
const slotName = (slot: number) => (slot < 0 ? "Signed out" : `Player ${slot + 1}`);

function SignIn({ profile }: { profile: Profile }) {
  return (
    <div role="radiogroup" aria-label={`${profile.gamertag} plays as`} className="mt-[0.3em] flex flex-wrap gap-[0.5em]">
      {SLOTS.map((slot) => (
        <button
          key={slot}
          type="button"
          role="radio"
          aria-checked={profile.slot === slot}
          className="pref-option rounded-row inline-flex min-h-[2.2em] cursor-pointer items-center whitespace-nowrap px-[0.85em]"
          onClick={() => {
            if (profile.slot === slot) return;
            send(slot < 0 ? { op: "profileSignOut", xuid: profile.xuid } : { op: "profileSignIn", xuid: profile.xuid, slot });
            showToast(profile.gamertag, slot < 0 ? "Signed out" : `Signed in as ${slotName(slot)}`);
          }}
        >
          {slotName(slot)}
        </button>
      ))}
    </div>
  );
}

/** A gamertag typed in: checked against the console's rule as you type. */
function GamertagField({
  title,
  initial,
  action,
  onCommit
}: {
  title: string;
  initial: string;
  action: string;
  onCommit: (name: string) => void;
}) {
  const [draft, setDraft] = useState(initial);
  useEffect(() => setDraft(initial), [initial]);
  const ok = validGamertag(draft);
  const changed = draft !== initial;
  return (
    <>
      <input
        type="text"
        spellCheck={false}
        maxLength={15}
        aria-label={title}
        placeholder="Gamertag"
        className="pref-text rounded-row mt-[0.3em] min-h-[2.2em] w-full px-[0.7em]"
        value={draft}
        onChange={(event) => setDraft(event.target.value)}
        onKeyDown={(event) => {
          if (event.key === "Enter" && ok && changed) {
            event.preventDefault();
            onCommit(draft);
          } else if (event.key === "Escape") {
            event.preventDefault();
            setDraft(initial);
            event.currentTarget.blur();
          }
        }}
      />
      {draft && !ok ? (
        <Aside>Up to 15 letters and numbers, starting with a letter; single spaces between words.</Aside>
      ) : null}
      <PaneAction disabled={!ok || !changed} onSelect={() => onCommit(draft)}>
        {action}
      </PaneAction>
    </>
  );
}

/** Deleting asks twice: it removes the profile's saves too. */
function DeleteProfile({ profile }: { profile: Profile }) {
  const { titleOpen } = useNative();
  const [armed, setArmed] = useState(false);
  useEffect(() => {
    if (!armed) return;
    const timer = window.setTimeout(() => setArmed(false), 6000);
    return () => window.clearTimeout(timer);
  }, [armed]);
  if (titleOpen) {
    return <Aside>Quit the game to delete a profile.</Aside>;
  }
  return (
    <button
      type="button"
      className="pane-button pane-button--danger rounded-row mt-[0.3em] inline-flex min-h-[2.2em] cursor-pointer items-center justify-self-start px-[0.9em]"
      data-armed={armed || undefined}
      onClick={() => {
        if (!armed) {
          setArmed(true);
          return;
        }
        setArmed(false);
        send({ op: "profileDelete", xuid: profile.xuid });
        showToast(profile.gamertag, "Profile deleted");
      }}
    >
      {armed ? "Press again to delete it and its saves" : "Delete profile"}
    </button>
  );
}

function ProfilePane({ xuid }: { xuid: string }) {
  const { profiles } = useNative();
  const profile = profiles.find((p) => p.xuid === xuid);
  if (!profile) return null;
  return (
    <>
      <p className="pref-group m-0">Sign in as</p>
      <SignIn profile={profile} />
      <p className="pref-group m-0">Gamertag</p>
      <GamertagField
        title={`Rename ${profile.gamertag}`}
        initial={profile.gamertag}
        action="Rename"
        onCommit={(name) => {
          send({ op: "profileRename", xuid: profile.xuid, gamertag: name });
          showToast("Profile renamed", name);
        }}
      />
      <p className="pref-group m-0">Saves</p>
      <PaneAction onSelect={() => send({ op: "profileFolder", xuid: profile.xuid })}>Show saves in Finder</PaneAction>
      <DeleteProfile profile={profile} />
      <p className="menu-spec m-0 text-[0.75em]">XUID {profile.xuid}</p>
    </>
  );
}

function CreateProfile() {
  return (
    <GamertagField
      title="New profile's gamertag"
      initial=""
      action="Create profile"
      onCommit={(name) => {
        send({ op: "profileCreate", gamertag: name });
        showToast("Profile created", name);
      }}
    />
  );
}

export function profiles(state: NativeState): BladeDef {
  const menu: MenuItem[] = state.profiles.map((profile) => ({
    id: `profile-${profile.xuid}`,
    label: profile.gamertag,
    glyph: "people",
    value: <>{slotName(profile.slot)}</>,
    heading: profile.gamertag,
    // Each profile's gamercard heads the pane while its row is highlighted.
    feature: <Gamercard profile={profile} />,
    detail: <ProfilePane xuid={profile.xuid} />
  }));
  menu.push({
    id: "create",
    section: state.profiles.length ? "New" : undefined,
    label: "Create a profile",
    glyph: "star",
    heading: "Create a profile",
    detail: (
      <>
        <p>
          Each profile keeps its own saves and achievements. Pick a gamertag, then sign the new
          profile in as a player.
        </p>
        <CreateProfile />
        {state.profiles.length === 0 ? (
          <Aside>Most games need a signed-in profile to save.</Aside>
        ) : null}
      </>
    )
  });
  if (state.gamerPictures.status !== "unavailable") {
    menu.push({
      id: "gamer-pictures",
      section: t("profiles.pictures.section"),
      label: t("profiles.pictures"),
      glyph: "people",
      value: <GamerPicturesValue />,
      heading: t("profiles.pictures"),
      detail: <GamerPicturesPane />
    });
  }

  // Otherwise the card of player one (or the first profile), as the
  // console's blade led with the signed-in player's.
  const player = state.profiles.find((profile) => profile.slot === 0) ?? state.profiles[0];
  return {
    id: "profiles",
    label: "profiles",
    title: "Profiles",
    tone: "rose",
    feature: player ? <Gamercard profile={player} /> : undefined,
    menu
  };
}
