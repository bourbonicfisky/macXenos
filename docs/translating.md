# Translating macXenos

macXenos's dashboard (the Blades settings, Home, All Games, profiles) and the
app's own menus are translated into the Xbox 360's languages, and Danish and
Finnish besides:

| Language | Code | Translated by |
| --- | --- | --- |
| English | `en` | the source |
| Dansk | `da` | machine-assisted, awaiting native review |
| Deutsch | `de` | machine-assisted, awaiting native review |
| Español | `es` | machine-assisted, awaiting native review |
| Français | `fr` | machine-assisted, awaiting native review |
| Italiano | `it` | machine-assisted, awaiting native review |
| 日本語 | `ja` | machine-assisted, awaiting native review |
| 한국어 | `ko` | machine-assisted, awaiting native review |
| Nederlands | `nl` | machine-assisted, awaiting native review |
| Norsk bokmål | `nb` | machine-assisted, awaiting native review |
| Polski | `pl` | machine-assisted, awaiting native review |
| Português (Brasil) | `pt-BR` | machine-assisted, awaiting native review |
| Русский | `ru` | foxsetDev |
| Suomi | `fi` | machine-assisted, awaiting native review |
| Svenska | `sv` | machine-assisted, awaiting native review |
| 简体中文 | `zh-Hans` | machine-assisted, awaiting native review |
| 繁體中文 | `zh-Hant` | machine-assisted, awaiting native review |

Still to come: Türkçe (`tr`).

A Mac set to a regional variant reads its language's catalog: Canadian
French reads `fr`, Mexican Spanish `es`. Some cases to know:

- **Portuguese.** There is only a Brazilian catalog, so every Portuguese,
  Portugal's included, reads it. A European Portuguese catalog is welcome; it
  would need its own code in `src/i18n/locales.ts`.
- **Chinese.** Taiwan, Hong Kong and Macau, and any `zh-Hant` setting, read
  繁體中文; all other Chinese reads 简体中文.
- **Norwegian.** Nynorsk and plain Norwegian read Bokmål.

**A review by a native speaker is welcome for every language**, including
the ones above: a pull request that fixes a few wrong or awkward phrases helps
as much as a whole new language.

The language you pick under **System > Dashboard language** is the
dashboard's alone. Games, and the language the emulated console reports to
them (**System > Language and region**), are separate, and translating the
dashboard never changes them.

## Where the text lives

Everything is under `src/xenia/app/blades_ui/`:

- `src/i18n/en.ts` is every message in English, and the one source for every
  language. Each message has a key like `"graphics.uncapped.label"`. Comments
  there say where some of them appear.
- `src/i18n/catalogs/<code>.ts` is one language: every key of `en.ts` with
  its translation. Copy the structure of an existing one, such as `ru.ts`.
- The keys starting `native.` are the app's menu bar, its About panel and the
  titles of its alerts and open panels. The build turns them into the app's
  `Localizable.strings`; you translate them in the same file as the rest.

## Adding a language

You need Node 22 or later.

1. Find the language's code in `src/i18n/locales.ts`, which lists each
   language with its name in its own language. A language that isn't there
   yet needs a line added (its BCP 47 code and its own name).
2. Create `src/i18n/catalogs/<code>.ts`:

   ```ts
   /**
    * Turkish (tr) translation of the macXenos dashboard and the app's menus.
    * Translated by <your name>.
    */
   import type { Catalog } from "../catalogs";

   export default {
     "legend.day": "Gündüz",
     "legend.night": "Gece",
     // ...every key in en.ts, in the same order
   } satisfies Catalog;
   ```

   It needs no other wiring: it appears under Dashboard language, and a Mac
   set to that language picks it.
3. In `src/xenia/app/blades_ui`, run:

   ```sh
   npm install
   npm run build    # the page, and the menus' Localizable.strings
   npm run check    # every key there, placeholders, plurals (see below)
   npm test         # unit tests, then the browser tests
   ```
4. To see it: `npm run dev`, open the address it prints, and choose the
   language under System > Dashboard language. `npm run layout -- /tmp/l`
   lists text that doesn't fit, in every language, and saves a screenshot of
   each blade (it needs Chrome, Chromium, Helium, Brave or Edge).
5. Commit the catalog, `dist/` and `src/xenia/app/localization/`.

`npm run check` prints each language's coverage, and fails until the new
catalog has every key. While you work, the dashboard shows English for any
key you haven't translated yet.

## Rules for messages

**Placeholders.** `{name}`, `{count}` and the like are filled in by the app.
Keep each one exactly as in English: the same names, none added or dropped.
You may move them to where your language needs them.

    "allGames.achievements": "{unlocked} of {total} Achievements"
    "allGames.achievements": "Osiągnięcia: {unlocked} z {total}"

**Plurals.** A message that counts things is written in ICU form:

    "{count, plural, one {# folder} other {# folders}}"

`#` is the number. Keep the variable and the word `plural`, and give one
branch for each of your language's plural categories, as Unicode CLDR
defines them (what `Intl.PluralRules` uses):

| Language | Categories to give |
| --- | --- |
| English, German, Dutch, Swedish, Italian, Spanish, Portuguese | `one`, `other` |
| Danish, Norwegian, Finnish | `one`, `other` |
| French, Brazilian Portuguese | `one` (0 and 1), `other` |
| Russian, Polish | `one`, `few`, `many`, `other` |
| Japanese, Korean, Chinese | `other` only |

For example, Russian and Korean:

    "{count, plural, one {# папка} few {# папки} many {# папок} other {# папки}}"
    "{count, plural, other {폴더 #개}}"

The check rejects a category your language doesn't have and asks for any
that a whole number can select. (Some languages have a category for very
large numbers only, such as French `many` for millions; you may give it, but
needn't.) An exact-number branch like `=0 {none}` may stay.

**Names stay as they are.** macXenos, Blades, Xbox, Xbox 360, Xbox Live, Mac,
macOS, Metal, MetalFX, Vulkan, MoltenVK, FSR, CAS, FXAA, Discord, Kinect, game
titles, button letters (A, B, X, Y, LB, RB), the Guide button's name (the
dashboard draws the button beside it), Back + Start, keys (F6, ⌘), setting
names written `like_this`, and file names.

**Use the platforms' own words.** Where the Xbox 360 had a word in your
language (achievements, Gamerscore, gamertag, profile, gamer picture, title
update, memory unit), use Microsoft's. Where macOS has one (Settings, Finder,
the File, View and Help menus, About, Hide, Quit, full screen), use Apple's.
The application menu items (`native.app.*`) should read exactly as in every
other Mac app in your language.

**Keep short labels short.** Tabs (keys ending `.tab`, lowercase as on the
360's Blades tabs), the legend's buttons, option sheets (`*.short`) and row
labels sit in fixed spaces; prefer the shortest natural wording there. The
longer text (keys ending `.detail`, `.body`, `.aside`, `.note`) has room.

**Typography.** Use your language's quotation marks, apostrophes and spacing
(for example French « » and a no-break space before `:`), and `…` rather
than `...`. Chinese, Japanese and Korean use the Mac's own fonts (PingFang,
Hiragino Sans, Apple SD Gothic Neo), as the dashboard's font has only Latin
letters.

## Correcting a translation

Find the message in `src/i18n/catalogs/<code>.ts` (search for the wrong text),
change it, then run `npm run build` and `npm run check`. If the English is
what's wrong or unclear, say so in the pull request: the English in `en.ts`
is the source for everyone.
