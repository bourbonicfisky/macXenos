# Building macXenos

These steps build macXenos from source on a Mac. Most people don't need to
do this: release builds are on [macxenos.com](https://macxenos.com).

macXenos builds on macOS only. The Windows and Linux build files from Xenia
Canary have been removed. The Windows and Linux source code is still in the
tree so changes from Xenia Canary and Xenia Edge can be brought over cleanly,
but it isn't built or tested here.

## What you need

- A Mac with Apple silicon, running macOS 15 or later. Intel builds are made
  by cross-compiling on Apple silicon (see [Universal builds](#universal-builds)).
- **Xcode**, not just the Command Line Tools. The build compiles Metal
  shaders with `xcrun metal`. On Xcode 26 and later, install the Metal
  toolchain if Xcode hasn't already:
  `xcodebuild -downloadComponent MetalToolchain`.
- [Homebrew](https://brew.sh) packages:

  ```sh
  brew install cmake ninja python molten-vk vulkan-loader
  ```

  MoltenVK and the Vulkan loader are needed for the Vulkan renderer and are
  bundled into the app when you package it.
- Node.js 22 or later, only if you change the dashboard
  (`src/xenia/app/blades_ui`).

## Getting the source

```sh
git clone --recursive <repository URL> macxenos
cd macxenos
```

If you cloned without `--recursive`, fetch the submodules:

```sh
git submodule update --init --recursive
```

## Building

```sh
# Configure. The Metal renderer is off unless you ask for it.
cmake --preset default -DXENIA_GPU_METAL=ON

# Write build/version.h (CMake doesn't generate it).
python3 tools/build/generate_version_header.py --source-root . --output build/version.h

# Build the emulator.
cmake --build build --config Release --target macXenos
```

The program is written to `build/bin/macOS/Release/macXenos`.

The first build takes a while because it also builds `dxilconv`, the
library that converts Xbox 360 shaders into a form Apple's Metal Shader
Converter can read. Later builds only rebuild what changed.

Build configurations:

| Configuration | Use it for |
| --- | --- |
| `Release` | Playing and performance testing. |
| `Debug` | Stepping through code in a debugger. |
| `Checked` | Debug plus AddressSanitizer, for memory bugs. |

Add `-DXENIA_BUILD_TESTS=ON` to the configure step to build the test
suites as well. They land next to the emulator in `build/bin/macOS/<Configuration>/`.

## Running a development build

The emulator generates code at run time, which macOS allows only for signed
programs that ask for it. Sign the binary with the development entitlements
after each build:

```sh
codesign --force --sign - \
  --entitlements src/xenia/app/xenia.entitlements \
  build/bin/macOS/Release/macXenos
```

Then start it, optionally with a game:

```sh
build/bin/macOS/Release/macXenos "/path/to/game.iso"
```

A bare binary writes its log next to itself. The packaged app writes it to
`~/Library/Application Support/macXenos/xenia.log`.

## Packaging the app

`tools/build/package_mac.sh` turns a Release build into `macXenos.app` and a
`.dmg` under `build/dist/`.

For your own use, sign it ad hoc:

```sh
tools/build/package_mac.sh --adhoc
```

For a release, copy `.env.example` to `.env`, fill in your Developer ID
signing identity, and run it without `--adhoc`. Add `--notarize` to submit
the app to Apple for notarization. The comment at the top of the script
lists every option, including the version string and the update feed.

## Universal builds

A Universal 2 app (Apple silicon and Intel) needs an x86_64 build next to
the native one. `xb` doesn't do cross-compiles, so use CMake directly.
`CMAKE_SYSTEM_NAME` has to be set, or CMake ignores the processor setting:

```sh
cmake -S . -B build-x86_64 -G "Ninja Multi-Config" \
  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_SYSTEM_NAME=Darwin -DCMAKE_SYSTEM_PROCESSOR=x86_64 \
  -DCMAKE_OSX_ARCHITECTURES=x86_64 -DXENIA_GPU_METAL=ON
cp build/version.h build-x86_64/version.h
cmake --build build-x86_64 --config Release --target macXenos
```

The configure output should say `XE_TARGET_X86_64=TRUE`. Then package both
slices together. The second path is a folder whose `lib/` holds x86_64
copies of `libMoltenVK.dylib` and `libvulkan.1.dylib`:

```sh
tools/build/package_mac.sh \
  --x86_64 build-x86_64/bin/macOS/Release/macXenos \
  --x86_64-libs <folder with x86_64 Vulkan libraries>
```

## Code style

C++ follows the inherited Xenia [style guide](dev/xenia/style_guide.md).
Format changes with `clang-format` before committing.

## More for developers

- [Developer documentation](dev/README.md): the app's interface and
  documentation inherited from Xenia.
- [Dashboard (Blades) README](../src/xenia/app/blades_ui/README.md): building
  and testing the app's front end.
