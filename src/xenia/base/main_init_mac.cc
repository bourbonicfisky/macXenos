/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2024 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/base/platform.h"

#if XE_ARCH_AMD64
#include <cstdio>
#include <cstdlib>

#include <xbyak/xbyak/xbyak_util.h>

// macOS counterpart of main_init_posix.cc, without the GTK dependency: the
// guest JIT requires AVX, so verify it before any initialization runs and bail
// out with a console message (there is no native dialog at this stage).
class StartupCpuFeatureCheck {
 public:
  StartupCpuFeatureCheck() {
    Xbyak::util::Cpu cpu;
    if (!cpu.has(Xbyak::util::Cpu::tAVX)) {
      std::fputs(
          "Your CPU does not support AVX, which is required by Xenia. See the "
          "FAQ for system requirements at https://xenia.jp\n",
          stderr);
      exit(1);
    }
  }
};

// Construct before any other initialization code so the AVX check happens first.
// Ref: https://reviews.llvm.org/D12689#243295
__attribute__((
    init_priority(101))) static StartupCpuFeatureCheck gStartupAvxCheck;
#endif  // XE_ARCH_AMD64
