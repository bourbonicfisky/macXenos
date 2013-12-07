/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2013 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include <alloy/compiler/pass.h>

#include <alloy/compiler/compiler.h>

using namespace alloy;
using namespace alloy::compiler;


Pass::Pass() :
    runtime_(0), compiler_(0) {
}

Pass::~Pass() {
}

int Pass::Initialize(Compiler* compiler) {
  runtime_ = compiler->runtime();
  compiler_ = compiler;
  return 0;
}
