/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2014 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XDB_UI_MAIN_FRAME_H_
#define XDB_UI_MAIN_FRAME_H_

#include <memory>

#include <xdb/debug_target.h>
#include <xdb/ui/xdb_ui.h>

namespace xdb {
namespace ui {

class MainFrame : public MainFrameBase {
 public:
  MainFrame(std::unique_ptr<DebugTarget> debug_target);

 protected:
  void OnIdle(wxIdleEvent& event) override;

 private:
  std::unique_ptr<DebugTarget> debug_target_;
  std::unique_ptr<Cursor> cursor_;
};

}  // namespace ui
}  // namespace xdb

#endif  // XDB_UI_MAIN_FRAME_H_
