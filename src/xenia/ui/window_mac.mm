/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2024 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/ui/window_mac.h"

#import <AppKit/AppKit.h>
#import <QuartzCore/CAMetalLayer.h>

#include "xenia/ui/surface_mac.h"
#include "xenia/ui/ui_event.h"
#include "xenia/ui/virtual_key.h"

// Delegate forwarding native window events back into the MacWindow.
@interface XeniaWindowDelegate : NSObject <NSWindowDelegate> {
  xe::ui::MacWindow* window_;
}
- (instancetype)initWithWindow:(xe::ui::MacWindow*)window;
@end

@implementation XeniaWindowDelegate
- (instancetype)initWithWindow:(xe::ui::MacWindow*)window {
  self = [super init];
  if (self) {
    window_ = window;
  }
  return self;
}
- (void)windowDidResize:(NSNotification*)notification {
  window_->HandleResize();
}
- (void)windowWillClose:(NSNotification*)notification {
  window_->HandleNativeClose();
}
- (void)windowDidBecomeKey:(NSNotification*)notification {
  window_->HandleFocus(true);
}
- (void)windowDidResignKey:(NSNotification*)notification {
  window_->HandleFocus(false);
}
@end

// Layer-hosting content view that accepts first-responder status so the window
// receives keyboard events, then forwards them into the MacWindow (and on into
// ImGui dialogs). Without this the view never sees keyDown:/keyUp:.
@interface XeniaContentView : NSView {
  xe::ui::MacWindow* window_;
}
- (instancetype)initWithFrame:(NSRect)frame
                       window:(xe::ui::MacWindow*)window;
@end

@implementation XeniaContentView
- (instancetype)initWithFrame:(NSRect)frame
                       window:(xe::ui::MacWindow*)window {
  self = [super initWithFrame:frame];
  if (self) {
    window_ = window;
  }
  return self;
}
- (BOOL)acceptsFirstResponder {
  return YES;
}
- (BOOL)canBecomeKeyView {
  return YES;
}
- (void)keyDown:(NSEvent*)event {
  // Note: intentionally not calling super, which would NSBeep on unhandled keys.
  window_->HandleKeyEvent((__bridge void*)event, true);
}
- (void)keyUp:(NSEvent*)event {
  window_->HandleKeyEvent((__bridge void*)event, false);
}
// Deliver the first click to an inactive window straight to the content (so the
// menu is usable without an extra activating click).
- (BOOL)acceptsFirstMouse:(NSEvent*)event {
  return YES;
}
// Mouse forwarding. button: 1=left, 2=right, 3=middle. action: 0=down, 1=up,
// 2=move, 3=wheel.
- (void)mouseDown:(NSEvent*)event {
  window_->HandleMouseEvent((__bridge void*)event, 1, 0);
}
- (void)mouseUp:(NSEvent*)event {
  window_->HandleMouseEvent((__bridge void*)event, 1, 1);
}
- (void)mouseDragged:(NSEvent*)event {
  window_->HandleMouseEvent((__bridge void*)event, 1, 2);
}
- (void)rightMouseDown:(NSEvent*)event {
  window_->HandleMouseEvent((__bridge void*)event, 2, 0);
}
- (void)rightMouseUp:(NSEvent*)event {
  window_->HandleMouseEvent((__bridge void*)event, 2, 1);
}
- (void)rightMouseDragged:(NSEvent*)event {
  window_->HandleMouseEvent((__bridge void*)event, 2, 2);
}
- (void)otherMouseDown:(NSEvent*)event {
  window_->HandleMouseEvent((__bridge void*)event, 3, 0);
}
- (void)otherMouseUp:(NSEvent*)event {
  window_->HandleMouseEvent((__bridge void*)event, 3, 1);
}
- (void)mouseMoved:(NSEvent*)event {
  window_->HandleMouseEvent((__bridge void*)event, 0, 2);
}
- (void)scrollWheel:(NSEvent*)event {
  window_->HandleMouseEvent((__bridge void*)event, 0, 3);
}
@end

namespace xe {
namespace ui {

namespace {

// Translates a native macOS NSEvent virtual key code (kVK_*) into Xenia's
// Windows-style VirtualKey for the navigation/editing keys ImGui consumes.
// Returns VirtualKey::kNone for keys that should be represented by their
// character instead (letters, digits, punctuation - derived by the caller).
VirtualKey TranslateMacKeyCode(uint16_t key_code) {
  switch (key_code) {
    case 0x24:  // kVK_Return
    case 0x4C:  // kVK_ANSI_KeypadEnter
      return VirtualKey::kReturn;
    case 0x30:  // kVK_Tab
      return VirtualKey::kTab;
    case 0x33:  // kVK_Delete (Backspace)
      return VirtualKey::kBack;
    case 0x75:  // kVK_ForwardDelete
      return VirtualKey::kDelete;
    case 0x35:  // kVK_Escape
      return VirtualKey::kEscape;
    case 0x31:  // kVK_Space
      return VirtualKey::kSpace;
    case 0x7B:  // kVK_LeftArrow
      return VirtualKey::kLeft;
    case 0x7C:  // kVK_RightArrow
      return VirtualKey::kRight;
    case 0x7D:  // kVK_DownArrow
      return VirtualKey::kDown;
    case 0x7E:  // kVK_UpArrow
      return VirtualKey::kUp;
    case 0x73:  // kVK_Home
      return VirtualKey::kHome;
    case 0x77:  // kVK_End
      return VirtualKey::kEnd;
    case 0x74:  // kVK_PageUp
      return VirtualKey::kPrior;
    case 0x79:  // kVK_PageDown
      return VirtualKey::kNext;
    case 0x7A:  // kVK_F1
      return VirtualKey::kF1;
    case 0x78:  // kVK_F2
      return VirtualKey::kF2;
    case 0x63:  // kVK_F3
      return VirtualKey::kF3;
    case 0x76:  // kVK_F4
      return VirtualKey::kF4;
    case 0x60:  // kVK_F5
      return VirtualKey::kF5;
    case 0x61:  // kVK_F6
      return VirtualKey::kF6;
    case 0x62:  // kVK_F7
      return VirtualKey::kF7;
    case 0x64:  // kVK_F8
      return VirtualKey::kF8;
    case 0x65:  // kVK_F9
      return VirtualKey::kF9;
    case 0x6D:  // kVK_F10
      return VirtualKey::kF10;
    case 0x67:  // kVK_F11
      return VirtualKey::kF11;
    case 0x6F:  // kVK_F12
      return VirtualKey::kF12;
    default:
      return VirtualKey::kNone;
  }
}

}  // namespace

std::unique_ptr<Window> Window::Create(WindowedAppContext& app_context,
                                       const std::string_view title,
                                       uint32_t desired_logical_width,
                                       uint32_t desired_logical_height) {
  return std::make_unique<MacWindow>(app_context, title, desired_logical_width,
                                     desired_logical_height);
}

std::unique_ptr<MenuItem> MenuItem::Create(Type type, const std::string& text,
                                           const std::string& hotkey,
                                           std::function<void()> callback) {
  return std::make_unique<MacMenuItem>(type, text, hotkey, callback);
}

MacWindow::~MacWindow() {
  EnterDestructor();
  @autoreleasepool {
    NSWindow* window = (__bridge_transfer NSWindow*)window_;
    window_ = nullptr;
    XeniaWindowDelegate* delegate =
        (__bridge_transfer XeniaWindowDelegate*)delegate_;
    delegate_ = nullptr;
    metal_view_ = nullptr;
    if (window) {
      [window setDelegate:nil];
      [window close];
    }
    (void)delegate;
  }
}

uint32_t MacWindow::GetLatestDpiImpl() const {
  CGFloat scale = 1.0;
  NSWindow* window = (__bridge NSWindow*)window_;
  if (window) {
    scale = [window backingScaleFactor];
  } else {
    scale = [[NSScreen mainScreen] backingScaleFactor];
  }
  return uint32_t(96.0 * scale);
}

bool MacWindow::OpenImpl() {
  @autoreleasepool {
    uint32_t logical_width = GetDesiredLogicalWidth();
    uint32_t logical_height = GetDesiredLogicalHeight();
    NSRect content_rect =
        NSMakeRect(0, 0, CGFloat(logical_width), CGFloat(logical_height));

    NSWindow* window = [[NSWindow alloc]
        initWithContentRect:content_rect
                  styleMask:(NSWindowStyleMaskTitled | NSWindowStyleMaskClosable |
                             NSWindowStyleMaskMiniaturizable |
                             NSWindowStyleMaskResizable)
                    backing:NSBackingStoreBuffered
                      defer:NO];

    // A layer-hosting view backed by a CAMetalLayer for MoltenVK presentation.
    // Use a first-responder-accepting subclass so keyboard input reaches ImGui.
    XeniaContentView* view =
        [[XeniaContentView alloc] initWithFrame:content_rect window:this];
    CAMetalLayer* metal_layer = [CAMetalLayer layer];
    metal_layer.contentsScale = [window backingScaleFactor];
    [view setLayer:metal_layer];
    [view setWantsLayer:YES];

    [window setContentView:view];
    [window setTitle:[NSString stringWithUTF8String:std::string(GetTitle())
                                                        .c_str()]];
    [window setReleasedWhenClosed:NO];
    // Deliver mouseMoved: to the first responder so ImGui gets hover updates
    // (menu highlight, etc.), not just clicks/drags.
    [window setAcceptsMouseMovedEvents:YES];

    XeniaWindowDelegate* delegate =
        [[XeniaWindowDelegate alloc] initWithWindow:this];
    [window setDelegate:delegate];

    [window center];
    [window makeKeyAndOrderFront:nil];
    // Route keyboard events to the content view.
    [window makeFirstResponder:view];

    window_ = (__bridge_retained void*)window;
    metal_view_ = (__bridge void*)view;
    delegate_ = (__bridge_retained void*)delegate;

    UpdateSizeFromView();
    OnSurfaceChanged(true);
  }
  return true;
}

void MacWindow::RequestCloseImpl() {
  NSWindow* window = (__bridge NSWindow*)window_;
  if (window) {
    [window performClose:nil];
  }
}

void MacWindow::FocusImpl() {
  NSWindow* window = (__bridge NSWindow*)window_;
  if (window) {
    [window makeKeyAndOrderFront:nil];
  }
}

void MacWindow::ApplyNewTitle() {
  NSWindow* window = (__bridge NSWindow*)window_;
  if (window) {
    [window setTitle:[NSString stringWithUTF8String:std::string(GetTitle())
                                                        .c_str()]];
  }
}

std::unique_ptr<Surface> MacWindow::CreateSurfaceImpl(
    Surface::TypeFlags allowed_types) {
  if ((allowed_types & Surface::kTypeFlag_MetalLayer) && metal_view_) {
    NSView* view = (__bridge NSView*)metal_view_;
    return std::make_unique<MetalLayerSurface>((__bridge void*)view.layer);
  }
  return nullptr;
}

void MacWindow::RequestPaintImpl() {
  // Coalesce: keep at most one paint queued so repeated RequestPaint calls (or a
  // self-reposting continuous paint) don't pile up.
  bool already_scheduled = paint_scheduled_.exchange(true);
  if (already_scheduled) {
    return;
  }
  // Schedule the paint on a *later* run-loop iteration via the main dispatch
  // queue. Deliberately NOT through CallInUIThreadDeferred: that routes into the
  // pending-functions queue, which ExecutePendingFunctionsFromUIThread drains in
  // a tight while-loop. A paint that re-requests itself every frame (e.g. while
  // an ImGui drawer is active, like the persistent macOS menu bar) would then
  // recurse there forever and never return to the Cocoa event loop -> beachball,
  // no input. A plain dispatch_async re-queues for a fresh loop pass, so input
  // events are processed between paints.
  dispatch_async(dispatch_get_main_queue(), ^{
    paint_scheduled_.store(false);
    if (!app_context().HasQuitFromUIThread()) {
      OnPaint();
    }
  });
}

void MacWindow::UpdateSizeFromView() {
  NSView* view = (__bridge NSView*)metal_view_;
  if (!view) {
    return;
  }
  NSSize backing = [view convertSizeToBacking:view.bounds.size];
  uint32_t physical_width = uint32_t(backing.width);
  uint32_t physical_height = uint32_t(backing.height);
  // Keep the CAMetalLayer's drawable size in physical pixels.
  CAMetalLayer* layer = (CAMetalLayer*)view.layer;
  layer.drawableSize = CGSizeMake(backing.width, backing.height);
  WindowDestructionReceiver destruction_receiver(this);
  // canary's OnActualSizeUpdate takes a WindowResizeAction; a native/user resize
  // is kManual (matching window_gtk.cc / window_win.cc).
  OnActualSizeUpdate(physical_width, physical_height,
                     WindowResizeAction::kManual, destruction_receiver);
}

void MacWindow::HandleResize() {
  UpdateSizeFromView();
}

void MacWindow::HandleKeyEvent(void* ns_event, bool is_down) {
  NSEvent* event = (__bridge NSEvent*)ns_event;
  if (!event) {
    return;
  }

  const NSEventModifierFlags flags = [event modifierFlags];
  const bool shift_pressed = (flags & NSEventModifierFlagShift) != 0;
  const bool ctrl_pressed = (flags & NSEventModifierFlagControl) != 0;
  const bool alt_pressed = (flags & NSEventModifierFlagOption) != 0;
  const bool super_pressed = (flags & NSEventModifierFlagCommand) != 0;

  // Resolve a Windows-style virtual key for the navigation/editing keys, and
  // fall back to deriving one from the (modifier-independent) character for
  // letters and digits so ImGui shortcut handling (Ctrl/Cmd+A/C/V/X/...) works.
  VirtualKey virtual_key = TranslateMacKeyCode([event keyCode]);
  if (virtual_key == VirtualKey::kNone) {
    NSString* bare = [event charactersIgnoringModifiers];
    if (bare && [bare length] > 0) {
      unichar c = [bare characterAtIndex:0];
      if (c >= 'a' && c <= 'z') {
        c = unichar(c - 'a' + 'A');
      }
      if ((c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9')) {
        virtual_key = VirtualKey(c);
      }
    }
  }

  WindowDestructionReceiver destruction_receiver(this);
  KeyEvent e(this, virtual_key, [event isARepeat] ? 1 : 0, !is_down,
             shift_pressed, ctrl_pressed, alt_pressed, super_pressed);

  if (is_down) {
    OnKeyDown(e, destruction_receiver);
    if (destruction_receiver.IsWindowDestroyedOrClosed()) {
      return;
    }
    // Forward printable characters to ImGui text fields. Skip when Ctrl/Cmd is
    // held so shortcuts don't also insert text.
    if (!ctrl_pressed && !super_pressed) {
      NSString* characters = [event characters];
      const NSUInteger length = characters ? [characters length] : 0;
      for (NSUInteger i = 0; i < length; ++i) {
        const unichar ch = [characters characterAtIndex:i];
        // Skip control chars, DEL, and the private-use range macOS uses for
        // function/arrow keys.
        if (ch < 0x20 || ch == 0x7F || (ch >= 0xF700 && ch <= 0xF8FF)) {
          continue;
        }
        KeyEvent char_event(this, VirtualKey(ch), 1, false, shift_pressed,
                            ctrl_pressed, alt_pressed, super_pressed);
        OnKeyChar(char_event, destruction_receiver);
        if (destruction_receiver.IsWindowDestroyedOrClosed()) {
          return;
        }
      }
    }
  } else {
    OnKeyUp(e, destruction_receiver);
  }
}

void MacWindow::HandleMouseEvent(void* ns_event, int button, int action) {
  NSEvent* event = (__bridge NSEvent*)ns_event;
  NSView* view = (__bridge NSView*)metal_view_;
  if (!event || !view) {
    return;
  }

  // Window-local position in physical (backing) pixels with a top-left origin -
  // the coordinate space ImGui expects (it scales by medium/actual DPI back to
  // logical points, consistent with the physical surface size). NSView reports
  // points with a bottom-left origin, so flip Y and scale by the backing factor.
  NSPoint loc_window = [event locationInWindow];
  NSPoint loc_view = [view convertPoint:loc_window fromView:nil];
  CGFloat scale = [[view window] backingScaleFactor];
  if (scale <= 0.0) {
    scale = 1.0;
  }
  int32_t x = int32_t(loc_view.x * scale);
  int32_t y = int32_t((view.bounds.size.height - loc_view.y) * scale);

  MouseEvent::Button mouse_button = MouseEvent::Button::kNone;
  switch (button) {
    case 1:
      mouse_button = MouseEvent::Button::kLeft;
      break;
    case 2:
      mouse_button = MouseEvent::Button::kRight;
      break;
    case 3:
      mouse_button = MouseEvent::Button::kMiddle;
      break;
    default:
      break;
  }

  int32_t scroll_y = 0;
  if (action == 3) {
    CGFloat dy = [event scrollingDeltaY];
    // Precise (trackpad) deltas are already in pixels; line deltas are in
    // notches, so scale them to Windows' WHEEL_DELTA units that ImGui expects.
    scroll_y = [event hasPreciseScrollingDeltas]
                   ? int32_t(dy)
                   : int32_t(dy * CGFloat(MouseEvent::kScrollPerDetent));
  }

  WindowDestructionReceiver destruction_receiver(this);
  MouseEvent e(this, mouse_button, x, y, 0, scroll_y);
  switch (action) {
    case 0:
      OnMouseDown(e, destruction_receiver);
      break;
    case 1:
      OnMouseUp(e, destruction_receiver);
      break;
    case 2:
      OnMouseMove(e, destruction_receiver);
      break;
    case 3:
      OnMouseWheel(e, destruction_receiver);
      break;
    default:
      break;
  }
}

void MacWindow::HandleNativeClose() {
  // For the single-window emulator, closing the window ends the app.
  app_context().QuitFromUIThread();
}

void MacWindow::HandleFocus(bool has_focus) {
  WindowDestructionReceiver destruction_receiver(this);
  OnFocusUpdate(has_focus, destruction_receiver);
}

}  // namespace ui
}  // namespace xe
