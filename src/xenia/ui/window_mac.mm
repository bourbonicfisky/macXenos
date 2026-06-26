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

// Shows the standard macOS About panel. The app icon and version come from the
// bundle's Info.plist automatically; the credits block carries clickable links
// (the standard panel makes NSLinkAttributeName runs clickable). Used by both
// the application-menu About item and the cross-platform Help > About.
static void ShowMacAboutPanel() {
  NSMutableParagraphStyle* para = [[NSMutableParagraphStyle alloc] init];
  para.alignment = NSTextAlignmentCenter;
  para.paragraphSpacing = 6.0;
  NSDictionary* text_attrs = @{
    NSParagraphStyleAttributeName : para,
    NSFontAttributeName : [NSFont systemFontOfSize:11.0],
    NSForegroundColorAttributeName : [NSColor labelColor],
  };
  NSMutableAttributedString* credits =
      [[NSMutableAttributedString alloc] init];
  void (^add_line)(NSString*, NSString*) = ^(NSString* label, NSString* url) {
    [credits appendAttributedString:[[NSAttributedString alloc]
                                         initWithString:label
                                             attributes:text_attrs]];
    NSMutableDictionary* link_attrs = [text_attrs mutableCopy];
    link_attrs[NSLinkAttributeName] = url;
    [credits appendAttributedString:[[NSAttributedString alloc]
                                         initWithString:url
                                             attributes:link_attrs]];
    [credits appendAttributedString:[[NSAttributedString alloc]
                                         initWithString:@"\n\n"
                                             attributes:text_attrs]];
  };

  add_line(@"Mac build of Xenia Canary. Ported by dmg:\n",
           @"https://www.patreon.com/dmug");
  add_line(@"Original project:\n",
           @"https://github.com/xenia-canary/xenia-canary");
  add_line(@"Project homepage:\n", @"https://xenia.jp/about/");
  add_line(@"Game compatibility:\n",
           @"https://xenia-manager.github.io/compatibility");
  add_line(@"Report game compatibility:\n",
           @"https://github.com/xenia-project/game-compatibility/issues");

  [NSApp orderFrontStandardAboutPanel:@{
    NSAboutPanelOptionApplicationName : @"Xenia Canary, Mac Build",
    NSAboutPanelOptionCredits : credits,
  }];
  [NSApp activateIgnoringOtherApps:YES];
}

// Delegate forwarding native window events back into the MacWindow.
@interface XeniaWindowDelegate : NSObject <NSWindowDelegate> {
  xe::ui::MacWindow* window_;
}
- (instancetype)initWithWindow:(xe::ui::MacWindow*)window;
// Standard application-menu actions. Routed through the delegate (already
// retained for the window's lifetime) so the app menu needs no extra owner.
- (void)quit:(id)sender;
- (void)openAbout:(id)sender;
@end

@implementation XeniaWindowDelegate
- (instancetype)initWithWindow:(xe::ui::MacWindow*)window {
  self = [super init];
  if (self) {
    window_ = window;
  }
  return self;
}
- (void)quit:(id)sender {
  // Same clean close path as the cross-platform File>Exit item.
  if (window_) {
    window_->RequestClose();
  }
}
- (void)openAbout:(id)sender { ShowMacAboutPanel(); }
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

// Bridges a native NSMenuItem target/action to a stored C++ callback. Each
// kString MacMenuItem owns one of these (retained), wiring the menu click to
// the cross-platform MenuItem::OnSelected.
@interface XeniaMenuTarget : NSObject {
 @public
  std::function<void()> action;
}
- (void)invoke:(id)sender;
@end

@implementation XeniaMenuTarget
- (void)invoke:(id)sender {
  // Copy onto the stack first: the action may synchronously tear down the
  // owning MenuItem (and thus this target), as the cross-platform contract for
  // OnSelected warns.
  std::function<void()> fn = action;
  if (fn) {
    fn();
  }
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

// Strips Windows-style '&' mnemonic markers for native display, collapsing the
// literal "&&" escape to a single '&' (mirrors the Win32/GTK convention).
NSString* MakeNativeMenuTitle(const std::string& text) {
  std::string out;
  out.reserve(text.size());
  for (size_t i = 0; i < text.size(); ++i) {
    if (text[i] == '&') {
      if (i + 1 < text.size() && text[i + 1] == '&') {
        out.push_back('&');
        ++i;
      }
      // Otherwise drop the mnemonic marker.
    } else {
      out.push_back(text[i]);
    }
  }
  return [NSString stringWithUTF8String:out.c_str()];
}

// Builds the standard macOS application menu (the first, app-named menu) with
// About / Hide / Quit. Quit and About route through the window delegate; the
// Hide/Show items use the standard NSApplication responder-chain selectors.
NSMenuItem* CreateApplicationMenuItem(XeniaWindowDelegate* delegate) {
  NSMenuItem* app_item = [[NSMenuItem alloc] init];
  NSMenu* app_menu = [[NSMenu alloc] initWithTitle:@""];
  app_menu.autoenablesItems = NO;

  NSString* app_name = [[NSProcessInfo processInfo] processName];

  NSMenuItem* about =
      [[NSMenuItem alloc] initWithTitle:[@"About " stringByAppendingString:app_name]
                                 action:@selector(openAbout:)
                          keyEquivalent:@""];
  about.target = delegate;
  [app_menu addItem:about];
  [app_menu addItem:[NSMenuItem separatorItem]];

  NSMenuItem* hide =
      [[NSMenuItem alloc] initWithTitle:[@"Hide " stringByAppendingString:app_name]
                                 action:@selector(hide:)
                          keyEquivalent:@"h"];
  [app_menu addItem:hide];
  NSMenuItem* hide_others =
      [[NSMenuItem alloc] initWithTitle:@"Hide Others"
                                 action:@selector(hideOtherApplications:)
                          keyEquivalent:@"h"];
  hide_others.keyEquivalentModifierMask =
      NSEventModifierFlagCommand | NSEventModifierFlagOption;
  [app_menu addItem:hide_others];
  [app_menu addItem:[[NSMenuItem alloc] initWithTitle:@"Show All"
                                               action:@selector(unhideAllApplications:)
                                        keyEquivalent:@""]];
  [app_menu addItem:[NSMenuItem separatorItem]];

  NSMenuItem* quit =
      [[NSMenuItem alloc] initWithTitle:[@"Quit " stringByAppendingString:app_name]
                                 action:@selector(quit:)
                          keyEquivalent:@"q"];
  if (delegate) {
    quit.target = delegate;
  } else {
    // Fallback if the delegate isn't available yet: standard termination.
    quit.action = @selector(terminate:);
  }
  [app_menu addItem:quit];

  app_item.submenu = app_menu;
  return app_item;
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

MacMenuItem::MacMenuItem(Type type, const std::string& text,
                         const std::string& hotkey,
                         std::function<void()> callback)
    : MenuItem(type, text, hotkey, std::move(callback)) {
  @autoreleasepool {
    NSString* title = MakeNativeMenuTitle(text);
    switch (type) {
      case Type::kNormal: {
        // Root container: the menu bar itself.
        NSMenu* menu = [[NSMenu alloc] initWithTitle:@""];
        menu.autoenablesItems = NO;
        ns_menu_ = (__bridge_retained void*)menu;
        break;
      }
      case Type::kPopup: {
        // A top-level/submenu title carrying its own NSMenu of children.
        NSMenuItem* item = [[NSMenuItem alloc] initWithTitle:title
                                                      action:nil
                                               keyEquivalent:@""];
        NSMenu* submenu = [[NSMenu alloc] initWithTitle:title];
        submenu.autoenablesItems = NO;
        item.submenu = submenu;
        ns_menu_item_ = (__bridge_retained void*)item;
        ns_menu_ = (__bridge_retained void*)submenu;
        break;
      }
      case Type::kSeparator: {
        ns_menu_item_ = (__bridge_retained void*)[NSMenuItem separatorItem];
        break;
      }
      case Type::kString: {
        NSMenuItem* item = [[NSMenuItem alloc] initWithTitle:title
                                                      action:nil
                                               keyEquivalent:@""];
        // Bridge the native action to the stored C++ callback. Deliberately no
        // keyEquivalent for cross-platform items: existing HandleKeyEvent stays
        // the source of truth for shortcuts, so menu items don't double-fire or
        // steal keystrokes. The shortcut text is rendered for discoverability.
        XeniaMenuTarget* target = [[XeniaMenuTarget alloc] init];
        MacMenuItem* owner = this;
        target->action = [owner]() { owner->OnSelected(); };
        item.target = target;
        item.action = @selector(invoke:);
        if (!hotkey.empty()) {
          NSString* shortcut =
              [NSString stringWithUTF8String:hotkey.c_str()];
          NSString* full =
              [NSString stringWithFormat:@"%@\t%@", title, shortcut];
          item.attributedTitle = [[NSAttributedString alloc] initWithString:full];
        }
        target_ = (__bridge_retained void*)target;
        ns_menu_item_ = (__bridge_retained void*)item;
        break;
      }
    }
  }
}

MacMenuItem::~MacMenuItem() {
  @autoreleasepool {
    if (ns_menu_) {
      NSMenu* menu = (__bridge_transfer NSMenu*)ns_menu_;
      ns_menu_ = nullptr;
      (void)menu;
    }
    if (ns_menu_item_) {
      NSMenuItem* item = (__bridge_transfer NSMenuItem*)ns_menu_item_;
      ns_menu_item_ = nullptr;
      (void)item;
    }
    if (target_) {
      XeniaMenuTarget* target = (__bridge_transfer XeniaMenuTarget*)target_;
      target_ = nullptr;
      (void)target;
    }
  }
}

void MacMenuItem::SetEnabled(bool enabled) {
  @autoreleasepool {
    // Container (root menu / popup submenu): apply to the items shown in this
    // menu. For the root menu these are the top-level titles, matching
    // SetMainMenuEnabled.
    NSMenu* menu = (__bridge NSMenu*)ns_menu_;
    if (menu) {
      for (NSMenuItem* item in menu.itemArray) {
        item.enabled = enabled ? YES : NO;
      }
      return;
    }
    // Leaf item (e.g. a kString action): enable/disable just this NSMenuItem,
    // so individual lifecycle items can be greyed per state. Relies on the
    // owning menu's autoenablesItems=NO (set when the menu is built).
    NSMenuItem* item = (__bridge NSMenuItem*)ns_menu_item_;
    if (item) {
      item.enabled = enabled ? YES : NO;
    }
  }
}

void MacMenuItem::OnChildAdded(MenuItem* generic_child_item) {
  @autoreleasepool {
    auto child = static_cast<MacMenuItem*>(generic_child_item);
    // Container is the NSMenu of a kNormal root or a kPopup submenu.
    NSMenu* menu = (__bridge NSMenu*)ns_menu_;
    if (!menu || child->type() == Type::kNormal) {
      return;
    }
    NSMenuItem* item = (__bridge NSMenuItem*)child->ns_menu_item();
    if (item) {
      [menu addItem:item];
    }
  }
}

void MacMenuItem::OnChildRemoved(MenuItem* generic_child_item) {}

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
    // Permit native (Spaces) fullscreen so the cross-platform fullscreen toggle
    // and the green title-bar button work; see ApplyNewFullscreen.
    window.collectionBehavior =
        window.collectionBehavior | NSWindowCollectionBehaviorFullScreenPrimary;
    // Deliver mouseMoved: to the first responder so ImGui gets hover updates
    // (menu highlight, etc.), not just clicks/drags.
    [window setAcceptsMouseMovedEvents:YES];

    XeniaWindowDelegate* delegate =
        [[XeniaWindowDelegate alloc] initWithWindow:this];
    [window setDelegate:delegate];

    [window center];
    [window makeKeyAndOrderFront:nil];
    // Bring the whole app to the foreground, not just order this window within
    // it. On a self-relaunch (posix_spawn from the prior instance, used by the
    // emulation lifecycle Reboot/Load controls) the new process can launch
    // backgrounded: the window stays unfocused, macOS marks it occluded, and
    // the presenter throttles painting - so it shows black until the user clicks
    // it. Activating here, when the window is actually shown (not at run-loop
    // start before it exists), foregrounds it reliably past the relaunch focus
    // race. Harmless on a normal launch (already frontmost).
    [NSApp activateIgnoringOtherApps:YES];
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

void MacWindow::ShowAboutDialog() { ShowMacAboutPanel(); }

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

void MacWindow::ApplyNewFullscreen() {
  // Reconcile the native window state with the desired IsFullscreen() (this hook
  // means "make native match desired", so toggle only when they differ — robust
  // to redundant applies). macOS auto-hides/reveals the menu bar in native
  // fullscreen, so unlike GTK the menu needs no detaching here.
  NSWindow* window = (__bridge NSWindow*)window_;
  if (!window) {
    return;
  }
  bool native_fullscreen =
      ([window styleMask] & NSWindowStyleMaskFullScreen) != 0;
  if (native_fullscreen != IsFullscreen()) {
    [window toggleFullScreen:nil];
  }
}

void MacWindow::SetDesiredLogicalSizeImpl() {
  @autoreleasepool {
    NSWindow* window = (__bridge NSWindow*)window_;
    if (!window) {
      return;
    }
    // Never fight native (Spaces) fullscreen; the common Window also guards this,
    // but the native state is the source of truth here.
    if (([window styleMask] & NSWindowStyleMaskFullScreen) != 0) {
      return;
    }
    // Un-maximize (un-zoom) so the requested size is actually applied.
    if ([window isZoomed]) {
      [window zoom:nil];
    }
    // Clamp the requested logical (point) size to the screen's visible frame so
    // the title bar can't land off-screen on smaller displays. visibleFrame is
    // in points, matching the logical desired size.
    NSScreen* screen = [window screen];
    if (!screen) {
      screen = [NSScreen mainScreen];
    }
    NSRect visible = [screen visibleFrame];
    CGFloat width = CGFloat(GetDesiredLogicalWidth());
    CGFloat height = CGFloat(GetDesiredLogicalHeight());
    if (width > visible.size.width) {
      width = visible.size.width;
    }
    if (height > visible.size.height) {
      height = visible.size.height;
    }
    // Store the (possibly clamped) logical size back as the desired size.
    OnDesiredLogicalSizeUpdate(uint32_t(width), uint32_t(height));
    // setContentSize: delivers a delegate resize (HandleResize); flag it so the
    // feedback isn't treated as a user resize. Re-center for a tidy placement.
    applying_desired_size_ = true;
    [window setContentSize:NSMakeSize(width, height)];
    [window center];
    UpdateSizeFromView();
    applying_desired_size_ = false;
  }
}

void MacWindow::ApplyNewMainMenu(MenuItem* old_main_menu) {
  // The macOS menu bar is app-global (NSApp.mainMenu); applied here from the
  // single emulator window that owns the menu. Runs on the UI thread, which is
  // the main thread on macOS (where AppKit requires menu mutation).
  @autoreleasepool {
    auto* main_menu = static_cast<MacMenuItem*>(GetMainMenu());
    NSMenu* bar = main_menu ? (__bridge NSMenu*)main_menu->ns_menu() : nil;
    if (!bar) {
      [NSApp setMainMenu:[[NSMenu alloc] initWithTitle:@""]];
      return;
    }
    // Prepend the standard application menu once. Tagged so repeated applies
    // (e.g. via CompleteMainMenuItemsUpdate) don't insert duplicates.
    static const NSInteger kAppMenuTag = 0x58454E49;  // 'XENI'
    if (bar.numberOfItems == 0 || [bar itemAtIndex:0].tag != kAppMenuTag) {
      NSMenuItem* app_item =
          CreateApplicationMenuItem((__bridge XeniaWindowDelegate*)delegate_);
      app_item.tag = kAppMenuTag;
      [bar insertItem:app_item atIndex:0];
    }
    [NSApp setMainMenu:bar];
  }
}

void MacWindow::CompleteMainMenuItemsUpdateImpl() {
  // Item state (enabled flags, titles) is reflected live on the NSMenu objects,
  // so no rebuild is required; ensure the bar is installed.
  ApplyNewMainMenu(GetMainMenu());
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
  // is kManual (matching window_gtk.cc / window_win.cc) and persists the size to
  // window_size_x/y. A programmatic auto-size (SetDesiredLogicalSizeImpl) is not
  // a user action, so report it as non-manual to avoid clobbering those cvars.
  WindowResizeAction action = applying_desired_size_
                                  ? WindowResizeAction::kAutoMaximize
                                  : WindowResizeAction::kManual;
  OnActualSizeUpdate(physical_width, physical_height, action,
                     destruction_receiver);
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
