/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Xenia Canary. All rights reserved.                          *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/app/blades_overlay_mac.h"

#import <AppKit/AppKit.h>
#import <WebKit/WebKit.h>

#include <algorithm>
#include <string>

#include "xenia/base/filesystem.h"
#include "xenia/base/logging.h"

namespace {

// The scheme the page is served from. Its own scheme (rather than file://)
// lets the page's absolute paths and fetch() of the sounds work as on the web.
NSString* const kScheme = @"xenia-blades";

NSString* MimeTypeFor(NSString* extension) {
  static NSDictionary<NSString*, NSString*>* types = @{
    @"html" : @"text/html",
    @"js" : @"text/javascript",
    @"css" : @"text/css",
    @"json" : @"application/json",
    @"woff2" : @"font/woff2",
    @"m4a" : @"audio/mp4",
    @"webp" : @"image/webp",
    @"png" : @"image/png",
    @"svg" : @"image/svg+xml",
  };
  return types[extension.lowercaseString] ?: @"application/octet-stream";
}

}  // namespace

// Serves files from the page root, and nothing outside it.
@interface XeniaBladesSchemeHandler : NSObject <WKURLSchemeHandler>
- (instancetype)initWithRoot:(NSString*)root
                    iconRoot:(NSString*)icon_root
                 contentRoot:(NSString*)content_root
                    dataRoot:(NSString*)data_root;
@end

@implementation XeniaBladesSchemeHandler {
  NSString* root_;
  NSString* icon_root_;
  NSString* content_root_;
  NSString* data_root_;
}
- (instancetype)initWithRoot:(NSString*)root
                    iconRoot:(NSString*)icon_root
                 contentRoot:(NSString*)content_root
                    dataRoot:(NSString*)data_root {
  self = [super init];
  if (self) {
    root_ = root.stringByStandardizingPath;
    icon_root_ = icon_root.stringByStandardizingPath;
    content_root_ = content_root.stringByStandardizingPath;
    data_root_ = data_root.stringByStandardizingPath;
  }
  return self;
}
- (void)webView:(WKWebView*)webView
    startURLSchemeTask:(id<WKURLSchemeTask>)task {
  NSString* path = task.request.URL.path;
  if (path.length == 0 || [path isEqualToString:@"/"]) {
    path = @"/index.html";
  }
  // /title-icons/ID.png comes from the games' icon store; /content/...png
  // from the content folder, PNGs only (profiles' pictures, downloaded gamer
  // pictures; never saves); /data/gamer_pictures.json is the picker's index;
  // the rest is the page itself. Nothing outside any of them.
  NSString* base = root_;
  if ([path hasPrefix:@"/title-icons/"] && icon_root_.length) {
    base = icon_root_;
    path = [path substringFromIndex:[@"/title-icons" length]];
  } else if ([path hasPrefix:@"/content/"] && content_root_.length &&
             [path.pathExtension.lowercaseString isEqualToString:@"png"]) {
    base = content_root_;
    path = [path substringFromIndex:[@"/content" length]];
  } else if ([path isEqualToString:@"/data/gamer_pictures.json"] &&
             data_root_.length) {
    base = data_root_;
    path = @"/gamer_pictures.json";
  }
  NSString* full =
      [base stringByAppendingPathComponent:path].stringByStandardizingPath;
  NSData* data = nil;
  if ([full hasPrefix:[base stringByAppendingString:@"/"]]) {
    data = [NSData dataWithContentsOfFile:full];
  }
  NSInteger status = data ? 200 : 404;
  NSDictionary* headers = @{
    @"Content-Type" : MimeTypeFor(full.pathExtension),
    @"Content-Length" : [NSString stringWithFormat:@"%lu", (unsigned long)data.length],
    @"Cache-Control" : @"no-cache",
  };
  NSHTTPURLResponse* response =
      [[NSHTTPURLResponse alloc] initWithURL:task.request.URL
                                  statusCode:status
                                 HTTPVersion:@"HTTP/1.1"
                                headerFields:headers];
  [task didReceiveResponse:response];
  if (data) {
    [task didReceiveData:data];
  }
  [task didFinish];
}
- (void)webView:(WKWebView*)webView
    stopURLSchemeTask:(id<WKURLSchemeTask>)task {
}
@end

// Owns the web view and hands the page's messages to C++. A separate object
// from the web view's configuration's handler so the retain cycle through
// WKUserContentController can be broken on teardown.
@interface XeniaBladesHost : NSObject <WKScriptMessageHandler, WKNavigationDelegate> {
 @public
  xe::app::BladesOverlay::Callbacks callbacks;
  WKWebView* web_view;
  NSView* container;
  NSView* game_view;
  NSTimer* timer;
  BOOL loaded;
  // The dashboard sounds, played here rather than by the page's Web Audio so
  // they don't depend on WebKit's autoplay and audio-session rules. They are
  // downloaded on request (dashboard_sounds_mac.h), never shipped.
  NSString* sound_root;
  NSMutableDictionary<NSString*, NSSound*>* sounds;
}
- (void)playSound:(NSString*)file gain:(double)gain;
@end

@implementation XeniaBladesHost
- (void)playSound:(NSString*)file gain:(double)gain {
  // Only the installed sounds' names: letters, digits and underscores.
  NSCharacterSet* allowed = [NSCharacterSet
      characterSetWithCharactersInString:
          @"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"];
  if (file.length == 0 || file.length > 64 ||
      [file rangeOfCharacterFromSet:allowed.invertedSet].location != NSNotFound) {
    return;
  }
  if (!sounds) {
    sounds = [NSMutableDictionary dictionary];
  }
  NSSound* sound = sounds[file];
  if (!sound) {
    NSString* path = [[sound_root stringByAppendingPathComponent:file]
        stringByAppendingPathExtension:@"wav"];
    if (![NSFileManager.defaultManager fileExistsAtPath:path]) {
      return;  // Not downloaded (yet): the dashboard is silent.
    }
    sound = [[NSSound alloc] initWithContentsOfFile:path byReference:YES];
    if (!sound) {
      XELOGW("Blades: dashboard sound {} did not load", file.UTF8String);
      return;
    }
    sounds[file] = sound;
  }
  // A quick run of focus ticks restarts the same sound, as the dashboard did.
  [sound stop];
  sound.volume = float(std::clamp(gain, 0.0, 1.0));
  [sound play];
}
- (void)userContentController:(WKUserContentController*)controller
      didReceiveScriptMessage:(WKScriptMessage*)message {
  if ([message.name isEqualToString:@"xeniaLog"]) {
    // The page's uncaught errors and console.error, into the emulator log.
    NSString* text = [message.body description];
    XELOGW("Blades page: {}", text.UTF8String ? text.UTF8String : "?");
    return;
  }
  if ([message.body isKindOfClass:[NSDictionary class]] &&
      [message.body[@"op"] isEqual:@"sound"]) {
    id file = message.body[@"file"];
    id gain = message.body[@"gain"];
    if ([file isKindOfClass:[NSString class]] &&
        [gain isKindOfClass:[NSNumber class]]) {
      [self playSound:file gain:[gain doubleValue]];
    }
    return;
  }
  if (!callbacks.on_message || ![NSJSONSerialization isValidJSONObject:message.body]) {
    return;
  }
  NSData* data = [NSJSONSerialization dataWithJSONObject:message.body
                                                 options:0
                                                   error:nil];
  if (!data) {
    return;
  }
  std::string json(static_cast<const char*>(data.bytes), data.length);
  // Copied: the handler may tear the overlay down (close, relaunch).
  auto on_message = callbacks.on_message;
  on_message(json);
}
- (void)tick:(NSTimer*)sender {
  if (callbacks.on_tick) {
    auto on_tick = callbacks.on_tick;
    on_tick();
  }
}
// The page lives on its own xenia-blades:// scheme; a link out of it (the
// website on About macXenos) opens in the Mac's browser rather than
// replacing the dashboard.
- (void)webView:(WKWebView*)webView
    decidePolicyForNavigationAction:(WKNavigationAction*)action
                    decisionHandler:
                        (void (^)(WKNavigationActionPolicy))decisionHandler {
  NSURL* url = action.request.URL;
  NSString* scheme = url.scheme.lowercaseString;
  if ([scheme isEqualToString:@"http"] || [scheme isEqualToString:@"https"] ||
      [scheme isEqualToString:@"mailto"]) {
    [[NSWorkspace sharedWorkspace] openURL:url];
    decisionHandler(WKNavigationActionPolicyCancel);
    return;
  }
  decisionHandler(WKNavigationActionPolicyAllow);
}
- (void)webView:(WKWebView*)webView
    didFinishNavigation:(WKNavigation*)navigation {
  XELOGI("Blades: Preferences page loaded");
}
- (void)webViewWebContentProcessDidTerminate:(WKWebView*)webView {
  XELOGE("Blades: the Preferences page's web process ended; reloading");
  [webView reload];
}
- (void)webView:(WKWebView*)webView
    didFailNavigation:(WKNavigation*)navigation
            withError:(NSError*)error {
  XELOGE("Blades: the Preferences page failed: {}",
         error.localizedDescription.UTF8String);
}
- (void)webView:(WKWebView*)webView
    didFailProvisionalNavigation:(WKNavigation*)navigation
                       withError:(NSError*)error {
  XELOGE("Blades: the Preferences page failed to load: {}",
         error.localizedDescription.UTF8String);
}
@end

namespace xe {
namespace app {

std::vector<std::string> PreferredLanguages() {
  std::vector<std::string> tags;
  @autoreleasepool {
    for (NSString* tag in [NSLocale preferredLanguages]) {
      tags.emplace_back(tag.UTF8String);
    }
  }
  return tags;
}

std::filesystem::path BladesOverlay::FindPageRoot() {
  const std::filesystem::path exe = xe::filesystem::GetExecutableFolder();
  for (const std::filesystem::path& candidate :
       {exe / "blades", exe / ".." / "Resources" / "blades"}) {
    std::error_code ec;
    if (std::filesystem::exists(candidate / "index.html", ec)) {
      return std::filesystem::weakly_canonical(candidate, ec);
    }
  }
  return {};
}

BladesOverlay::BladesOverlay(void* container_view, void* game_view,
                             const std::filesystem::path& title_icon_root,
                             const std::filesystem::path& sound_root,
                             const std::filesystem::path& content_root,
                             const std::filesystem::path& data_root,
                             Callbacks callbacks) {
  @autoreleasepool {
    XeniaBladesHost* host = [[XeniaBladesHost alloc] init];
    host->callbacks = std::move(callbacks);
    host->container = (__bridge NSView*)container_view;
    host->game_view = (__bridge NSView*)game_view;

    std::filesystem::path root = FindPageRoot();
    if (root.empty()) {
      XELOGE(
          "Blades: no Preferences page beside the binary (blades/index.html); "
          "build src/xenia/app/blades_ui");
    }

    WKWebViewConfiguration* config = [[WKWebViewConfiguration alloc] init];
    // Persistent, so the page's own settings (theme, sounds) survive.
    config.websiteDataStore = [WKWebsiteDataStore defaultDataStore];
    [config setURLSchemeHandler:
                [[XeniaBladesSchemeHandler alloc]
                    initWithRoot:[NSString stringWithUTF8String:
                                               xe::path_to_utf8(root).c_str()]
                        iconRoot:[NSString
                                     stringWithUTF8String:
                                         xe::path_to_utf8(title_icon_root)
                                             .c_str()]
                     contentRoot:[NSString
                                     stringWithUTF8String:
                                         xe::path_to_utf8(content_root).c_str()]
                        dataRoot:[NSString
                                     stringWithUTF8String:
                                         xe::path_to_utf8(data_root).c_str()]]
                   forURLScheme:kScheme];
    [config.userContentController addScriptMessageHandler:host name:@"xenia"];
    [config.userContentController addScriptMessageHandler:host name:@"xeniaLog"];
    // Uncaught errors and console.error reach the emulator log (xeniaLog).
    NSString* log_script =
        @"(function(){var h=window.webkit.messageHandlers.xeniaLog;"
         "function s(a){try{return Array.prototype.map.call(a,function(x){"
         "return x&&x.stack?x.stack:String(x)}).join(' ')}catch(e){return '?'}}"
         "window.addEventListener('error',function(e){h.postMessage("
         "'error: '+e.message+' at '+e.filename+':'+e.lineno)});"
         "window.addEventListener('unhandledrejection',function(e){"
         "h.postMessage('rejection: '+s([e.reason]))});"
         "var ce=console.error;console.error=function(){h.postMessage("
         "'console.error: '+s(arguments));ce.apply(console,arguments)};})();";
    [config.userContentController
        addUserScript:[[WKUserScript alloc]
                          initWithSource:log_script
                           injectionTime:WKUserScriptInjectionTimeAtDocumentStart
                        forMainFrameOnly:YES]];
    // Sounds play when a key is pressed, without a click first.
    config.mediaTypesRequiringUserActionForPlayback = WKAudiovisualMediaTypeNone;

    NSView* container = host->container;
    WKWebView* web_view = [[WKWebView alloc] initWithFrame:container.bounds
                                             configuration:config];
    web_view.autoresizingMask = NSViewWidthSizable | NSViewHeightSizable;
    web_view.navigationDelegate = host;
    web_view.hidden = YES;
    web_view.underPageBackgroundColor = NSColor.blackColor;
    if (@available(macOS 13.3, *)) {
      // Safari's Develop menu can inspect the page.
      web_view.inspectable = YES;
    }
    [container addSubview:web_view
               positioned:NSWindowAbove
               relativeTo:host->game_view];
    host->web_view = web_view;
    host->sound_root = [NSString
        stringWithUTF8String:xe::path_to_utf8(sound_root).c_str()];

    impl_ = (__bridge_retained void*)host;
  }
}

BladesOverlay::~BladesOverlay() {
  @autoreleasepool {
    XeniaBladesHost* host = (__bridge_transfer XeniaBladesHost*)impl_;
    impl_ = nullptr;
    [host->timer invalidate];
    host->timer = nil;
    [host->web_view.configuration.userContentController
        removeScriptMessageHandlerForName:@"xenia"];
    [host->web_view.configuration.userContentController
        removeScriptMessageHandlerForName:@"xeniaLog"];
    [host->web_view removeFromSuperview];
    host->web_view = nil;
    host->callbacks = {};
  }
}

void BladesOverlay::Show(const std::string& blade_id) {
  XeniaBladesHost* host = (__bridge XeniaBladesHost*)impl_;
  XELOGI("Blades: showing Preferences ({})",
         blade_id.empty() ? "last blade" : blade_id);
  NSString* blade = [NSString stringWithUTF8String:blade_id.c_str()];
  if (!host->loaded) {
    host->loaded = YES;
    NSString* url = [NSString
        stringWithFormat:@"%@://app/index.html%@%@", kScheme,
                         blade.length ? @"#" : @"", blade];
    [host->web_view loadRequest:[NSURLRequest requestWithURL:[NSURL URLWithString:url]]];
  } else if (blade.length) {
    NSString* script = [NSString
        stringWithFormat:@"if (location.hash !== '#%@') location.hash = '#%@';",
                         blade, blade];
    [host->web_view evaluateJavaScript:script completionHandler:nil];
  }
  host->web_view.hidden = NO;
  [host->web_view.window makeFirstResponder:host->web_view];
  // Each opening starts at the top of the blade, cursor and focus together.
  [host->web_view evaluateJavaScript:@"window.__xeniaShown && window.__xeniaShown()"
                   completionHandler:nil];
  if (!host->timer) {
    host->timer = [NSTimer timerWithTimeInterval:1.0 / 30.0
                                          target:host
                                        selector:@selector(tick:)
                                        userInfo:nil
                                         repeats:YES];
    [[NSRunLoop mainRunLoop] addTimer:host->timer forMode:NSRunLoopCommonModes];
  }
  visible_ = true;
}

void BladesOverlay::Hide() {
  XeniaBladesHost* host = (__bridge XeniaBladesHost*)impl_;
  [host->timer invalidate];
  host->timer = nil;
  host->web_view.hidden = YES;
  [host->web_view.window makeFirstResponder:host->game_view];
  visible_ = false;
}

void BladesOverlay::PushState(const std::string& json) {
  XeniaBladesHost* host = (__bridge XeniaBladesHost*)impl_;
  if (!host->loaded) {
    return;
  }
  // The state is a JSON object, which is also a JavaScript expression; the
  // two line separators JSON allows raw but JavaScript does not are escaped.
  std::string safe;
  safe.reserve(json.size() + 64);
  for (size_t i = 0; i < json.size(); ++i) {
    if (i + 2 < json.size() && uint8_t(json[i]) == 0xE2 &&
        uint8_t(json[i + 1]) == 0x80 &&
        (uint8_t(json[i + 2]) == 0xA8 || uint8_t(json[i + 2]) == 0xA9)) {
      safe += uint8_t(json[i + 2]) == 0xA8 ? "\\u2028" : "\\u2029";
      i += 2;
    } else {
      safe.push_back(json[i]);
    }
  }
  std::string script =
      "window.__xeniaReceive && window.__xeniaReceive(JSON.stringify(" + safe +
      "));";
  NSString* source = [[NSString alloc] initWithBytes:script.data()
                                              length:script.size()
                                            encoding:NSUTF8StringEncoding];
  if (!source) {
    XELOGE("Blades: the state is not valid UTF-8 ({} bytes); not sent",
           script.size());
    return;
  }
  [host->web_view evaluateJavaScript:source
                   completionHandler:^(id result, NSError* error) {
                     if (error) {
                       XELOGE("Blades: the page rejected the state: {}",
                              error.localizedDescription.UTF8String);
                     }
                   }];
}

void BladesOverlay::PadButton(const char* button, bool down) {
  XeniaBladesHost* host = (__bridge XeniaBladesHost*)impl_;
  if (!host->loaded || !visible_) {
    return;
  }
  NSString* script = [NSString
      stringWithFormat:@"window.__xeniaPad && window.__xeniaPad('%s', %s);",
                       button, down ? "true" : "false"];
  [host->web_view evaluateJavaScript:script completionHandler:nil];
}

}  // namespace app
}  // namespace xe
