// usage: mkicon <src.png> <out.png>  -> 1024x1024 macOS app icon: art scaled
// onto the 824pt Big Sur squircle grid with a continuous-corner mask.
import AppKit
let args = CommandLine.arguments
guard let src = NSImage(contentsOfFile: args[1]) else { print("no src"); exit(1) }
let size = 1024.0, art = 824.0, inset = (size - art) / 2, radius = art * 0.2237
let canvas = NSImage(size: NSSize(width: size, height: size))
canvas.lockFocus()
NSColor.clear.set(); NSRect(x: 0, y: 0, width: size, height: size).fill()
let rect = NSRect(x: inset, y: inset, width: art, height: art)
let path = NSBezierPath(roundedRect: rect, xRadius: radius, yRadius: radius)
NSGraphicsContext.current?.imageInterpolation = .high
path.addClip()
src.draw(in: rect, from: .zero, operation: .sourceOver, fraction: 1.0)
canvas.unlockFocus()
guard let tiff = canvas.tiffRepresentation, let rep = NSBitmapImageRep(data: tiff) else { exit(1) }
rep.size = NSSize(width: size, height: size)
let png = rep.representation(using: .png, properties: [:])!
try! png.write(to: URL(fileURLWithPath: args[2]))
print("wrote \(args[2]) \(rep.pixelsWide)x\(rep.pixelsHigh)")
