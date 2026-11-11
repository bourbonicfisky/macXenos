# macXenos app icon

Original artwork by Greg Gant. Reuse requires credit to Greg Gant as its
creator (see the README's License section).

`macxenos_ring_dark_2048.png` / `macxenos_ring_light_2048.png` are the source
artwork (full-bleed squares). `src/xenia/app/xenia.icns`, which
`tools/build/package_mac.sh` bundles, is generated from the dark variant:

    swiftc -O -o /tmp/mkicon tools/build/make_mac_icon.swift
    /tmp/mkicon assets/mac-icon/macxenos_ring_dark_2048.png assets/mac-icon/macxenos_icon_1024.png
    mkdir macxenos.iconset
    for sz in 16 32 128 256 512; do
      sips -z $sz $sz assets/mac-icon/macxenos_icon_1024.png --out macxenos.iconset/icon_${sz}x${sz}.png
      sips -z $((sz*2)) $((sz*2)) assets/mac-icon/macxenos_icon_1024.png --out macxenos.iconset/icon_${sz}x${sz}@2x.png
    done
    iconutil -c icns macxenos.iconset -o src/xenia/app/xenia.icns

`make_mac_icon.swift` places the art on the 824/1024 macOS icon grid behind a
continuous-corner mask so it sits level with other Dock icons.
