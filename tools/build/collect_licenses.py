#!/usr/bin/env python3
"""Collect macXenos's third-party notices into acknowledgements.json.

Reads docs/acknowledgements/manifest.json and writes one JSON file the
Preferences page shows under System > About > Acknowledgements:

  {"components": [{"name", "license", "url", "note", "text"}, ...]}

usage:
  collect_licenses.py --out <file> [--strict] [--link-ninja <impl-*.ninja>]
                      [--sparkle-dir <dir>] [--homebrew-prefix <dir>]
                      [--bundled <name> ...]

--strict          A license file that can't be read is an error (packaging).
                  Without it, such a component is left out with a warning
                  (development builds may lack Sparkle or Homebrew's copies).
--link-ninja      The CMake impl-<config>.ninja holding macXenos's link
                  line: every third-party static library or dylib it links
                  must be named by some entry's "libs", or this fails.
--bundled         Further shipped binaries to check the same way
                  (libMoltenVK.dylib, Sparkle.framework, ...).
"""

import argparse
import json
import os
import re
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MANIFEST = os.path.join(REPO_ROOT, "docs", "acknowledgements", "manifest.json")


def read_source(spec, args):
    if "path" in spec:
        path = os.path.join(REPO_ROOT, spec["path"])
    elif "homebrew" in spec:
        path = os.path.join(args.homebrew_prefix, "opt", spec["homebrew"])
    elif "sparkle" in spec:
        if not args.sparkle_dir:
            raise FileNotFoundError("no --sparkle-dir for " + spec["sparkle"])
        path = os.path.join(args.sparkle_dir, spec["sparkle"])
    else:
        raise ValueError("file entry without path/homebrew/sparkle: %r" % spec)
    if spec.get("rtf"):
        text = subprocess.run(["textutil", "-convert", "txt", "-stdout", path],
                              check=True, capture_output=True, text=True).stdout
    else:
        with open(path, encoding="utf-8", errors="replace") as f:
            text = f.read()
    if "lines" in spec:
        first, last = spec["lines"]
        text = "\n".join(text.splitlines()[first - 1:last])
    return text.strip("\n")


def linked_third_party(ninja_path):
    """Third-party libraries on macXenos's link line."""
    with open(ninja_path, encoding="utf-8", errors="replace") as f:
        lines = f.read().splitlines()
    for i, line in enumerate(lines):
        if re.match(r"^build \S*/macXenos(\.\S+)?:", line):
            for follow in lines[i + 1:i + 40]:
                if "LINK_LIBRARIES" in follow:
                    names = set()
                    for token in follow.split():
                        base = os.path.basename(token)
                        if base.endswith((".a", ".dylib")) and not base.startswith("libxenia-"):
                            names.add(base)
                    return names
    raise RuntimeError("no macXenos link line in " + ninja_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--link-ninja")
    parser.add_argument("--sparkle-dir")
    parser.add_argument("--homebrew-prefix", default="/opt/homebrew")
    parser.add_argument("--bundled", nargs="*", default=[])
    args = parser.parse_args()

    with open(MANIFEST, encoding="utf-8") as f:
        manifest = json.load(f)

    problems = []
    accounted = set()
    out = []
    for component in manifest["components"]:
        accounted.update(component.get("libs", []))
        texts = []
        try:
            for spec in component["files"]:
                texts.append(read_source(spec, args))
        except (OSError, subprocess.CalledProcessError) as error:
            message = "%s: %s" % (component["name"], error)
            if args.strict:
                problems.append(message)
            else:
                print("collect_licenses: leaving out " + message, file=sys.stderr)
            continue
        out.append({
            "name": component["name"],
            "license": component["license"],
            "url": component.get("url", ""),
            "note": component.get("note", ""),
            "text": "\n\n".join(texts),
        })

    shipped = set(args.bundled)
    if args.link_ninja:
        shipped |= linked_third_party(args.link_ninja)
    for name in sorted(shipped - accounted):
        problems.append("%s is shipped but no manifest entry names it in 'libs'" % name)

    if problems:
        for problem in problems:
            print("collect_licenses: error: " + problem, file=sys.stderr)
        return 1

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"components": out}, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print("collect_licenses: %d components -> %s" % (len(out), args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
