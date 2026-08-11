#!/usr/bin/env python3
"""Patch TensorBoard's bundled webfiles.zip to remove the forced
overflow:auto on the image-card component's host element (and its
.img-container), so you don't have to disable it in devtools every
session.
"""
import os
import shutil
import zipfile
import tempfile
import sys

import tensorboard

TB_DIR = os.path.dirname(tensorboard.__file__)
WEBFILES = os.path.join(TB_DIR, "webfiles.zip")
BACKUP = WEBFILES + ".orig"

# (old, new, expected_count) - each patched independently
PATCHES = [
    (
        b"[_nghost-%COMP%]{box-sizing:border-box;display:flex;flex-basis:318px;flex-direction:column;flex-grow:1;height:100%;overflow:auto;padding:16px;padding-top:4px}",
        b"[_nghost-%COMP%]{box-sizing:border-box;display:flex;flex-basis:318px;flex-direction:column;flex-grow:1;height:100%;overflow:visible;padding:16px;padding-top:4px}",
        2,
    ),
    (
        b".img-container[_ngcontent-%COMP%]{flex-grow:1;overflow-y:auto;position:relative}",
        b".img-container[_ngcontent-%COMP%]{flex-grow:1;overflow-y:visible;position:relative}",
        1,
    ),
]

def main():
    if not os.path.exists(WEBFILES):
        sys.exit(f"Could not find {WEBFILES}")

    if not os.path.exists(BACKUP):
        shutil.copy2(WEBFILES, BACKUP)
        print(f"Backed up original to {BACKUP}")
    else:
        print(f"Backup already exists at {BACKUP} (not overwriting)")

    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(WEBFILES, "r") as zin:
            names = zin.namelist()
            zin.extractall(tmp)

        index_path = os.path.join(tmp, "index.js")
        if not os.path.exists(index_path):
            sys.exit("index.js not found inside webfiles.zip - TensorBoard version may differ, aborting.")

        with open(index_path, "rb") as f:
            data = f.read()

        any_applied = False
        for old, new, expected in PATCHES:
            count = data.count(old)
            if count == 0:
                print(f"SKIP: pattern not found (expected {expected}) - {old[:60]}...")
                continue
            if count != expected:
                print(f"WARNING: found {count} occurrences (expected {expected}) - applying to all anyway - {old[:60]}...")
            data = data.replace(old, new)
            any_applied = True
            print(f"Patched {count} occurrence(s) of: {old[:60]}...")

        if not any_applied:
            sys.exit("No known patterns matched - TensorBoard version differs from expected, aborting without changes.")

        with open(index_path, "wb") as f:
            f.write(data)

        new_zip_path = WEBFILES + ".new"
        with zipfile.ZipFile(new_zip_path, "w", zipfile.ZIP_DEFLATED) as zout:
            for name in names:
                zout.write(os.path.join(tmp, name), name)

        os.replace(new_zip_path, WEBFILES)

    print("Patched successfully. Fully kill and restart the TensorBoard server, then hard-refresh (or use a private window / disable cache) in the browser.")
    print(f"To revert: cp {BACKUP} {WEBFILES}")

if __name__ == "__main__":
    main()