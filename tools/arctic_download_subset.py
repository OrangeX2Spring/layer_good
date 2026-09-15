"""Download three KV-Tracker S01 sequences and GT from the official ARCTIC server.

Run interactively inside a CAMP data allocation. Uses only Python's standard
library. Prompts for ARCTIC credentials; never saves them. Archives remain packed.
"""

import getpass
import hashlib
import http.cookiejar
import json
import os
from pathlib import Path
import shutil
import socket
import urllib.parse
import urllib.request


if socket.gethostname().split(".")[0] == "head" or "SLURM_JOB_ID" not in os.environ:
    raise SystemExit("Run inside a Slurm compute allocation, not on head.")

dest = Path("/mnt/projects/gr/3DRecon/kvt_object_data/arctic_s01_pilot")
stage = Path("/tmp/kvt_arctic_download")
stage.mkdir(exist_ok=True)
dest.mkdir(parents=True, exist_ok=True)
base = "https://raw.githubusercontent.com/zc-alexfan/arctic/master/bash/assets/"
metadata = {}
for name in ("urls/cropped_images.txt", "urls/misc.txt", "checksum.json"):
    with urllib.request.urlopen(base + name, timeout=60) as response:
        metadata[name] = response.read().decode()
    (dest / Path(name).name).write_text(metadata[name])
hashes = json.loads(metadata["checksum.json"])
keys = ["/data/raw_seqs.zip", "/data/meta.zip"] + [
    f"/data/cropped_images_zips/s01/{name}_grab_01.zip"
    for name in ("box", "ketchup", "espressomachine")
]
urls = (metadata["urls/cropped_images.txt"] + "\n" + metadata["urls/misc.txt"]).splitlines()
selected = []
for key in keys:
    matches = [url for url in urls if url.endswith(key)]
    assert len(matches) == 1, (key, len(matches))
    selected.append((key, matches[0], hashes[key]))
(dest / "manifest.json").write_text(json.dumps(selected, indent=2) + "\n")

credentials = urllib.parse.urlencode({
    "username": getpass.getpass("ARCTIC email (hidden): ").strip(),
    "password": getpass.getpass("ARCTIC password: "),
}).encode()
# Preserve session cookies across the download server's login redirects.
opener = urllib.request.build_opener(
    urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar())
)
total = 0
for key, url, expected in selected:
    name = Path(key).name
    output = dest / name
    if output.exists():
        digest = hashlib.sha256()
        with output.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
        assert digest.hexdigest() == expected, f"Existing file checksum mismatch: {name}"
        print("ALREADY VERIFIED", name, flush=True)
        continue
    request = urllib.request.Request(url, data=credentials)
    temporary = stage / (name + ".part")
    digest = hashlib.sha256()
    size = 0
    with opener.open(request, timeout=120) as response:
        endpoint = urllib.parse.urlsplit(response.url)
        content_type = response.headers.get_content_type()
        print("HTTP", response.status, content_type,
              endpoint.hostname, endpoint.path, flush=True)
        if content_type in ("text/html", "application/xhtml+xml"):
            raise SystemExit(
                "Server returned a web page instead of the ZIP. "
                "Download authentication/access is unresolved; no archive was saved."
            )
        with temporary.open("wb") as target:
            print("DOWNLOADING", name, "bytes=", response.headers.get("Content-Length", "unknown"), flush=True)
            while chunk := response.read(1024 * 1024):
                total += len(chunk)
                size += len(chunk)
                if total > 10 * 1024**3:
                    raise SystemExit("Pilot transfer exceeded 10 GiB; review archive sizes before continuing.")
                digest.update(chunk)
                target.write(chunk)
                if size % (32 * 1024**2) == 0:
                    print(name, size // 1024**2, "MiB received", flush=True)
    assert digest.hexdigest() == expected, f"Download checksum mismatch: {name}"
    persistent_partial = dest / (name + ".part")
    shutil.copyfile(temporary, persistent_partial)
    persistent_partial.replace(output)
    print("VERIFIED", name, size, "bytes", flush=True)
print("DOWNLOAD OK:", dest)
