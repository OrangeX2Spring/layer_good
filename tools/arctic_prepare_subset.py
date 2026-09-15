"""Stage the downloaded S01 pilot and export KV-Tracker's initialization frames.

Run with system python3 in a CAMP data allocation. No model or third-party imports.
The prepared dataset remains one TAR on project storage; unpack it in job-local
/tmp for inference. Initial masks are deliberately left for SAM 2 and user review.
"""

import ast
import io
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import socket
import struct
import tarfile
import zipfile


if socket.gethostname().split(".")[0] == "head" or "SLURM_JOB_ID" not in os.environ:
    raise SystemExit("Run inside a Slurm compute allocation, not on head.")

source = Path("/mnt/projects/gr/3DRecon/kvt_object_data/arctic_s01_pilot")
output = Path("/mnt/projects/gr/3DRecon/kvt_arctic_out")
preview = output / "initial_frames"
preview.mkdir(parents=True, exist_ok=True)
scenes = ("box_grab_01", "ketchup_grab_01", "espressomachine_grab_01")
# Member paths are the layout run_dataset.py and eval.py expect, so the TAR
# extracts straight into datasets/arctic_data without any remapping.
IMAGES = "data/cropped_images_grab_only_subset/s01"
manifest = {
    "subject": "s01",
    "view": "0",
    "loader_offset": 2,
    "source": str(source),
    "source_download_manifest": json.loads((source / "manifest.json").read_text()),
    "alignment_check": "Image IDs and object GT rows; camera arrays checked at evaluation",
    "scenes": {},
}

with zipfile.ZipFile(source / "meta.zip") as archive:
    names = [n for n in archive.namelist() if PurePosixPath(n).name == "misc.json"]
    assert len(names) == 1, ("Expected one misc.json", names)
    misc_bytes = archive.read(names[0])
    ioi_offset = json.loads(misc_bytes)["s01"]["ioi_offset"]
manifest["ioi_offset"] = ioi_offset

partial = output / "prepared.tar.part"
with tarfile.open(partial, "w") as prepared, zipfile.ZipFile(source / "raw_seqs.zip") as raw:
    info = tarfile.TarInfo("data/meta/misc.json")
    info.size = len(misc_bytes)
    prepared.addfile(info, io.BytesIO(misc_bytes))
    for scene in scenes:
        gt_names = {}
        for suffix in ("object.npy", "egocam.dist.npy"):
            matches = [n for n in raw.namelist()
                       if PurePosixPath(n).parent.name == "s01"
                       and PurePosixPath(n).name == f"{scene}.{suffix}"]
            assert len(matches) == 1, (scene, suffix, matches)
            gt_names[suffix] = matches[0]

        # Read just the official NPY header, without importing NumPy or unpickling.
        with raw.open(gt_names["object.npy"]) as gt:
            assert gt.read(6) == b"\x93NUMPY", "Invalid object GT NPY header"
            version = tuple(gt.read(2))
            assert version in ((1, 0), (2, 0)), ("Unexpected NPY version", version)
            size_format = "<H" if version == (1, 0) else "<I"
            header_size = struct.unpack(size_format, gt.read(struct.calcsize(size_format)))[0]
            header = ast.literal_eval(gt.read(header_size).decode("latin1"))
        assert len(header["shape"]) == 2 and header["shape"][1] == 7, header
        gt_count = header["shape"][0]

        with zipfile.ZipFile(source / f"{scene}.zip") as images:
            names = sorted(n for n in images.namelist()
                           if PurePosixPath(n).parent.name == "0"
                           and PurePosixPath(n).suffix == ".jpg")
            assert len(names) > 2, (scene, "No usable camera-0 images")
            ids = [int(PurePosixPath(n).stem) for n in names]
            assert len(names) == gt_count, (scene, "Image/GT count mismatch", len(names), gt_count)
            assert ids == list(range(ioi_offset, ioi_offset + gt_count)), (
                scene, "Image IDs do not map to all GT rows; do not trim silently",
                ids[:3], ids[-3:], ioi_offset, gt_count,
            )
            frames = []
            for index, name in enumerate(names):
                relative = f"{IMAGES}/{scene}/0/{PurePosixPath(name).name}"
                info = tarfile.TarInfo(relative)
                info.size = images.getinfo(name).file_size
                with images.open(name) as image_file:
                    prepared.addfile(info, image_file)
                frames.append({"path": relative, "image_id": ids[index], "gt_row": index})
            initial = preview / f"{scene}.jpg"
            with images.open(names[2]) as image_file, initial.open("wb") as target:
                shutil.copyfileobj(image_file, target)

        for suffix, name in gt_names.items():
            info = tarfile.TarInfo(f"data/raw_seqs/s01/{scene}.{suffix}")
            info.size = raw.getinfo(name).file_size
            with raw.open(name) as gt:
                prepared.addfile(info, gt)
        manifest["scenes"][scene] = {
            "frames_before_offset": gt_count,
            "tracking_frames": gt_count - 2,
            "initial_image": frames[2]["path"],
            "initial_gt_row": 2,
            "frames": frames,
        }
        print(f"ALIGNED {scene}: {gt_count} images / object GT rows; "
              f"track {gt_count - 2}; init={PurePosixPath(names[2]).name} GT row=2", flush=True)

    manifest_bytes = (json.dumps(manifest, indent=2) + "\n").encode()
    info = tarfile.TarInfo("manifest.json")
    info.size = len(manifest_bytes)
    prepared.addfile(info, io.BytesIO(manifest_bytes))

with tarfile.open(partial) as prepared:
    expected_files = 2 + sum(s["frames_before_offset"] + 2 for s in manifest["scenes"].values())
    assert len(prepared.getmembers()) == expected_files
    assert prepared.extractfile("manifest.json").read() == manifest_bytes
partial.replace(output / "prepared.tar")
(preview / "manifest.json").write_bytes(manifest_bytes)
print("PREPARE OK:", output / "prepared.tar", flush=True)
print("REVIEW INPUT FRAMES:", preview, flush=True)
print("Next: select the object in each image, generate SAM 2 masks, then review masks.")
