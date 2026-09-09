import json
import struct
import zlib
from pathlib import Path

import numpy as np

OUT = Path("tools/kvt_roi_cup_s75")
d = np.load("cluster_results/kvt_starts/cup518_s75/reference.npz")
om, xyz, valid = d["object_mask"], d["xyz"], d["valid"]


def flood_holes(mask):
    comp = ~mask
    out = np.zeros_like(comp)
    out[0] |= comp[0]; out[-1] |= comp[-1]
    out[:, 0] |= comp[:, 0]; out[:, -1] |= comp[:, -1]
    while True:
        g = out.copy()
        g[1:] |= out[:-1]; g[:-1] |= out[1:]
        g[:, 1:] |= out[:, :-1]; g[:, :-1] |= out[:, 1:]
        g &= comp
        if np.array_equal(g, out):
            return comp & ~out
        out = g


def dilate(m, n=1):
    for _ in range(n):
        g = m.copy()
        g[1:] |= m[:-1]; g[:-1] |= m[1:]
        g[:, 1:] |= m[:, :-1]; g[:, :-1] |= m[:, 1:]
        m = g
    return m


def erode(m, n=1):
    return ~dilate(~m, n)


def write_png(path, arr):
    raw = b"".join(b"\x00" + arr[i].tobytes() for i in range(arr.shape[0]))

    def chunk(tag, payload):
        body = tag + payload
        return struct.pack(">I", len(payload)) + body + struct.pack(">I", zlib.crc32(body))

    head = struct.pack(">IIBBBBB", arr.shape[1], arr.shape[0], 8, 0, 0, 0, 0)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", head)
                     + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


hole = flood_holes(om)
z = xyz[..., 2]
ring = dilate(hole, 2) & om & valid
z0 = float(np.median(z[ring]))
ok = z > 0
ray = np.zeros_like(xyz)
ray[ok] = xyz[ok] / z[ok][:, None]
vy, vx = np.nonzero(ok)
coef, *_ = np.linalg.lstsq(np.c_[vx, vy, np.ones(vx.size)], ray[ok][:, :2], rcond=None)
hy, hx = np.nonzero(hole)
centre = (np.c_[np.c_[hx, hy, np.ones(hx.size)] @ coef, np.ones(hx.size)] * z0).mean(0)

# local depth range over a 3x3 window: reject discontinuities
pad = np.where(ok, z, np.nan)
stack = np.stack([np.roll(np.roll(pad, i, 0), j, 1) for i in (-1, 0, 1) for j in (-1, 0, 1)])
with np.errstate(invalid="ignore"):
    spread = np.nanmax(stack, 0) - np.nanmin(stack, 0)
smooth = np.nan_to_num(spread, nan=1.0) < 0.003

anchor = erode(om & valid, 3) & smooth & ~dilate(hole, 12)
img = np.zeros(om.shape, np.uint8)
img[anchor] = 255
OUT.mkdir(exist_ok=True)
write_png(OUT / "anchor_mask.png", img)

roi_to_reference = np.eye(4)
roi_to_reference[:3, 3] = centre
p = xyz[om & valid]
cx, cy = p[:, 0].mean(), p[:, 1].mean()
span = max(np.ptp(p[:, 0]), np.ptp(p[:, 1])) * 1.6 / 2
EVIDENCE = (
    "HouseCat6D test_scene1 instance 4 (cup-grey_handle), reference frame 000075. "
    "The annotation mask encloses a 231 px hole at the handle; the wooden table is "
    "directly visible through the loop in the reference RGB. GT depth is dense on the "
    "object (6407/6407 object pixels valid) and returns nothing inside the hole. "
    "The {0} cm box contains no GT point; the nearest valid GT point is {1} mm away. "
    "The scanned mesh was not consulted."
)

base = {"reference": "reference.npz", "anchor_mask": "anchor_mask.png",
        "roi_to_reference": roi_to_reference.tolist(),
        "voxel_size_m": 0.002, "max_alignment_rmse_m": 0.005,
        "render_reference_to_view": np.eye(4).tolist(),
        "render_bounds_m": [float(cx - span), float(cx + span),
                            float(cy - span), float(cy + span)]}
for name, extent, label, gap in (("evaluation.json", [0.014, 0.014, 0.012], "1.4x1.4x1.2", "1.8"),
                                 ("evaluation_tight.json", [0.012, 0.012, 0.010], "1.2x1.2x1.0", "2.9")):
    cfg = dict(base, roi_extent_m=extent, empty_region_evidence=EVIDENCE.format(label, gap))
    (OUT / name).write_text(json.dumps(cfg, indent=2) + "\n")

# report
ref_to_roi = np.linalg.inv(roi_to_reference)
a = xyz[anchor] @ ref_to_roi[:3, :3].T + ref_to_roi[:3, 3]
print("anchor pixels:", int(anchor.sum()))
print("anchors inside ROI (must be 0):",
      int((np.abs(a) < np.array([0.014, 0.014, 0.012]) / 2).all(1).sum()))
print("anchor rank:", np.linalg.matrix_rank(xyz[anchor] - xyz[anchor].mean(0)))
print("roi centre:", centre.round(4))
print("render bounds:", np.round(base["render_bounds_m"], 4))
print("grid 1.4cm:", np.ceil(np.array([.014, .014, .012]) / .002).astype(int),
      "= ", int(np.prod(np.ceil(np.array([.014, .014, .012]) / .002))), "voxels")
