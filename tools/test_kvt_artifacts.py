"""Remote-only integration check: known Sim(3), duplicate bootstrap, intrusion, MP4 decode."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np


def main():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        capture = root / "capture"
        (capture / "keyframes").mkdir(parents=True)
        (capture / "capture_complete.json").write_text("{}")
        yy, xx = np.indices((6, 6))
        surface = np.stack((xx * 0.01, yy * 0.01, np.ones_like(xx)), axis=-1).astype(float)
        mask = np.ones((6, 6), bool)
        rgb = np.full((6, 6, 3), 180, np.uint8)
        np.savez_compressed(root / "reference.npz", xyz=surface, valid=mask, object_mask=mask, rgb=rgb)
        assert cv2.imwrite(str(root / "anchor.png"), mask.astype(np.uint8) * 255)
        cfg = {"reference": "reference.npz", "anchor_mask": "anchor.png",
               "roi_to_reference": [[1, 0, 0, 0.025], [0, 1, 0, 0.025], [0, 0, 1, 2], [0, 0, 0, 1]],
               "roi_extent_m": [0.1, 0.1, 0.1], "voxel_size_m": 0.01,
               "max_alignment_rmse_m": 0.001, "empty_region_evidence": "Synthetic plane at z=1; ROI at z=2",
               "render_reference_to_view": np.eye(4).tolist(), "render_bounds_m": [-0.1, 0.2, -0.1, 0.2]}
        (root / "evaluation.json").write_text(json.dumps(cfg))
        rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        translation = np.array([3, 4, 5])
        for step in range(2):
            maps = np.stack((surface.copy(), surface.copy()))
            if step == 1:
                maps[1, 0, 0] = [0.025, 0.025, 2.0]
            prediction = (maps - translation) @ rotation / 2.0
            np.savez_compressed(capture / "keyframes" / f"{step:06d}.npz", xyz=prediction,
                                rgb=np.stack((rgb, rgb)), confidence=np.ones((2, 6, 6)),
                                masks=np.stack((mask, mask)), frame_ids=[0, step], threshold=0.5,
                                latest_rgb=rgb)
        subprocess.run([sys.executable, str(Path(__file__).with_name("kvt_artifacts.py")),
                        "--capture", str(capture), "--evaluation", str(root / "evaluation.json"),
                        "--out", str(root / "out")], check=True)
        rows = json.loads((root / "out" / "metrics.json").read_text())
        assert [row["keyframes"] for row in rows] == [1, 2]
        assert [row["retained_points"] for row in rows] == [36, 72]
        assert [row["roi_points"] for row in rows] == [0, 1]
        assert rows[1]["occupied_voxels"] == 1
        assert rows[1]["roi_points_old_views"] == 0
        assert rows[1]["roi_points_new_views"] == 1
        assert all(row["alignment_valid"] for row in rows)
        assert rows[1]["old_view_displacement_rmse_m"] < 1e-10
        print("SYNTHETIC CHECK OK")


if __name__ == "__main__":
    main()
