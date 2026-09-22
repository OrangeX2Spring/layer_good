"""Stage consecutive ARCTIC object frames and saved KV-Tracker SAM masks on CAMP."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile

import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--kvt', type=Path, required=True)
    parser.add_argument('--scene', default='box_grab_01')
    parser.add_argument('--count', type=int, default=241)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    assert args.count >= 2
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / 'rgb').mkdir()
    (args.out / 'masks').mkdir()
    with tarfile.open(args.prepared) as source, tarfile.open(args.kvt) as baseline:
        manifest = json.load(source.extractfile('manifest.json'))
        reference = json.load(baseline.extractfile('manifest.json'))
        assert manifest == reference['prepare_manifest']
        assert manifest['loader_offset'] == reference['offset'] == 2
        frames = manifest['scenes'][args.scene]['frames'][2:]
        assert len(frames) >= args.count
        mask_names = {member.name for member in baseline.getmembers() if member.isfile()
                      and member.name.startswith(f'{args.scene}/results/sam_masks/')}
        assert mask_names == {f'{args.scene}/results/sam_masks/{i:05d}.png'
                              for i in range(len(frames))}
        rows = []
        for index, frame in enumerate(frames[:args.count]):
            rgb_bytes = source.extractfile(frame['path']).read()
            mask_member = f'{args.scene}/results/sam_masks/{index:05d}.png'
            mask_bytes = baseline.extractfile(mask_member).read()
            rgb = np.asarray(Image.open(io.BytesIO(rgb_bytes)).convert('RGB'))
            mask = np.asarray(Image.open(io.BytesIO(mask_bytes)).convert('L')) > 127
            assert rgb.shape[:2] == mask.shape and mask.any() and (~mask).any()
            rgb_path = f'rgb/{index:06d}.png'
            mask_path = f'masks/{index:06d}.png'
            Image.fromarray(rgb).save(args.out / rgb_path)
            (args.out / mask_path).write_bytes(mask_bytes)
            rows.append(dict(rgb=rgb_path, mask=mask_path, source_member=frame['path'],
                             source_sha256=hashlib.sha256(rgb_bytes).hexdigest(),
                             mask_member=mask_member,
                             source_mask_sha256=hashlib.sha256(mask_bytes).hexdigest(),
                             image_id=frame.get('image_id'), kvt_index=index))
    (args.out / 'manifest.json').write_text(json.dumps(dict(
        sequence=f'ARCTIC s01 {args.scene}', frames=rows, mask_input=True,
        selection='first consecutive post-offset frames',
        prepared_archive=str(args.prepared), kvt_archive=str(args.kvt)), indent=2) + '\n')
    print(f'ARCTIC STAGED {args.scene} {args.count}', flush=True)


if __name__ == '__main__':
    main()
