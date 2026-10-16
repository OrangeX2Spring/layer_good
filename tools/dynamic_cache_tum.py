"""Prepare one fixed TUM clip with SAM3 predicted person masks on CAMP."""

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
from PIL import Image
import torch


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def select_frames(rgb_text, gt_text):
    rgb = sorted((float(row.split()[0]), row.split()[1])
                 for row in rgb_text.splitlines() if row.strip() and not row.lstrip().startswith('#'))
    gt = np.loadtxt(gt_text.splitlines())
    if gt.ndim != 2 or gt.shape[1] != 8 or not np.isfinite(gt).all():
        raise ValueError('Expected finite TUM timestamp/translation/quaternion rows')
    # Fixed before seeing masks or tracking results: 5 seconds in, stride two.
    selected = [row for row in rgb if row[0] >= rgb[0][0] + 5][::2][:32]
    if len(selected) != 32:
        raise ValueError('Need 32 frames at the fixed clip offset and stride')
    frames = []
    for timestamp, member in selected:
        nearest = gt[np.argmin(abs(gt[:, 0] - timestamp))]
        pose = None
        if abs(nearest[0] - timestamp) <= .02:
            q = nearest[4:8]
            if not np.isclose(np.linalg.norm(q), 1, atol=1e-3):
                raise ValueError('Invalid TUM unit quaternion')
            x, y, z, w = q / np.linalg.norm(q)
            matrix = np.eye(4)
            matrix[:3, :3] = [
                [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]]
            matrix[:3, 3] = nearest[1:4]
            pose = matrix.tolist()
        frames.append(dict(timestamp=timestamp, archive_member=member,
                           gt_timestamp=float(nearest[0]) if pose is not None else None,
                           gt_c2w=pose))
    return frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--zip', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert sys.platform == 'linux' and torch.cuda.is_available(), 'CAMP GPU only'
    assert args.output.resolve().is_relative_to('/tmp')
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    args.output.mkdir()
    (args.output / 'rgb').mkdir()
    (args.output / 'masks').mkdir()
    sequence = 'rgbd_dataset_freiburg3_walking_xyz'
    with zipfile.ZipFile(args.zip) as archive:
        metadata = {}
        for name in ('rgb.txt', 'depth.txt', 'groundtruth.txt'):
            content = archive.read(f'{sequence}/{name}')
            (args.output / name).write_bytes(content)
            metadata[name] = content.decode('utf-8')
        frames = select_frames(metadata['rgb.txt'], metadata['groundtruth.txt'])
        for index, row in enumerate(frames):
            member = row['archive_member']
            if not member.startswith('rgb/') or '..' in Path(member).parts:
                raise ValueError(f'Unexpected RGB member: {member}')
            row['rgb'] = f'rgb/{index:06d}.png'
            (args.output / row['rgb']).write_bytes(archive.read(f'{sequence}/{member}'))

    torch.manual_seed(0)
    model = build_sam3_image_model(checkpoint_path=str(args.checkpoint),
                                  load_from_HF=False, device='cuda', eval_mode=True)
    processor = Sam3Processor(model, confidence_threshold=0.5)
    for index, row in enumerate(frames):
        with Image.open(args.output / row['rgb']) as image:
            image = image.convert('RGB')
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                state = processor.set_image(image)
                state = processor.set_text_prompt(prompt='person', state=state)
            masks = state['masks']
            assert masks.dtype == torch.bool and masks.ndim == 4
            assert tuple(masks.shape[1:]) == (1, image.height, image.width)
            union = masks[:, 0].any(dim=0).cpu().numpy()
            row['motion_mask'] = f'masks/{index:06d}.png'
            Image.fromarray(union.astype(np.uint8) * 255).save(args.output / row['motion_mask'])
            row['person_instances'] = len(masks)
            row['person_scores'] = state['scores'].detach().cpu().tolist()
            row['person_pixel_fraction'] = float(union.mean())
            print(f'SAM3 PERSON frame={index} instances={len(masks)} fraction={union.mean():.4f}', flush=True)
    manifest = dict(sequence=sequence, role='primary_dynamic', pose_units='metres',
                    selection=dict(offset_seconds=5, stride=2, frames=32),
                    archive_path=str(args.zip), archive_sha256=sha256(args.zip),
                    mask_source=dict(kind='predicted_person',
                        description='SAM3 person-prompt instance union; background is non-person, not verified static',
                        reference='sam3.model_builder.build_sam3_image_model / Sam3Processor',
                        prompt='person', confidence_threshold=0.5, mask_probability_threshold=0.5,
                        inference_autocast='bfloat16',
                        checkpoint=str(args.checkpoint), checkpoint_sha256=sha256(args.checkpoint),
                        builder_sha256=sha256(Path(inspect.getfile(build_sam3_image_model))),
                        processor_sha256=sha256(Path(inspect.getfile(Sam3Processor))),
                        torch_version=torch.__version__, cuda_version=torch.version.cuda), frames=frames)
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
    print('TUM PERSON INPUTS OK; predicted foreground, not motion GT', flush=True)


if __name__ == '__main__':
    main()
