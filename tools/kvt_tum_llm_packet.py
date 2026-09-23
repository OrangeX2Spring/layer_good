"""Prepare a score-blind, whole-clip TUM RGB packet on CAMP; no model inference."""
import argparse
import hashlib
from pathlib import Path
import shutil
import tarfile

import cv2
import numpy as np

from kvt_tum_run import prepare, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--tag', required=True)
    args = parser.parse_args()
    scene = 'freiburg3_long_office_household'
    source = Path('/mnt/datasets/tum-rgbd') / f'rgbd_dataset_{scene}.zip'
    staged = args.work / 'inputs' / scene
    manifest = prepare(source, staged, 308, .02)
    digest = hashlib.sha256()
    with source.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    packet = args.work / 'llm_packet'
    packet.mkdir()
    (packet / 'rgb').mkdir()
    (packet / 'sheets').mkdir()
    records = []
    sheets = []
    for start in range(0, manifest['frames'], 12):
        batch = manifest['inputs'][start:start + 12]
        height, width, channels = batch[0]['shape']
        assert channels == 3
        sheet = np.full((3 * (height + 28), 4 * width, 3), 255, np.uint8)
        sheet_name = f'sheets/{start:06d}.jpg'
        for position, row in enumerate(batch):
            index = row['index']
            assert index == len(records)
            image_path = staged / 'model_rgb' / Path(row['file']).name
            bgr = cv2.imread(str(image_path))
            assert bgr is not None and list(bgr.shape) == row['shape']
            assert hashlib.sha256(bgr[:, :, ::-1].tobytes()).hexdigest() == row['model_rgb_sha256']
            name = f'rgb/{index:06d}.png'
            shutil.copyfile(image_path, packet / name)
            records.append(dict(index=index, file=name, timestamp=row['timestamp'],
                                source_file=row['file'], source_sha256=row['source_sha256'],
                                rgb_sha256=row['model_rgb_sha256'],
                                png_sha256=hashlib.sha256(image_path.read_bytes()).hexdigest(),
                                sheet=sheet_name))
            y, x = (position // 4) * (height + 28), (position % 4) * width
            cv2.putText(sheet, str(index), (x + 5, y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 0, 0), 1, cv2.LINE_AA)
            sheet[y + 28:y + 28 + height, x:x + width] = bgr
        assert cv2.imwrite(str(packet / sheet_name), sheet, [cv2.IMWRITE_JPEG_QUALITY, 95])
        sheets.append(dict(file=sheet_name, indices=[r['index'] for r in batch],
                           sha256=hashlib.sha256((packet / sheet_name).read_bytes()).hexdigest()))
    write_json(packet / 'manifest.json', dict(scene=scene, frames=len(records),
        budget_including_bootstrap=20, bootstrap_index=0, insertion_count=19,
        resize_dim=308, offset=0, source_zip_sha256=digest.hexdigest(),
        inputs=records, sheets=sheets))
    attribution = (
        'TUM RGB-D Benchmark — J. Sturm, N. Engelhard, F. Endres, W. Burgard, '
        'D. Cremers, A Benchmark for the Evaluation of RGB-D SLAM Systems, IROS 2012.\n'
        'Source: https://cvg.cit.tum.de/data/datasets/rgbd-dataset\n'
        'License: CC BY 4.0, https://creativecommons.org/licenses/by/4.0/\n'
        'Changes: RGB resized using KV-Tracker Pi3 preprocessing at resolution 308; '
        'numbered JPEG contact sheets added. Individual PNGs retain model-input pixels.\n')
    (packet / 'ATTRIBUTION.txt').write_text(attribution)
    prompt = Path(__file__).with_name('KVT_TUM_LLM_PROMPT.md').read_text()
    (packet / 'PROMPT.md').write_text(prompt.replace('{{N}}', str(len(records))))
    write_json(packet / 'PACKET_OK.json', dict(frames=len(records), sheets=len(sheets),
        all_frames_verified=True, selection_complete=False))
    destination = args.out / f'{args.tag}_llm_packet.tar'
    assert not destination.exists(), destination
    temporary = destination.with_suffix('.tar.partial')
    with tarfile.open(temporary, 'w') as archive:
        archive.add(packet, arcname='llm_packet')
    temporary.replace(destination)
    # The wrapper archives work/context separately; avoid a second packet copy.
    shutil.rmtree(packet)
    (args.work / 'JOB_OK').write_text('Packet preparation only; no selection or tracking run.\n')
    print(f'LLM PACKET OK {destination}: {len(records)} frames, {len(sheets)} sheets', flush=True)


if __name__ == '__main__':
    main()
