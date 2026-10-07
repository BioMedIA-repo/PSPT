import argparse
import os
import time

import h5py
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from encoders import get_encoder
from utils.file_utils import save_hdf5


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class ScatterPngBag(Dataset):
    def __init__(self, scatter_png_dir, slide_id, img_transforms, coord_h5_path=None,
                 allow_index_coords=False):
        self.scatter_png_dir = scatter_png_dir
        self.slide_id = slide_id
        self.img_transforms = img_transforms
        self.patch_dir = os.path.join(scatter_png_dir, slide_id)
        if not os.path.isdir(self.patch_dir):
            raise FileNotFoundError(f'Missing scatter PNG directory: {self.patch_dir}')

        self.patch_paths = []
        idx = 0
        while True:
            patch_path = os.path.join(self.patch_dir, f'{slide_id}_patch_{idx}.png')
            if not os.path.exists(patch_path):
                break
            self.patch_paths.append(patch_path)
            idx += 1
        if not self.patch_paths:
            raise RuntimeError(f'No contiguous patch PNGs found for {slide_id}')

        self.coords = self._load_coords(coord_h5_path, allow_index_coords)
        if self.coords.shape[0] != len(self.patch_paths):
            raise RuntimeError(
                f'{slide_id}: coords length {self.coords.shape[0]} != '
                f'patch count {len(self.patch_paths)}'
            )

    def _load_coords(self, coord_h5_path, allow_index_coords):
        if coord_h5_path is not None and os.path.exists(coord_h5_path):
            with h5py.File(coord_h5_path, 'r') as handle:
                if 'coords' not in handle:
                    raise KeyError(f'{coord_h5_path} does not contain coords')
                return handle['coords'][:].astype(np.int32)
        if not allow_index_coords:
            raise FileNotFoundError(
                f'Missing coord h5 for {self.slide_id}: {coord_h5_path}. '
                'Pass --allow_index_coords only if downstream code does not use coords.'
            )
        coords = np.zeros((len(self.patch_paths), 2), dtype=np.int32)
        coords[:, 0] = np.arange(len(self.patch_paths), dtype=np.int32)
        return coords

    def __len__(self):
        return len(self.patch_paths)

    def __getitem__(self, idx):
        img = Image.open(self.patch_paths[idx]).convert('RGB')
        if self.img_transforms is not None:
            img = self.img_transforms(img)
        return {'img': img, 'coord': self.coords[idx]}


def read_slide_ids(csv_path, slide_ext):
    df = pd.read_csv(csv_path)
    if 'wsi_id' in df.columns:
        ids = df['wsi_id'].astype(str).tolist()
    elif 'slide_id' in df.columns:
        ids = df['slide_id'].astype(str).tolist()
    else:
        ids = df.iloc[:, 0].astype(str).tolist()
    suffix = slide_ext if slide_ext.startswith('.') else f'.{slide_ext}'
    return [x[:-len(suffix)] if x.endswith(suffix) else x for x in ids]


def compute_w_loader(output_path, loader, model, verbose=0):
    if verbose > 0:
        print(f'processing a total of {len(loader)} batches')
    mode = 'w'
    for data in tqdm(loader):
        with torch.inference_mode():
            batch = data['img'].to(device, non_blocking=True)
            coords = data['coord'].numpy().astype(np.int32)
            features = model(batch).cpu().numpy().astype(np.float32)
            save_hdf5(
                output_path,
                {'features': features, 'coords': coords},
                attr_dict=None,
                mode=mode,
            )
            mode = 'a'
    return output_path


def main(args):
    os.makedirs(args.feat_dir, exist_ok=True)
    os.makedirs(os.path.join(args.feat_dir, 'pt_files'), exist_ok=True)
    os.makedirs(os.path.join(args.feat_dir, 'h5_files'), exist_ok=True)
    dest_files = set(os.listdir(os.path.join(args.feat_dir, 'pt_files')))

    slide_ids = read_slide_ids(args.csv_path, args.slide_ext)
    if args.only_slide_ids:
        wanted = set(args.only_slide_ids.split(','))
        slide_ids = [slide_id for slide_id in slide_ids if slide_id in wanted]

    model, img_transforms = get_encoder(args.model_name, target_img_size=args.target_patch_size)
    model.eval()
    model = model.to(device)
    loader_kwargs = {'num_workers': args.num_workers, 'pin_memory': True} if device.type == 'cuda' else {'num_workers': args.num_workers}

    for slide_idx, slide_id in enumerate(tqdm(slide_ids)):
        print(f'\nprogress: {slide_idx}/{len(slide_ids)}')
        print(slide_id)
        if not args.no_auto_skip and f'{slide_id}.pt' in dest_files:
            print(f'skipped {slide_id}')
            continue

        bag_name = f'{slide_id}.h5'
        output_path = os.path.join(args.feat_dir, 'h5_files', bag_name)
        coord_h5_path = None
        if args.coord_h5_dir:
            coord_h5_path = os.path.join(args.coord_h5_dir, 'patches', bag_name)
            if not os.path.exists(coord_h5_path):
                coord_h5_path = os.path.join(args.coord_h5_dir, bag_name)

        time_start = time.time()
        dataset = ScatterPngBag(
            args.scatter_png_dir,
            slide_id,
            img_transforms,
            coord_h5_path=coord_h5_path,
            allow_index_coords=args.allow_index_coords,
        )
        loader = DataLoader(dataset=dataset, batch_size=args.batch_size, **loader_kwargs)
        output_file_path = compute_w_loader(output_path, loader, model=model, verbose=1)
        print(f'\ncomputing features for {output_file_path} took {time.time() - time_start} s')

        with h5py.File(output_file_path, 'r') as handle:
            features = handle['features'][:]
            print('features size: ', features.shape)
            print('coordinates size: ', handle['coords'].shape)

        features = torch.from_numpy(features)
        torch.save(features, os.path.join(args.feat_dir, 'pt_files', f'{slide_id}.pt'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Feature extraction from pre-exported scatter PNG patches')
    parser.add_argument('--scatter_png_dir', type=str, required=True)
    parser.add_argument('--coord_h5_dir', type=str, default=None,
                        help='Directory containing CLAM coordinate h5 files, with optional patches/ subdir')
    parser.add_argument('--csv_path', type=str, required=True)
    parser.add_argument('--feat_dir', type=str, required=True)
    parser.add_argument('--model_name', type=str, default='uni_v1',
                        choices=['uni_v1', 'conch_v1', 'plip'])
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--no_auto_skip', default=False, action='store_true')
    parser.add_argument('--target_patch_size', type=int, default=224)
    parser.add_argument('--slide_ext', type=str, default='.svs')
    parser.add_argument('--only_slide_ids', type=str, default=None,
                        help='Comma-separated slide ids for smoke tests')
    parser.add_argument('--allow_index_coords', default=False, action='store_true')
    parser.add_argument("--encoder-weights", required=True)
    args = parser.parse_args()
    os.environ["PSPT_ENCODER_WEIGHTS"] = args.encoder_weights
    main(args)
