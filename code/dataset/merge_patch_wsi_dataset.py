import os.path
import json
import random

import cv2
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import pytorch_lightning as pl
from tqdm import tqdm
import kornia as K
from collections import defaultdict
try:
    import jpeg4py as jpeg
    use_jpeg4py = True
except:
    use_jpeg4py = False


def read_rgb_img(img_p):
    if use_jpeg4py and img_p.lower().endswith(('.jpg', 'jpeg')):
        img = jpeg.JPEG(img_p).decode()
    else:
        img = cv2.cvtColor(cv2.imread(img_p, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    return img


class MergePatchWsiDataset(Dataset):

    def __init__(self, dataset_root, dataset_csv_path, data_type, data_ext='.jpg', classes_names=None,
                 drop_out=0., val_fold_id=-1, **kwargs):
        super().__init__()
        if classes_names is None:
            self.CLASSES = None
            self.CLASS_NAMES = None
        else:
            self.CLASSES = classes_names[0]
            self.CLASS_NAMES = classes_names[1]

        self.dataset_root = dataset_root
        self.dataset_csv_path = dataset_csv_path
        self.data_ext = data_ext

        if data_type not in ['train', 'validation', 'test']:
            raise Exception('Not supported dataset type. It should be train or test')
        self.data_type = data_type
        self.val_fold_id = val_fold_id
        if data_type == 'test':
            self.val_fold_id = -1

        if val_fold_id >= 0:
            self.wsi_list = self.read_cv_dataset_csv()
        else:
            if data_type == 'validation':
                data_type = 'test'
                self.data_type = data_type
            self.wsi_list = self.read_dataset_csv()

        self.drop_out = drop_out

    def read_dataset_csv(self):
        df = pd.read_csv(self.dataset_csv_path, header=0)
        if self.data_type in ['test']:
            df = df[df['is_test'] > 0]
        else:  # train
            df = df[df['is_test'] == 0]
        return df

    def read_cv_dataset_csv(self):
        df = pd.read_csv(self.dataset_csv_path, header=0)
        if self.data_type in ['validation']:
            df = df[df['fold'] == self.val_fold_id]
        elif self.data_type in ['test']:
            df = df[df['fold'] < 0]
        else:
            df = df[df['fold'] > 0]
            df = df[df['fold'] != self.val_fold_id]
        return df

    def __len__(self):
        return len(self.wsi_list)

    def __getitem__(self, i):
        row = self.wsi_list.iloc[i]
        wsi_id = row['wsi_id']
        label = row['label']
        len_img = row['len_img']

        assert self.dataset_root, "dataset_root required for strip images; use --scatter-png-dir for individual patch PNGs"
        tiles = []
        for i in range(len_img):
            tile = read_rgb_img(os.path.join(self.dataset_root, '%s_%d%s' % (wsi_id, i, self.data_ext)))
            assert len(tile.shape) == 3
            h, w, c = tile.shape
            tile = tile.reshape(h // w, w, w, c)
            tiles.append(tile)
        tiles = np.concatenate(tiles, axis=0)

        tiles = K.utils.image_to_tensor(tiles)
        tiles = K.enhance.normalize(tiles, torch.tensor(0.), torch.tensor(255.))

        if self.drop_out > 0:
            perm = torch.randperm(tiles.size(0))
            idx = perm[:int((1 - self.drop_out) * tiles.size(0))]
            tiles = tiles[idx]

        return tiles, label

    def get_label(self, idx):
        label = self.wsi_list.iloc[idx]['label']
        return int(label)

    def get_weights_of_class(self):
        labels = self.wsi_list['label']
        unique, counts = np.unique(labels, return_counts=True)
        label_cnt = list(zip(unique, counts))
        label_cnt.sort(key=lambda x: x[0])
        weight_arr = np.array([x[1] for x in label_cnt], dtype=float)
        weight_arr = np.max(weight_arr) / weight_arr
        return torch.from_numpy(weight_arr.astype(np.float32))


class PCPSSelectedWSIDataset(Dataset):
    def __init__(self, dataset_root, dataset_csv_path, data_type, data_ext='.jpg', classes_names=None,
                 drop_out=0., val_fold_id=-1, 
                 pcps_selection_path=None, pcps_selected_count=384, pcps_random_count=128,
                 pcps_sampling_mode='legacy', pcps_total_count=None,
                 pcps_evidence_concentration=0.5,
                 pcps_context_temperature=float('inf'),
                 pcps_distribution_concentration=1.0,
                 pcps_eval_mode='full', pcps_eval_sampling_seed=None,
                 scatter_png_dir=None,
                 **kwargs):
        super().__init__()
        if classes_names is None:
            self.CLASSES = None
            self.CLASS_NAMES = None
        else:
            self.CLASSES = classes_names[0]
            self.CLASS_NAMES = classes_names[1]

        self.dataset_root = dataset_root
        self.dataset_csv_path = dataset_csv_path
        self.data_ext = data_ext

        if data_type not in ['train', 'validation', 'test']:
            raise Exception('Not supported dataset type. It should be train or test')
        self.data_type = data_type
        self.val_fold_id = val_fold_id
        if data_type == 'test':
            self.val_fold_id = -1

        if val_fold_id >= 0:
            self.wsi_list = self.read_cv_dataset_csv()
        else:
            if data_type == 'validation':
                data_type = 'test'
                self.data_type = data_type
            self.wsi_list = self.read_dataset_csv()

        self.drop_out = drop_out
        self.pcps_selected_count = pcps_selected_count
        self.pcps_random_count = pcps_random_count
        self.pcps_sampling_mode = pcps_sampling_mode
        self.pcps_total_count = pcps_total_count
        self.pcps_evidence_concentration = float(pcps_evidence_concentration)
        self.pcps_context_temperature = float(pcps_context_temperature)
        self.pcps_distribution_concentration = float(pcps_distribution_concentration)
        self.pcps_eval_mode = pcps_eval_mode
        self.pcps_eval_sampling_seed = (
            0 if pcps_eval_sampling_seed is None else int(pcps_eval_sampling_seed)
        )
        if self.pcps_eval_mode not in {'full', 'topk', 'matched'}:
            raise ValueError('--pcps-eval-mode must be full, topk, or matched')
        if self.pcps_sampling_mode == 'evidence_context':
            if self.pcps_total_count is None or self.pcps_total_count <= 0:
                raise ValueError('evidence_context mode requires --pcps-total-count > 0')
            if not 0.0 <= self.pcps_evidence_concentration <= 1.0:
                raise ValueError('--pcps-evidence-concentration must be in [0, 1]')
            if self.pcps_context_temperature <= 0:
                raise ValueError('--pcps-context-temperature must be positive')
        if self.pcps_sampling_mode == 'gumbel_top_m':
            if self.pcps_total_count is None or self.pcps_total_count <= 0:
                raise ValueError('gumbel_top_m mode requires --pcps-total-count > 0')
            if self.pcps_distribution_concentration < 0:
                raise ValueError('--pcps-distribution-concentration must be nonnegative')
        self.scatter_png_dir = scatter_png_dir
        self._uniform_patch_counts = {}
        if self.pcps_sampling_mode == 'uniform':
            if not self.scatter_png_dir or not self.pcps_total_count or self.pcps_total_count <= 0:
                raise ValueError('uniform mode requires PNG root and a positive patch budget')
            if pcps_selection_path is not None or self.pcps_eval_mode != 'matched':
                raise ValueError('uniform mode uses no PCPS cache and matched-budget evaluation')

        # PCPS selection and all-patch diagnostics
        self.pcps_selection = None
        self.pcps_patch_diagnostics = None
        if pcps_selection_path is not None:
            with open(pcps_selection_path, 'r') as f:
                self.pcps_selection = json.load(f)
            json_path = os.path.abspath(pcps_selection_path)
            diagnostic_name = os.path.basename(json_path).replace(
                'pcps_selection_', 'pcps_patch_diagnostics_'
            ).replace('.json', '.pt')
            diagnostic_path = os.path.join(os.path.dirname(json_path), diagnostic_name)
            if os.path.exists(diagnostic_path):
                self.pcps_patch_diagnostics = torch.load(
                    diagnostic_path, map_location='cpu', weights_only=False
                )
        if self.pcps_selection is not None and self.pcps_sampling_mode in {'evidence_context', 'gumbel_top_m'} and self.pcps_patch_diagnostics is None:
            raise FileNotFoundError(
                f'{self.pcps_sampling_mode} mode requires the all-patch PCPS diagnostics PT '
                'next to the selection JSON'
            )

    def _full_pcps_ranking_and_scores(self, wsi_id, total_patches):
        """Return a stable all-patch ranking while preserving exported Top-K order."""
        item = self.pcps_patch_diagnostics[wsi_id]
        score_key = (
            'pcps_selection_score' if 'pcps_selection_score' in item
            else 'task_relevance_posterior'
        )
        scores = item[score_key].float().reshape(-1)
        if scores.numel() != total_patches:
            raise ValueError(
                f'{wsi_id}: diagnostics contain {scores.numel()} scores but '
                f'selection metadata reports {total_patches} patches'
            )
        exported = [
            int(index) for index in self.pcps_selection[wsi_id]['selected_patch_indices']
            if 0 <= int(index) < total_patches
        ]
        exported_set = set(exported)
        tail = [
            int(index) for index in torch.argsort(scores, descending=True).tolist()
            if int(index) not in exported_set
        ]
        return exported + tail, scores

    def _sample_training_indices(self, wsi_id, total_patches):
        """Sample one training bag; exposed separately for reproducibility tests."""
        if self.pcps_sampling_mode == 'uniform':
            return sorted(random.sample(range(total_patches), min(int(self.pcps_total_count), total_patches)))
        if self.pcps_sampling_mode == 'legacy':
            ranked = self.pcps_selection[wsi_id]['selected_patch_indices']
            core = ranked[:self.pcps_selected_count]
            core_set = set(core)
            remaining = [index for index in range(total_patches) if index not in core_set]
            context = random.sample(remaining, min(self.pcps_random_count, len(remaining)))
            return sorted(core + context)

        ranking, scores = self._full_pcps_ranking_and_scores(wsi_id, total_patches)
        if self.pcps_sampling_mode == 'gumbel_top_m':
            total_count = min(int(self.pcps_total_count), total_patches)
            if total_count == total_patches:
                return list(range(total_patches))
            eps = 1e-6
            posterior_logits = torch.logit(scores.clamp(eps, 1.0 - eps), eps=eps)
            standardized_evidence = (
                posterior_logits - posterior_logits.mean()
            ) / posterior_logits.std(unbiased=False).clamp_min(1e-6)
            # Gumbel-Top-k samples without replacement from the Plackett-Luce
            # distribution induced by softmax(kappa * standardized_evidence).
            # Every patch has non-zero probability for every finite kappa.
            uniform = torch.rand_like(standardized_evidence).clamp_(eps, 1.0 - eps)
            gumbel = -torch.log(-torch.log(uniform))
            keys = self.pcps_distribution_concentration * standardized_evidence + gumbel
            return sorted(torch.topk(keys, total_count).indices.tolist())

        total_count = min(int(self.pcps_total_count), total_patches)
        core_count = min(
            total_count,
            int(round(total_count * self.pcps_evidence_concentration)),
        )
        core = ranking[:core_count]
        core_set = set(core)
        remaining = [index for index in range(total_patches) if index not in core_set]
        context_count = min(total_count - core_count, len(remaining))
        if context_count == 0:
            context = []
        elif np.isinf(self.pcps_context_temperature):
            context = random.sample(remaining, context_count)
        else:
            remaining_tensor = torch.as_tensor(remaining, dtype=torch.long)
            context_scores = scores[remaining_tensor]
            # Per-slide standardization makes one temperature transferable across
            # WSIs whose posterior ranges differ after budget calibration.
            context_scores = (
                context_scores - context_scores.mean()
            ) / context_scores.std(unbiased=False).clamp_min(1e-6)
            weights = torch.softmax(
                context_scores / self.pcps_context_temperature, dim=0
            )
            sampled_positions = torch.multinomial(
                weights, context_count, replacement=False
            )
            context = remaining_tensor[sampled_positions].tolist()
        return sorted(core + context)

    def _sample_evaluation_indices(self, wsi_id, total_patches, item_index):
        """Select one reproducible validation/test bag without changing global RNG state."""
        split_offset = 0 if self.data_type == 'validation' else 10_000_019
        seed = self.pcps_eval_sampling_seed + split_offset + int(item_index) * 104729
        if self.pcps_sampling_mode == 'uniform':
            return sorted(random.Random(seed).sample(range(total_patches), min(int(self.pcps_total_count), total_patches)))
        if self.pcps_eval_mode == 'topk':
            count = self.pcps_total_count if self.pcps_total_count else self.pcps_selected_count
            return sorted(
                int(index) for index in
                self.pcps_selection[wsi_id]['selected_patch_indices'][:min(int(count), total_patches)]
            )

        if self.pcps_sampling_mode == 'legacy':
            ranked = self.pcps_selection[wsi_id]['selected_patch_indices']
            core = [int(index) for index in ranked[:self.pcps_selected_count]]
            core_set = set(core)
            remaining = [index for index in range(total_patches) if index not in core_set]
            context = random.Random(seed).sample(
                remaining, min(self.pcps_random_count, len(remaining))
            )
            return sorted(core + context)

        ranking, scores = self._full_pcps_ranking_and_scores(wsi_id, total_patches)
        generator = torch.Generator(device='cpu').manual_seed(seed)
        total_count = min(int(self.pcps_total_count), total_patches)
        if self.pcps_sampling_mode == 'gumbel_top_m':
            if total_count == total_patches:
                return list(range(total_patches))
            eps = 1e-6
            posterior_logits = torch.logit(scores.clamp(eps, 1.0 - eps), eps=eps)
            standardized_evidence = (
                posterior_logits - posterior_logits.mean()
            ) / posterior_logits.std(unbiased=False).clamp_min(eps)
            uniform = torch.rand(
                standardized_evidence.shape, generator=generator
            ).clamp_(eps, 1.0 - eps)
            gumbel = -torch.log(-torch.log(uniform))
            keys = self.pcps_distribution_concentration * standardized_evidence + gumbel
            return sorted(torch.topk(keys, total_count).indices.tolist())

        core_count = min(
            total_count,
            int(round(total_count * self.pcps_evidence_concentration)),
        )
        core = ranking[:core_count]
        core_set = set(core)
        remaining = [index for index in range(total_patches) if index not in core_set]
        context_count = min(total_count - core_count, len(remaining))
        if context_count == 0:
            context = []
        elif np.isinf(self.pcps_context_temperature):
            context = random.Random(seed).sample(remaining, context_count)
        else:
            remaining_tensor = torch.as_tensor(remaining, dtype=torch.long)
            context_scores = scores[remaining_tensor]
            context_scores = (
                context_scores - context_scores.mean()
            ) / context_scores.std(unbiased=False).clamp_min(1e-6)
            weights = torch.softmax(
                context_scores / self.pcps_context_temperature, dim=0
            )
            sampled_positions = torch.multinomial(
                weights, context_count, replacement=False, generator=generator
            )
            context = remaining_tensor[sampled_positions].tolist()
        return sorted(core + context)

    def read_dataset_csv(self):
        df = pd.read_csv(self.dataset_csv_path, header=0)
        if self.data_type in ['test']:
            df = df[df['is_test'] > 0]
        else:
            df = df[df['is_test'] == 0]
        return df

    def read_cv_dataset_csv(self):
        df = pd.read_csv(self.dataset_csv_path, header=0)
        if self.data_type in ['validation']:
            df = df[df['fold'] == self.val_fold_id]
        elif self.data_type in ['test']:
            df = df[df['fold'] < 0]
        else:
            df = df[df['fold'] > 0]
            df = df[df['fold'] != self.val_fold_id]
        return df

    def __len__(self):
        return len(self.wsi_list)

    def _load_specific_patches(self, wsi_id, indices):
        """Load patches directly from scatter PNG files by filename index
        
        Filename: {scatter_png_dir}/{wsi_id}/{wsi_id}_patch_{idx}.png
        Compared to computing offsets from strips, this method guarantees exact index correspondence.
        """
        if not indices:
            return np.empty((0, 256, 256, 3), dtype=np.uint8)
        
        result = []
        for idx in sorted(indices):
            png_path = os.path.join(
                self.scatter_png_dir, wsi_id,
                f'{wsi_id}_patch_{idx}.png'
            )
            if not os.path.exists(png_path):
                raise FileNotFoundError(
                    f'{png_path} is missing; patch indices would become misaligned'
                )
            patch = read_rgb_img(png_path)
            result.append(patch)
        
        return np.stack(result)

    def _load_wsi_all_patches(self, wsi_id, len_img):
        # scatter_png_dir available -> load all patches from scatter PNGs (same source as A+B sampling)
        if self.scatter_png_dir is not None:
            folder = os.path.join(self.scatter_png_dir, wsi_id)
            coordinate_path = os.path.join(os.path.dirname(self.scatter_png_dir), 'coordinates', 'patches', f'{wsi_id}.h5')
            if os.path.isfile(coordinate_path):
                import h5py
                with h5py.File(coordinate_path, 'r') as handle:
                    expected_patches = len(handle['coords'])
            else:
                import glob
                expected_patches = len(glob.glob(os.path.join(folder, f'{wsi_id}_patch_*.png')))
            if expected_patches == 0:
                raise ValueError(f'No PNG patches for {wsi_id}')
            from dataset.scatter_png import load_scatter_patches
            return load_scatter_patches(self.scatter_png_dir, wsi_id, range(expected_patches))
        else:
            # fallback: load from strips
            tiles = []
            for i in range(len_img):
                tile = read_rgb_img(os.path.join(self.dataset_root, '%s_%d%s' % (wsi_id, i, self.data_ext)))
                assert len(tile.shape) == 3
                h, w, c = tile.shape
                tile = tile.reshape(h // w, w, w, c)
                tiles.append(tile)
            tiles_np = np.concatenate(tiles, axis=0)
        tiles = K.utils.image_to_tensor(tiles_np)
        tiles = K.enhance.normalize(tiles, torch.tensor(0.), torch.tensor(255.))
        return tiles

    def __getitem__(self, i):
        row = self.wsi_list.iloc[i]
        wsi_id = row['wsi_id']
        label = row['label']
        len_img = row['len_img']

        if self.pcps_sampling_mode == 'uniform':
            # Old len_img can count merged image strips, not individual patches.
            # Enumerate the actual exported PNGs and require contiguous indices.
            if wsi_id not in self._uniform_patch_counts:
                folder = os.path.join(self.scatter_png_dir, wsi_id)
                prefix = f'{wsi_id}_patch_'
                indices = {int(name[len(prefix):-4]) for name in os.listdir(folder)
                           if name.startswith(prefix) and name.endswith('.png')}
                if not indices or indices != set(range(len(indices))):
                    raise ValueError(f'{wsi_id}: non-contiguous exported patch indices')
                coordinate_path = os.path.join(os.path.dirname(self.scatter_png_dir),
                                               'coordinates', 'patches', f'{wsi_id}.h5')
                if os.path.isfile(coordinate_path):
                    import h5py
                    with h5py.File(coordinate_path, 'r') as handle:
                        expected = len(handle['coords'])
                    if len(indices) != expected:
                        raise ValueError(f'{wsi_id}: {len(indices)} PNGs but {expected} coordinate patches')
                self._uniform_patch_counts[wsi_id] = len(indices)
            total_patches = self._uniform_patch_counts[wsi_id]
            selected = (self._sample_training_indices(wsi_id, total_patches)
                        if self.data_type == 'train'
                        else self._sample_evaluation_indices(wsi_id, total_patches, i))
            tiles_np = self._load_specific_patches(wsi_id, selected)
            tiles = K.utils.image_to_tensor(tiles_np)
            tiles = K.enhance.normalize(tiles, torch.tensor(0.), torch.tensor(255.))
            if self.drop_out > 0:
                raise ValueError('uniform matched-budget ablations require patch dropout=0')
            return tiles, label, torch.full((len(tiles),), -1, dtype=torch.long), wsi_id

        total_patches = (
            self.pcps_selection[wsi_id]['total_patches']
            if self.pcps_selection is not None and wsi_id in self.pcps_selection
            else int(len_img) if self.scatter_png_dir else 0
        )
        if self.pcps_patch_diagnostics is not None and wsi_id in self.pcps_patch_diagnostics:
            all_prototype_ids = self.pcps_patch_diagnostics[wsi_id][
                'prototype_ids'
            ].long()
        else:
            all_prototype_ids = torch.full((total_patches,), -1, dtype=torch.long)

        if self.pcps_selection is None or wsi_id not in self.pcps_selection:
            tiles = self._load_wsi_all_patches(wsi_id, len_img)
            return tiles, label, torch.full((len(tiles),), -1, dtype=torch.long), wsi_id

        if self.data_type != 'train' and (
            self.pcps_eval_mode == 'full'
            or self.pcps_selection is None
            or wsi_id not in self.pcps_selection
        ):
            return (
                self._load_wsi_all_patches(wsi_id, len_img),
                label,
                all_prototype_ids,
                wsi_id,
            )

        wsi_selection = self.pcps_selection[wsi_id]
        total_patches = wsi_selection['total_patches']
        all_indices = (
            self._sample_training_indices(wsi_id, total_patches)
            if self.data_type == 'train'
            else self._sample_evaluation_indices(wsi_id, total_patches, i)
        )
        tiles_np = self._load_specific_patches(wsi_id, all_indices)

        tiles = K.utils.image_to_tensor(tiles_np)
        tiles = K.enhance.normalize(tiles, torch.tensor(0.), torch.tensor(255.))
        selected_prototype_ids = all_prototype_ids[all_indices]

        if self.drop_out > 0:
            perm = torch.randperm(tiles.size(0))
            idx = perm[:int((1 - self.drop_out) * tiles.size(0))]
            tiles = tiles[idx]
            selected_prototype_ids = selected_prototype_ids[idx]

        return (
            tiles, label, selected_prototype_ids, wsi_id
        )

    def get_label(self, idx):
        label = self.wsi_list.iloc[idx]['label']
        return int(label)

    def get_weights_of_class(self):
        labels = self.wsi_list['label']
        unique, counts = np.unique(labels, return_counts=True)
        label_cnt = list(zip(unique, counts))
        label_cnt.sort(key=lambda x: x[0])
        weight_arr = np.array([x[1] for x in label_cnt], dtype=float)
        weight_arr = np.max(weight_arr) / weight_arr
        return torch.from_numpy(weight_arr.astype(np.float32))


class PatchWsiDataModule(pl.LightningDataModule):
    def __init__(self, dataset_root, dataset_csv, val_fold=-1, data_ext='.jpg', classes_names=None, num_workers=2,
                 num_workers_eval=0, drop_out=0., shuffule_train=True, weighted_sample=False,
                 pcps_selection_path=None, pcps_selected_count=384, pcps_random_count=128,
                 pcps_sampling_mode='legacy', pcps_total_count=None,
                 pcps_evidence_concentration=0.5,
                 pcps_context_temperature=float('inf'),
                 pcps_distribution_concentration=1.0,
                 pcps_eval_mode='full', pcps_eval_sampling_seed=None,
                 scatter_png_dir=None):
        super().__init__()
        if classes_names is None:
            self.CLASSES = None
            self.CLASS_NAMES = None
        else:
            self.CLASSES = classes_names[0]
            self.CLASS_NAMES = classes_names[1]

        self.dataset_root = dataset_root
        self.dataset_csv = dataset_csv
        self.data_ext = data_ext

        self.val_fold = val_fold
        self.num_workers = num_workers
        self.num_workers_eval = num_workers_eval
        self.drop_out = drop_out

        self.dataset_train = None
        self.dataset_val = None
        self.dataset_test = None
        self.shuffule_train = shuffule_train
        self.weighted_sample = weighted_sample

        self.pcps_selection_path = pcps_selection_path
        self.pcps_selected_count = pcps_selected_count
        self.pcps_random_count = pcps_random_count
        self.pcps_sampling_mode = pcps_sampling_mode
        self.pcps_total_count = pcps_total_count
        self.pcps_evidence_concentration = pcps_evidence_concentration
        self.pcps_context_temperature = pcps_context_temperature
        self.pcps_distribution_concentration = pcps_distribution_concentration
        self.pcps_eval_mode = pcps_eval_mode
        self.pcps_eval_sampling_seed = pcps_eval_sampling_seed
        self.scatter_png_dir = scatter_png_dir

    def setup(self, stage=None):
        if self.dataset_train is None:
            if self.pcps_selection_path is not None or self.scatter_png_dir is not None:
                dataset_cls = PCPSSelectedWSIDataset
                extra_kwargs = {
                    'pcps_selection_path': self.pcps_selection_path,
                    'pcps_selected_count': self.pcps_selected_count,
                    'pcps_random_count': self.pcps_random_count,
                    'pcps_sampling_mode': self.pcps_sampling_mode,
                    'pcps_total_count': self.pcps_total_count,
                    'pcps_evidence_concentration': self.pcps_evidence_concentration,
                    'pcps_context_temperature': self.pcps_context_temperature,
                    'pcps_distribution_concentration': self.pcps_distribution_concentration,
                    'pcps_eval_mode': self.pcps_eval_mode,
                    'pcps_eval_sampling_seed': self.pcps_eval_sampling_seed,
                    'scatter_png_dir': self.scatter_png_dir,
                }
            else:
                dataset_cls = MergePatchWsiDataset
                extra_kwargs = {}

            self.dataset_train = dataset_cls(self.dataset_root, self.dataset_csv, 'train',
                                              data_ext=self.data_ext, val_fold_id=self.val_fold,
                                              classes_names=[self.CLASSES, self.CLASS_NAMES],
                                              drop_out=self.drop_out, **extra_kwargs)
            self.dataset_val = dataset_cls(self.dataset_root, self.dataset_csv, 'validation',
                                            data_ext=self.data_ext, val_fold_id=self.val_fold,
                                            classes_names=[self.CLASSES, self.CLASS_NAMES],
                                            drop_out=0., **extra_kwargs)
            self.dataset_test = dataset_cls(self.dataset_root, self.dataset_csv, 'test',
                                             data_ext=self.data_ext, val_fold_id=self.val_fold,
                                             classes_names=[self.CLASSES, self.CLASS_NAMES],
                                             drop_out=0., **extra_kwargs)

    def train_dataloader(self):
        from torch.utils.data import WeightedRandomSampler
        import collections

        if getattr(self, 'weighted_sample', False):
            labels = [self.dataset_train.get_label(i) for i in range(len(self.dataset_train))]
            class_counts = collections.Counter(labels)
            weights = {cls: 1.0 / count for cls, count in class_counts.items()}
            samples_weight = [weights[label] for label in labels]
            samples_weight = torch.DoubleTensor(samples_weight)
            sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
            print(f'\n[INFO] Weighted Sample Enabled! Class counts: {class_counts}')
            return DataLoader(self.dataset_train, batch_size=1,
                              sampler=sampler, num_workers=self.num_workers,
                              drop_last=False, pin_memory=False)
        else:
            return DataLoader(self.dataset_train, batch_size=1, shuffle=True,
                              num_workers=self.num_workers, drop_last=False, pin_memory=False)

    def val_dataloader(self):
        if self.val_fold >= 0:
            return DataLoader(self.dataset_val, batch_size=1, shuffle=False,
                              num_workers=self.num_workers_eval,
                              drop_last=False, pin_memory=False)
        else:
            return DataLoader(self.dataset_val, batch_size=1, shuffle=False, num_workers=self.num_workers_eval,
                              drop_last=False, pin_memory=False)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=1, shuffle=False, num_workers=self.num_workers_eval,
                          drop_last=False, pin_memory=False)
