import json
import os
import time

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn

from .forward_fn import get_classifer_fuc


from .streaming import NativePatchMixin


class MilE2EModule(NativePatchMixin, pl.LightningModule):
    def __init__(self, backbone, classifier, loss, metrics, cus_transforms, args, num_classes,
                 classifier_type, **kwargs):
        super().__init__()
        self.automatic_optimization = False
        self.args = args
        self.kwargs = kwargs
        self.backbone = backbone
        self.freeze_backbone = args.transfer_type in ["frozen", "mil"]
        self.classifier = classifier
        self.classifier_type = classifier_type
        self.classifer_fuc = get_classifer_fuc(classifier_type)
        self.loss = loss
        self.num_classes = num_classes
        self.lr = args.lr
        self.batch_size_train = args.batch_size_train
        self.batch_size_eval = args.batch_size_eval
        self.accumulate_grad_batches = args.accumulate_grad_batches
        self.train_metrics = metrics.clone(postfix='/train')
        self.valid_metrics = nn.ModuleList([
            metrics.clone(postfix='/val'), metrics.clone(postfix='/test')
        ])
        self.test_metrics = metrics.clone(prefix='final_test/')
        self.test_slide_diagnostics = []

        if cus_transforms is None:
            self.transforms_train, self.transforms_eval = None, None
        elif isinstance(cus_transforms, (list, tuple)):
            self.transforms_train, self.transforms_eval = cus_transforms
        else:
            self.transforms_train, self.transforms_eval = cus_transforms, cus_transforms
        self.save_hyperparameters("args")

    def _unpack_batch(self, batch):
        if len(batch) == 4:
            img, label, prototype_ids, wsi_id = batch
            return img, label, prototype_ids, wsi_id
        if len(batch) == 3:
            img, label, prototype_ids = batch
            return img, label, prototype_ids, None
        img, label = batch
        return img, label, None, None

    def forward(self, data, label=None, train=False, prototype_ids=None, loss_scale=1.0):
        self._prepare_wsi_prompt_context(data)
        raw_features = self.backbone_forward(data)

        if train and not self.freeze_backbone:
            raw_features = raw_features.detach().requires_grad_(True)

        features = raw_features
        diffusion_time = None
        if getattr(self.args, 'enable_fprd', False) and hasattr(
            self.backbone, 'apply_fidelity_preserving_residual_diffusion'
        ):
            features = self.backbone.apply_fidelity_preserving_residual_diffusion(raw_features)
            diffusion_time = getattr(self.backbone, 'last_diffusion_time', None)
            if train and not self.freeze_backbone:
                features.retain_grad()

        bag_prediction, loss, y_prob = self.classifier_forward(features, label)

        if train:
            self.manual_backward(loss * loss_scale)
            if not self.freeze_backbone:
                self._log_gradient_distribution('head_side', features.grad)
                self._log_gradient_distribution('backbone_side', raw_features.grad)
                self.backbone_backward(data, raw_features)
            self._log_scpm_fprd_diagnostics(diffusion_time)
        return bag_prediction, loss, y_prob

    def _prepare_wsi_prompt_context(self, data):
        """Build one sampled-WSI token context and reuse it for every chunk/recompute."""
        if not getattr(self.backbone, 'scpm_enabled', True):
            return
        scpm = getattr(self.backbone, 'scpm', None)
        compute_context = getattr(
            self.backbone, 'compute_wsi_initial_token_context', None
        )
        if scpm is None or compute_context is None:
            return
        chunk_size = self.batch_size_train if self.training else self.batch_size_eval
        context = compute_context(self.split_tensor(data, chunk_size))
        scpm.set_wsi_initial_context(context)

    def _log_scpm_fprd_diagnostics(self, diffusion_time):
        if diffusion_time is not None:
            self.log('train/fprd_diffusion_time', diffusion_time.detach(),
                     on_step=True, on_epoch=True, sync_dist=True)
        scpm = getattr(self.backbone, 'scpm', None)
        if scpm is not None and hasattr(scpm, 'last_residual_norms'):
            norms = scpm.last_residual_norms.float()
            cosine = scpm.last_adjacent_residual_cosine.float()
            self.log('train/scpm_residual_norm_mean', norms.mean(),
                     on_step=True, on_epoch=True, sync_dist=True)
            self.log('train/scpm_residual_norm_std', norms.std(unbiased=False),
                     on_step=True, on_epoch=True, sync_dist=True)
            if cosine.numel() > 0:
                self.log('train/scpm_adjacent_cosine_mean', cosine.mean(),
                         on_step=True, on_epoch=True, sync_dist=True)
            self.log('train/scpm_transition_orthogonality_error',
                     scpm.last_transition_orthogonality_error,
                     on_step=True, on_epoch=True, sync_dist=True)
            if hasattr(scpm, 'last_state_smoothness'):
                self.log('train/scpm_flow_smoothness', scpm.last_state_smoothness,
                         on_step=True, on_epoch=True, sync_dist=True)
            if hasattr(scpm, 'last_correction_mean') and torch.isfinite(scpm.last_correction_mean):
                self.log('train/scpm_correction_mean', scpm.last_correction_mean,
                         on_step=True, on_epoch=True, sync_dist=True)
        if hasattr(self.backbone, 'last_diffusion_delta'):
            self.log('train/fprd_delta_mean',
                     self.backbone.last_diffusion_delta.float().mean(),
                     on_step=True, on_epoch=True, sync_dist=True)
        if hasattr(self.backbone, 'last_fprd_beta'):
            self.log('train/fprd_beta', self.backbone.last_fprd_beta.float(),
                     on_step=True, on_epoch=True, sync_dist=True)
        if hasattr(self.backbone, 'last_fprd_neighbor_similarity'):
            self.log('train/fprd_neighbor_similarity',
                     self.backbone.last_fprd_neighbor_similarity.float(),
                     on_step=True, on_epoch=True, sync_dist=True)

    def _log_gradient_distribution(self, name, gradient):
        if gradient is None:
            return
        magnitude = gradient.detach().float().norm(dim=1)
        total = magnitude.sum().clamp_min(1e-12)
        probability = magnitude / total
        entropy = -(probability * probability.clamp_min(1e-12).log()).sum()
        effective_ratio = entropy.exp() / max(1, magnitude.numel())
        top_count = max(1, int(0.1 * magnitude.numel()))
        top_mass = magnitude.topk(top_count).values.sum() / total
        relative_threshold = magnitude.max() * 1e-6
        coverage = (magnitude > relative_threshold).float().mean()
        self.log(f'train/gradient_{name}_effective_ratio', effective_ratio,
                 on_step=True, on_epoch=True, sync_dist=True)
        self.log(f'train/gradient_{name}_top10_mass', top_mass,
                 on_step=True, on_epoch=True, sync_dist=True)
        self.log(f'train/gradient_{name}_coverage', coverage,
                 on_step=True, on_epoch=True, sync_dist=True)

    def backbone_forward(self, data):
        features = []
        self._last_scpm_chunk_diagnostics = []
        chunk_size = self.batch_size_train if self.training else self.batch_size_eval
        with torch.no_grad():
            for data_i in self.split_tensor(data, chunk_size):
                features.append(self.backbone(data_i))
                scpm = getattr(self.backbone, 'scpm', None)
                if scpm is not None and hasattr(scpm, 'last_residual_vectors'):
                    self._last_scpm_chunk_diagnostics.append({
                        'num_images': int(data_i.shape[0]),
                        'residual_vectors': scpm.last_residual_vectors[0].cpu(),
                        'state_vectors': scpm.last_state_vectors[0].cpu(),
                        'residual_norms': scpm.last_residual_norms[0].cpu(),
                        'adjacent_cosine': scpm.last_adjacent_residual_cosine[0].cpu(),
                    })
        return torch.cat(features, dim=0)

    def backbone_backward(self, data, features):
        feature_grads = features.grad
        if feature_grads is None:
            raise RuntimeError('Missing feature gradient before backbone recomputation.')
        data_chunks = self.split_tensor(data, self.batch_size_train)
        grad_chunks = self.split_tensor(feature_grads, self.batch_size_train)
        for data_i, grad_i in zip(data_chunks, grad_chunks):
            recomputed_features = self.backbone(data_i)
            recomputed_features.backward(grad_i)

    def classifier_forward(self, data, label=None):
        return self.classifer_fuc(
            data, self.classifier, self.loss, self.num_classes, label=label
        )

    def split_tensor(self, data, batch_size):
        num_chunks = int(np.ceil(data.shape[0] / batch_size))
        return torch.chunk(data, num_chunks, dim=0)


    def training_step(self, batch, batch_idx):
        img, label, prototype_ids, _ = self._unpack_batch(batch)
        img = self.augment_data(img, train=True)

        accumulation = max(1, self.accumulate_grad_batches)
        num_batches = int(self.trainer.num_training_batches)
        group_start = (batch_idx // accumulation) * accumulation
        group_size = min(accumulation, num_batches - group_start)
        should_step = ((batch_idx + 1) % accumulation == 0) or (
            batch_idx + 1 == num_batches
        )

        y, loss, y_prob = self(
            img, label=label, train=True,
            prototype_ids=prototype_ids,
            loss_scale=1.0 / group_size,
        )

        optimizer = self.optimizers()
        if should_step:
            optimizer.step()
            optimizer.zero_grad()

        self.log('Loss/train', loss, on_step=True, on_epoch=True, sync_dist=True)
        self.train_metrics(y_prob, label)
        self.log_dict(
            self.train_metrics, on_step=False, on_epoch=True, sync_dist=True
        )
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        img, label, prototype_ids, _ = self._unpack_batch(batch)
        img = self.augment_data(img)
        y, loss, y_prob = self(img, label=label, prototype_ids=prototype_ids)
        if not self.trainer.sanity_checking:
            prefix = get_prefix_from_val_id(dataloader_idx)
            metrics_idx = dataloader_idx if dataloader_idx is not None else 0
            self.log(
                'Loss/%s' % prefix, loss, on_step=False, on_epoch=True,
                sync_dist=True, add_dataloader_idx=False
            )
            self.valid_metrics[metrics_idx](y_prob, label)
            self.log_dict(
                self.valid_metrics[metrics_idx], on_step=False, on_epoch=True,
                sync_dist=True, add_dataloader_idx=False
            )
        return loss

    def test_step(self, batch, batch_idx):
        img, label, prototype_ids, wsi_id = self._unpack_batch(batch)
        img = self.augment_data(img)
        y, loss, y_prob = self(img, label=label, prototype_ids=prototype_ids)
        self._collect_test_diagnostic(wsi_id, label, y_prob)
        self.log(
            'Loss/final_test', loss, on_step=False, on_epoch=True, sync_dist=True
        )
        self.test_metrics(y_prob, label)
        self.log_dict(
            self.test_metrics, on_step=False, on_epoch=True, sync_dist=True
        )
        return self.test_metrics

    def on_test_epoch_start(self):
        self.test_slide_diagnostics = []

    def _collect_test_diagnostic(self, wsi_id, label, y_prob):
        if wsi_id is None:
            slide_name = f'test_{len(self.test_slide_diagnostics)}'
        elif isinstance(wsi_id, (list, tuple)):
            slide_name = str(wsi_id[0])
        else:
            slide_name = str(wsi_id)
        scpm = getattr(self.backbone, 'scpm', None)
        record = {
            'wsi_id': slide_name,
            'label': int(label.reshape(-1)[0].detach().cpu()),
            'probability': y_prob.detach().float().cpu().reshape(-1).tolist(),
            'fprd_diffusion_time': float(self.backbone.last_diffusion_time.detach().float().cpu()),
        }
        if scpm is not None and hasattr(scpm, 'last_residual_norms'):
            chunks = getattr(self, '_last_scpm_chunk_diagnostics', [])
            if chunks:
                weights = torch.tensor(
                    [chunk['num_images'] for chunk in chunks], dtype=torch.float32
                )
                weights = weights / weights.sum()
                residual_vectors = torch.stack(
                    [chunk['residual_vectors'] for chunk in chunks]
                )
                state_vectors = torch.stack(
                    [chunk['state_vectors'] for chunk in chunks]
                )
                slide_residual = (residual_vectors * weights[:, None, None]).sum(0)
                slide_state = (state_vectors * weights[:, None, None]).sum(0)
                record['scpm_context_scope'] = (
                    'sampled_wsi_initialization_then_chunk_correction'
                )
                record['scpm_chunk_sizes'] = [chunk['num_images'] for chunk in chunks]
                record['scpm_chunk_residual_norms'] = [
                    chunk['residual_norms'].tolist() for chunk in chunks
                ]
                record['scpm_chunk_state_vectors'] = state_vectors.tolist()
                record['scpm_slide_mean_residual_vectors'] = slide_residual.tolist()
                record['scpm_slide_mean_state_vectors'] = slide_state.tolist()
                record['scpm_residual_norms'] = slide_residual.norm(dim=1).tolist()
                if slide_residual.shape[0] > 1:
                    record['scpm_adjacent_cosine'] = torch.nn.functional.cosine_similarity(
                        slide_residual[1:], slide_residual[:-1], dim=1
                    ).tolist()
                else:
                    record['scpm_adjacent_cosine'] = []
            else:
                record['scpm_context_scope'] = 'last_chunk_legacy_fallback'
                record['scpm_residual_norms'] = scpm.last_residual_norms[0].cpu().tolist()
                record['scpm_adjacent_cosine'] = scpm.last_adjacent_residual_cosine[0].cpu().tolist()
            record['scpm_transition_orthogonality_error'] = float(
                scpm.last_transition_orthogonality_error.cpu()
            )
        if hasattr(self.backbone, 'last_diffusion_delta'):
            delta = self.backbone.last_diffusion_delta.float().cpu()
            record['fprd_transport_delta'] = delta.tolist()
            record['fprd_transport_delta_mean'] = float(delta.mean())
            record['fprd_transport_delta_max'] = float(delta.max())
        self.test_slide_diagnostics.append(record)

    def on_test_epoch_end(self):
        if not self.trainer.is_global_zero or not self.test_slide_diagnostics:
            return
        output_dir = os.path.join(
            self.args.output_dir, self.args.run_name, self.args.tag, 'diagnostics'
        )
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'test_slide_diagnostics.json')
        with open(output_path, 'w') as handle:
            json.dump(self.test_slide_diagnostics, handle)
        print(f'Saved PSPT test diagnostics: {output_path}')

    def on_train_epoch_start(self):
        self._epoch_started = time.monotonic()

    def on_train_epoch_end(self):
        directory = os.path.join(self.args.output_dir, self.args.run_name, self.args.tag)
        os.makedirs(directory, exist_ok=True)
        record = {'epoch': self.current_epoch,
                  'seconds_train_plus_validation': time.monotonic() - self._epoch_started,
                  'backbone_trainable_parameters': sum(p.numel() for p in self.backbone.parameters() if p.requires_grad),
                  'head_trainable_parameters': sum(p.numel() for p in self.classifier.parameters() if p.requires_grad),
                  'scpm_enabled': getattr(self.backbone, 'scpm_enabled', True),
                  'fprd_enabled': self.args.enable_fprd}
        with open(os.path.join(directory, 'epoch_resources.jsonl'), 'a') as handle:
            handle.write(json.dumps(record) + '\n')
        self.lr_schedulers().step()

    def configure_optimizers(self):
        parameters = [
            {
                'params': filter(lambda p: p.requires_grad, self.backbone.parameters()),
                'lr': self.lr * self.args.lr_factor,
            },
            {'params': filter(lambda p: p.requires_grad, self.classifier.parameters())},
        ]
        optimizer_cls = torch.optim.Adam if self.args.adam else torch.optim.AdamW
        optimizer_kwargs = {'betas': (0.5, 0.9)} if self.args.adam else {}
        optimizer = optimizer_cls(
            parameters, lr=self.lr, weight_decay=self.args.weight_decay,
            **optimizer_kwargs
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.args.epochs, eta_min=5e-6
        )
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}


def get_prefix_from_val_id(dataloader_idx):
    if dataloader_idx is None or dataloader_idx == 0:
        return 'val'
    if dataloader_idx == 1:
        return 'test'
    raise NotImplementedError
