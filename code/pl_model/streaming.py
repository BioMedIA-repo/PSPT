import torch


def _move(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _move(item, device) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_move(item, device) for item in value)
    return value


class NativePatchMixin:
    @property
    def streaming_patch_threshold(self):


        return float('inf')

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        return _move(batch, device)

    def augment_data(self, data, train=False):
        if data.ndim == 5:
            data = data.squeeze(0)
        if data.ndim != 4:
            raise ValueError(f'Expected [N,3,H,W] patch bag, got {data.shape}')
        transform = self.transforms_train if train else self.transforms_eval
        if transform is not None:
            with torch.no_grad():
                for start in range(0, data.shape[0], self.batch_size_eval):
                    end = start + self.batch_size_eval
                    data[start:end] = transform(data[start:end])
        return data
