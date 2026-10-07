import pandas as pd
import torch

from .merge_patch_wsi_dataset import PCPSSelectedWSIDataset, PatchWsiDataModule


class PCPSSurvivalWSIDataset(PCPSSelectedWSIDataset):


    def __getitem__(self, index):
        tiles, _, prototype_ids, wsi_id = super().__getitem__(index)
        row = self.wsi_list.iloc[index]
        target = {
            "time_bin": torch.tensor(int(row["time_bin"]), dtype=torch.long),
            "event": torch.tensor(int(row["event"]), dtype=torch.long),
            "censorship": torch.tensor(int(row["censorship"]), dtype=torch.long),
            "time": torch.tensor(float(row["survival_months"]), dtype=torch.float32),
            "case_id": str(row["case_id"]),
            "wsi_id": str(wsi_id),
        }
        return tiles, target, prototype_ids, wsi_id


class SurvivalPatchWsiDataModule(PatchWsiDataModule):
    def setup(self, stage=None):
        if self.dataset_train is not None:
            return
        extra_kwargs = {
            "pcps_selection_path": self.pcps_selection_path,
            "pcps_selected_count": self.pcps_selected_count,
            "pcps_random_count": self.pcps_random_count,
            "pcps_sampling_mode": self.pcps_sampling_mode,
            "pcps_total_count": self.pcps_total_count,
            "pcps_evidence_concentration": self.pcps_evidence_concentration,
            "pcps_context_temperature": self.pcps_context_temperature,
            "pcps_distribution_concentration": self.pcps_distribution_concentration,
            "pcps_eval_mode": self.pcps_eval_mode,
            "pcps_eval_sampling_seed": self.pcps_eval_sampling_seed,
            "scatter_png_dir": self.scatter_png_dir,
        }
        self.dataset_train = PCPSSurvivalWSIDataset(
            self.dataset_root, self.dataset_csv, "train",
            data_ext=self.data_ext, val_fold_id=self.val_fold,
            classes_names=[self.CLASSES, self.CLASS_NAMES],
            drop_out=self.drop_out, **extra_kwargs,
        )
        self.dataset_val = PCPSSurvivalWSIDataset(
            self.dataset_root, self.dataset_csv, "validation",
            data_ext=self.data_ext, val_fold_id=self.val_fold,
            classes_names=[self.CLASSES, self.CLASS_NAMES],
            drop_out=0.0, **extra_kwargs,
        )
        self.dataset_test = PCPSSurvivalWSIDataset(
            self.dataset_root, self.dataset_csv, "test",
            data_ext=self.data_ext, val_fold_id=self.val_fold,
            classes_names=[self.CLASSES, self.CLASS_NAMES],
            drop_out=0.0, **extra_kwargs,
        )


def summarize_survival_split(csv_path):
    frame = pd.read_csv(csv_path)
    rows = []
    for split, group in frame.groupby("split"):
        cases = group.drop_duplicates("case_id")
        rows.append({
            "split": split,
            "slides": int(len(group)),
            "cases": int(cases.shape[0]),
            "events": int(cases["event"].sum()),
            "mean_patches": float(group["len_img"].mean()),
            "median_patches": float(group["len_img"].median()),
        })
    return pd.DataFrame(rows)
