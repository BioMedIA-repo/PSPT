import argparse
import os
import pandas as pd
import numpy as np
import pytorch_lightning as pl
import torch

# Compatibility shim for PyTorch 2.5/2.6 to allow loading Lightning-saved hyperparameters
if hasattr(torch.serialization, 'add_safe_globals'):
    torch.serialization.add_safe_globals([argparse.Namespace])
torch.set_float32_matmul_precision('medium')

from pytorch_lightning.loggers import TensorBoardLogger
from torch import nn
from torchmetrics import MetricCollection, Accuracy, AUROC
from torchmetrics import F1Score as F1
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

import network.get_network
from dataset import get_class_names
from dataset.merge_patch_wsi_dataset import PatchWsiDataModule
from options import get_arguments, get_arguments_additional
from pl_model.mil_e2e_trainer import MilE2EModule
from pl_model.runtime_checks import precision_options, verify_optimizer_updates
from utils import save_parameters, switch_dim

import kornia.augmentation as K


def get_transforms(args):
    mean = args.data_mean if args.data_mean is not None else [0.485, 0.456, 0.406]
    std = args.data_std if args.data_std is not None else [0.229, 0.224, 0.225]
    if args.data_norm:
        transforms_train = nn.Sequential(
            K.Normalize(mean=torch.tensor(mean), std=torch.tensor(std))
        )
        transforms_eval = nn.Sequential(
            K.Normalize(mean=torch.tensor(mean), std=torch.tensor(std))
        )
    else:
        transforms_train = nn.Sequential()
        transforms_eval = nn.Sequential()
    return transforms_train, transforms_eval


def get_metric(num_classes, task):
    metrics = MetricCollection({
        "Accuracy": Accuracy(num_classes=num_classes, task=task),
        "BA": Accuracy(num_classes=num_classes, average="macro", task=task),
        "F1": F1(num_classes=num_classes, average="macro", task=task),
        "AUROC": AUROC(num_classes=num_classes, task=task),
    })
    return metrics


def get_network(args):
    if args.network.startswith("uni"):
        from network.get_network import get_uni_peft_model
        backbone = get_uni_peft_model(args)
        num_fts = backbone.num_features
    elif args.network.startswith("conch"):
        from network.get_network import get_conch_peft_model
        backbone = get_conch_peft_model(args)
        num_fts = backbone.num_features
    elif args.network.startswith("plip"):
        from network.get_network import get_plip_peft_model
        backbone = get_plip_peft_model(args)
        num_fts = backbone.num_features
    else:
        raise NotImplementedError(f"Unsupported network: {args.network}. Supported: uni, conch, plip")
    return backbone, num_fts


def get_loss_weight(args, data_module):
    if args.loss_weight is not None:
        loss_weight = args.loss_weight
    elif args.auto_loss_weight:
        data_module.setup()
        loss_weight = data_module.dataset_train.get_weights_of_class()
    else:
        loss_weight = None
    if loss_weight is not None:
        print("Using loss weight:", loss_weight)
        loss_weight = torch.Tensor(loss_weight)
    return loss_weight


def get_mil_network(mil_type, num_fts, num_classes, args, loss_weight=None):
    if mil_type in ("clam_sb", "clam_mb"):
        from network.model_clam import CLAM_SB, CLAM_MB
        CLAM = CLAM_SB if mil_type == "clam_sb" else CLAM_MB
        clam_model_dict = {"dropout": getattr(args, "clam_dropout", True), 'n_classes': num_classes, 'subtyping': getattr(args, "clam_subtyping", True), "size": args.clam_size,
                           'k_sample': getattr(args, 'clam_k_sample', 8), 'bag_weight': getattr(args, 'clam_bag_weight', 0.7)}
        classifier_model = CLAM(**clam_model_dict, instance_loss_fn=getattr(args, 'clam_instance_loss', 'svm'))
        loss = nn.CrossEntropyLoss(weight=loss_weight)
    else:
        raise NotImplementedError(f"Unsupported MIL model: {mil_type}. Supported: clam_sb, clam_mb")
    return classifier_model, loss


def get_model(args, backbone, num_fts, num_classes, loss_weight=None):
    from pl_model.forward_fn import model_to_classifier_type
    task = "multiclass"
    classifier_model, loss = get_mil_network(args.model, num_fts, num_classes, args, loss_weight=loss_weight)
    classifier_type = model_to_classifier_type[args.model]
    trainer_model = MilE2EModule(backbone, classifier_model, loss, get_metric(num_classes, task),
                                     get_transforms(args), args, num_classes=num_classes,
                                     classifier_type=classifier_type)
    return trainer_model


def save_test_results_to_csv(args, test_results):
    if not test_results:
        return
    res_dict = test_results[0]
    
    save_dict = {
        "dataset": args.dataset_name,
        "run_name": args.run_name,
        "tag": args.tag,
        "test_fold": args.val_fold,
        "model": args.model,
        "network": args.network,
        "transfer_type": args.transfer_type
    }
    save_dict.update(res_dict)
    
    df = pd.DataFrame([save_dict])
    
    csv_dir = os.path.join(args.output_dir, args.run_name)
    os.makedirs(csv_dir, exist_ok=True)
    csv_path = os.path.join(csv_dir, "all_experiments_results.csv")
    
    if os.path.exists(csv_path):
        df.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        df.to_csv(csv_path, mode='w', header=True, index=False)
    print(f"\n=========================================")
    print(f"result save to: {csv_path}")
    print(f"=========================================\n")


def main(args):
    pl.seed_everything(args.seed, workers=True)

    classes_names = get_class_names(args.dataset_name)
    data_module = PatchWsiDataModule(args.dataset_root, args.dataset_csv, classes_names=classes_names,
                                     val_fold=args.val_fold, 
                                     num_workers=args.num_workers, 
                                     num_workers_eval=args.num_workers_eval,
                                     drop_out=args.dropout_inst, weighted_sample=args.weighted_sample,
                                     pcps_selection_path=args.pcps_selection_path,
                                     pcps_selected_count=args.pcps_selected_count,
                                     pcps_random_count=args.pcps_random_count,
                                     pcps_sampling_mode=args.pcps_sampling_mode,
                                     pcps_total_count=args.pcps_total_count,
                                     pcps_evidence_concentration=args.pcps_evidence_concentration,
                                     pcps_context_temperature=args.pcps_context_temperature,
                                     pcps_distribution_concentration=args.pcps_distribution_concentration,
                                     pcps_eval_mode=args.pcps_eval_mode,
                                     pcps_eval_sampling_seed=(
                                         args.seed if args.pcps_eval_sampling_seed is None
                                         else args.pcps_eval_sampling_seed
                                     ),
                                     scatter_png_dir=args.scatter_png_dir)

    num_classes = len(classes_names[0])
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    checkpoint_callback = ModelCheckpoint(
        monitor='AUROC/val',
        dirpath=os.path.join(args.output_dir, args.run_name, args.tag, 'checkpoints'),
        filename='best-epoch={epoch:02d}-val_auroc={AUROC/val:.4f}',
        save_top_k=1,
        mode='max',
        save_last=True,
        auto_insert_metric_name=False,
    )

    backbone, num_fts = get_network(args)
    loss_weight = get_loss_weight(args, data_module)
    trainer_model = get_model(args, backbone, num_fts, num_classes, loss_weight)

    logger = TensorBoardLogger(save_dir=os.path.join(args.output_dir, args.run_name), name=args.tag)
    from pytorch_lightning.strategies import DDPStrategy
    if len(args.gpu_id) > 1:
        strategy = DDPStrategy(find_unused_parameters=True, static_graph=True)
    else:
        strategy = "auto"
    trainer = pl.Trainer(default_root_dir=os.path.join(args.output_dir, args.run_name),
                         max_epochs=args.epochs, log_every_n_steps=50, num_sanity_val_steps=0,
                         **precision_options(args),
                         accelerator="gpu", devices=args.gpu_id,
                         logger=logger,
                         callbacks=[lr_monitor, checkpoint_callback],
                         strategy=strategy,
                         )

    trainer.fit(trainer_model, data_module)
    verify_optimizer_updates(trainer, trainer_model, args)
    
    best_model_path = checkpoint_callback.best_model_path
    if trainer.is_global_zero:
        print(f"\n" + "="*50)
        print(f"Training complete! Best model saved to:\n{best_model_path}")
        print("="*50 + "\n")

    if len(args.gpu_id) > 1:
        torch.distributed.destroy_process_group()
        if trainer.is_global_zero:
            trainer_test = pl.Trainer(default_root_dir=os.path.join(args.output_dir, args.run_name), 
                                 num_sanity_val_steps=0, logger=logger,
                                 accelerator="gpu", devices=[args.gpu_id[0]], )
            test_results = trainer_test.test(trainer_model, data_module, ckpt_path=best_model_path)
            save_test_results_to_csv(args, test_results)
    else:
        test_results = trainer.test(trainer_model, data_module, ckpt_path='best')
        if trainer.is_global_zero:
            save_test_results_to_csv(args, test_results)


def add_argument_fun(parser):
    parser.add_argument("--clam-size", type=lambda s: [int(item) for item in s.split(',')], default=[192, 128, 128],
                        help="Choose the number of samples")
    return parser


def process_argument_fun(opts):
    return opts


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = get_arguments_additional(parser, add_argument_fun, process_argument_fun)

    save_parameters(args)

    main(args)
