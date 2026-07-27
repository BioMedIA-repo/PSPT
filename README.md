# PSPT

Code for **PSPT: Patch-Efficient Slide-Aware Prompt Tuning for
End-to-End Whole-Slide Learning**.

## Installation

```bash
conda create -n pspt python=3.10 -y
conda activate pspt
pip install -r requirements.txt
pip install -e .
```

CONCH additionally requires the official CONCH v1 package and checkpoint.
OpenSlide also requires the system OpenSlide library.

Model weights: https://drive.google.com/drive/folders/1mokCWTtSlyaMBWqT3MJrzMDTo3Zsno48?usp=sharing

More code details will be updated after the paper is accepted.

## Patch extraction

Extract patches from the original slides using the provided coordinate H5
files:

```bash
python preprocessing/extract_scatter_png.py \
  --svs_dir /path/to/slides \
  --h5_dir /path/to/coordinate_h5 \
  --output_dir /path/to/patch_root \
  --patch_size 256 \
  --num_workers 8
```

The H5 coordinate order must match the PCPS score files.

The generated directory is:

```text
patch_root/
└── WSI_ID/
    ├── WSI_ID_patch_0.png
    ├── WSI_ID_patch_1.png
    └── ...
```

## Evaluation

UNI:

```bash
python -m pspt.evaluate \
  --checkpoint /path/to/UNI_M256_model000.pt \
  --split-csv /path/to/bright3_label.csv \
  --scatter-png-dir /path/to/patch_root \
  --pcps-scores /path/to/UNI_pcps_scores_fold0.pt \
  --output-dir outputs/uni_model000
```

CONCH:

```bash
python -m pspt.evaluate \
  --checkpoint /path/to/CONCH_M256_model000.pt \
  --split-csv /path/to/bright3_label.csv \
  --scatter-png-dir /path/to/patch_root \
  --pcps-scores /path/to/CONCH_pcps_scores_fold0.pt \
  --conch-backbone-checkpoint /path/to/conch_v1_checkpoint.bin \
  --output-dir outputs/conch_model000
```

To evaluate all released checkpoints:

```bash
PSPT_MODEL_PACKAGE=/path/to/PSPT_PUBLIC_WEIGHTS_BRACS_20260727 \
BRACS_PATCH_DIR=/path/to/patch_root \
OUTPUT_DIR=/path/to/outputs \
bash scripts/reproduce_bracs_all.sh
```
