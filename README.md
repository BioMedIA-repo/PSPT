# PSPT

Code for the paper **PSPT: Patch-Efficient Slide-Aware Prompt Tuning for End-to-End Whole-Slide Learning**.



## Installation

```bash
sudo apt-get install libopenslide0
pip install -r requirements.txt
```

## Workflow

BRACS uses the fixed official split. Run K random seeds on this same split
(K=5 below); `--fold` selects the online training seed, not a new partition.

### Segment tissue and export patches

```bash
python run.py patches --wsi-dir /data/BRACS --data-dir work/BRACS
```

### Extract frozen patch features

```bash
python run.py features --encoder-weights /models/UNI/pytorch_model.bin --data-dir work/BRACS
```

### Train the PCPS selector

```bash
python run.py pcps --data-dir work/BRACS --output-dir outputs/bracs_uni --fold 0
```

### Train and evaluate PSPT

```bash
python run.py train --encoder-weights /models/UNI/pytorch_model.bin --data-dir work/BRACS --output-dir outputs/bracs_uni --fold 0
```

### Run the remaining four seeds

```bash
for fold in 1 2 3 4; do
  python run.py pcps --data-dir work/BRACS --output-dir outputs/bracs_uni --fold $fold
  python run.py train --encoder-weights /models/UNI/pytorch_model.bin --data-dir work/BRACS --output-dir outputs/bracs_uni --fold $fold
done
```

## Other workflows

Follow BRACS; replace task and paths:

- COAD: `--task COAD`, `/data/COAD`, `work/COAD`, `outputs/coad_uni`.
- LUAD: `--task LUAD`, `/data/LUAD`, `work/LUAD`, `outputs/luad_uni`.

COAD and LUAD use five patient-level Monte Carlo partitions, selected by `--fold`
from `splits/coad` and `splits/luad`.
CONCH and PLIP follow the same workflow: set `--encoder CONCH` or `--encoder PLIP`.





Model weights: [https://drive.google.com/drive/folders/1mokCWTtSlyaMBWqT3MJrzMDTo3Zsno48?usp=sharing](https://drive.google.com/drive/folders/1mokCWTtSlyaMBWqT3MJrzMDTo3Zsno48?usp=sharing)
