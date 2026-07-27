"""Metrics used by the PSPT inference release."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    roc_auc_score,
)


def summarize(records, class_bias=None):
    labels = np.asarray(
        [record["label"] for record in records], dtype=int
    )
    probabilities = np.asarray(
        [record["probability"] for record in records], dtype=float
    )
    raw_predictions = probabilities.argmax(axis=1)
    output = {
        "num_slides": int(len(records)),
        "macro_auroc": float(
            roc_auc_score(
                labels,
                probabilities,
                multi_class="ovr",
                average="macro",
            )
        ),
        "raw_accuracy": float(
            accuracy_score(labels, raw_predictions)
        ),
        "raw_balanced_accuracy": float(
            balanced_accuracy_score(labels, raw_predictions)
        ),
        "raw_macro_f1": float(
            f1_score(labels, raw_predictions, average="macro")
        ),
        "cross_entropy": float(
            log_loss(labels, probabilities, labels=[0, 1, 2])
        ),
    }
    if class_bias is not None:
        calibrated = (
            np.log(np.clip(probabilities, 1e-12, 1.0))
            + np.asarray(class_bias, dtype=float)
        ).argmax(axis=1)
        output.update(
            {
                "val_calibrated_accuracy": float(
                    accuracy_score(labels, calibrated)
                ),
                "val_calibrated_balanced_accuracy": float(
                    balanced_accuracy_score(labels, calibrated)
                ),
                "val_calibrated_macro_f1": float(
                    f1_score(
                        labels, calibrated, average="macro"
                    )
                ),
            }
        )
    return output

