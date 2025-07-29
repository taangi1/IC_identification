# Models

This directory contains the machine learning models used by the software for inference and predictions.

## Current Models in Use

The following models are currently in active use:

* **`160p500es-seg.pt`**: This is the primary and most stable model currently in use. It has demonstrated reliable performance across a wide range of board configurations.
    * **Naming Convention Breakdown:**
        * `160p`: Trained with 160 pictures.
        * `500es`: Trained for 500 epochs.
        * `s-seg`: Denotes a small segmentation model.

## Archived Models

For reference, older or experimental models that are not actively used and may not be stable are stored in the `archived_models` directory. These models may produce poor or unreliable results if used.