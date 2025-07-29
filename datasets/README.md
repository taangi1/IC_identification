# Datasets

This directory contains the various datasets used for training and evaluating the machine learning models within this project.

## Dataset Structure

Each dataset is typically organized into an `images` folder and a `labels` folder. Within these, data is further split into:
* `train/`: Contains images and corresponding labels for model training.
* `val/`: Contains images and corresponding labels for model validation during training.

Additionally, common files found within each dataset directory include:
* `classes.txt`: A plain text file listing the names of all classes featured in the dataset, one per line.
* `notes.json` (or similar): A JSON file containing mappings of class IDs to their corresponding labels, along with any other relevant metadata.

## Datasets Available

Currently, there are three primary datasets available, each tailored for different annotation types:

1.  **`ic_dataset`**: This dataset uses basic axis-aligned bounding boxes for object detection.
2.  **`ic_dataset_obb`**: This dataset uses oriented bounding boxes (OBB) for object detection, providing more precise localization for rotated objects.
3.  **`ic_dataset_polygon`**: This dataset uses polygon annotations for instance segmentation, outlining objects with precise boundaries.

### Dataset Used in Final Model

The `ic_dataset_polygon` dataset was exclusively used for training the final segmentation model (`160p500es-seg.pt`). While the complete dataset contains 160 labeled images, only a representative subset is publicly displayed on GitHub due to file size constraints. The labels within this dataset adhere to the standard YOLO format for segmentation models, and the dataset was meticulously annotated using [Label Studio](https://labelstud.io/).