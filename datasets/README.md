# Dataset structure

## Dataset used

The dataset that was used in the final model is a custom dataset `ic_dataset_polygon`, that currently contains 160 labeled images, but only a small part is shown in github. The labels are in standard yolo format for segmentation model. The dataset was annotated using label-studio.


## Description

There are 3 datasets, for basic bouding boxes `ic_dataset`, oriented bounding boxes `ic_dataset_obb`, and polygons `ic_dataset_polygon`. Each of them has images and labels folders, where the data is split into training `train` and validation sets `val`. `classes.txt` contains the classes featured in the dataset, while `notes.json` has the id of the classes along with corresponding labels.