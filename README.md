# Automated IC Chip Datasheet Lookup

IC detection using YOLOv11 Segment and identification using Tesseract trained on custom dataset.

## Project Overview

The project aims to automate the process of integrated ciruit (IC) chip datasheet lookup for professionals in electronics repair, businesses related to electronics and IC chips, and just people interested in electronics 😀

The project is still in development, and the datasheet lookup functionality is still unavailable.

Currently the project supports segmenting IC chips on different PCBs.

## How to run

> [!NOTE]
> It is recommended to use python virtual environment venv to avoid conflicts.
> 
> `python -m venv .venv`
> 
> `source .venv/bin/activate`

First, clone the repository with `git clone https://github.com/taangi1/IC_detection.git`

Then with pip install requirements with `pip install -r requirements.txt`

Run `predict.py` script with `python3 predict.py -p <picture_path>` to get back a segmented picture saved as `prediction-*timestamp*.jpg` in `predictions` directory, or run `python3 predict.py -c <camera_id>` to use a specific camera of the device (default is 0) and see the predictions in real time.

## Train on custom data

In order to train the model on custom data, `train.py` script can be used.

Use `--data` to specify the location of `data.yaml` file, where the training and validation sets are specified.

Use `--model` to specify the pretrained model to use for further training.

Use `--name` to specify the name of the training run.

For other functionality, use `python3 train.py --help`

The trained model will be saved to `runs/segment` by default.

## Solution Overview

In order to segment the IC chips from an image of PCB, YOLO-v11s Segment model was used. It was choosen as an accurate model, that can quickly deliver results, and run on basic hardware. The basic bounding boxes approach, or oriented bounding boxes (OBB) were not choosen, as some IC chips have non-rectangular shape, and sometimes the image of the chips is skewed. Because of that, a decision to use segmentation model was made.

For labeling the dataset, label-studio was used. It was choosen as easy to use, yet versatile software for image labeling.

The `160p500es-seg.pt` model is the stable model that works with most boards. The name contains information about number of pictures `160p`, number of epochs `500e`, and model used `s-seg`: small segment model. The `trained_models` directory contains old models or experimental models that are not stable and might show poor results.

## Difficulties

There were several different experiments to decide which model works the best for IC segmentation. The basic bounding box did not provide sufficient results, often capturing pins of the IC or parts of the motherboard when the image of the IC is rotated. The oriented bounding boxes fixed the previous problem, but a new problem became evident - the skewenes of chips from the photo. If the photo of the chips is taken at an angle, the shape of the chips is often trapezoidal, which makes bounding boxes not very accurate in selecting the IC chip.

## Future work

In the future, the project will utilize OCR library like tesseract, for IC chip identification, and search datasheet database for a match. It might include more types of different electronic components.

Any suggestions and contributions are welcome 😀
