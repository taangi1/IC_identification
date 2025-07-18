# Automated IC Chip Identification

IC detection using YOLOv11 Segment, trained on custom dataset, and identification using PaddleOCR.

## Project Overview

The project aims to automate the process of integrated ciruit (IC) chip datasheet lookup for professionals in electronics repair, businesses related to electronics and IC chips, and just people interested in electronics 😀

The project is still in development, and may have bugs. Please, report any bugs to [**Issues**](https://github.com/taangi1/IC_identification/issues).

<img width="786" height="933" alt="Image" src="https://github.com/user-attachments/assets/b9965878-7ff4-4f8e-bedc-957be4782a5f" />

## How to run

> [!NOTE]
> It is recommended to use python virtual environment venv to avoid conflicts.
> 
> `python -m venv .venv`
> 
> `source .venv/bin/activate`

First, clone the repository with `git clone https://github.com/taangi1/IC_identification.git`

`cd IC_identification/`

Then with pip install requirements with `pip install -r requirements.txt`

### Run full IC Identification in a web-app [NEW]

To run a web-app on local machine run the following commands:

Go to web-app directory `cd web-app`

Run the _app.py_ script with `python3 app.py`

Open your browser and in the search bar type `localhost:5000`

Allow the application to access the camera, then select the camera from drop-down menu. Click `Start Stream`. After the selected camera loads, put an IC chip in the frame. Click on the IC chip and see the extracted text below the stream.

### Run detection only

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

The PaddleOCR was choosen as the OCR model for it's ability to automatically detect rotation in the image. It also showed the best results compared to EasyOCR or Tesseract

Flask was used as lightweight web framework.

## Future work

The project will continously improve. In the future it is planned to utilize datasheet website archive api for lookup of the extracted IC chips.

Any suggestions and contributions are welcome 😀
