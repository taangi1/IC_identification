# Automated IC Chip Identification

integrated circuit (IC) detection using YOLOv11 Segment, trained on custom dataset, and identification using PaddleOCR.

## Project Overview

The project aims to automate the process of manual entry of IC chip markings to find datasheets, made for professionals in electronics repair, businesses related to electronics and IC chips, and enthusiasts interested in electronics 😀

The project is still in development, and may have bugs. Please, report any bugs to [**Issues**](https://github.com/taangi1/IC_identification/issues).

<img width="786" height="933" alt="Image" src="https://github.com/user-attachments/assets/b9965878-7ff4-4f8e-bedc-957be4782a5f" />

## How to Run

> [!NOTE]
> It is recommended to use python virtual environment venv to avoid conflicts.
> 
> `python -m venv .venv`
> 
> `source .venv/bin/activate` # On Windows, use `.venv\Scripts\activate`

First, clone the repository with `git clone https://github.com/taangi1/IC_identification.git`

`cd IC_identification/`

Then, install the required packages with `pip install -r requirements.txt`

### Run IC Identification Software in a local Web App

To launch the web application on your local machine, run the following command:

`python3 web-app/app.py`

The application should automatically open in your default internet browser within 5 seconds. If not, open your web browser and navigate to `http://localhost:5000`.

**Usage:** Allow the application to access your camera. Then, select your camera from the drop-down menu and click Start Stream. Once the selected camera feed loads, position an IC chip within the frame. Click on the IC chip in the stream to see the extracted text displayed below.

### Run IC Detection only

Run `python3 scripts/predict.py -p <picture_path>` to get back a segmented picture saved as `prediction-*timestamp*.jpg` in `predictions` directory, or run `python3 scripts/predict.py -c <camera_id>` to use a specific camera of the device (default is 0) and see the predictions in real time. To exit press `q`.

### Train on Custom Data

In order to train the model on custom data, `scripts/train.py` script can be used.

Use `--data` to specify the location of `data.yaml` file, where the training and validation sets are specified.

Use `--model` to specify the pretrained model to use for further training.

Use `--name` to specify the name of the training run.

For other functionality, use `python3 scripts/train.py --help`

The trained model will be saved to `runs/segment` by default.

## Solution Overview

To segment IC chips from an image of a Printed Circuit Board (PCB), the YOLO-v11s Segment model was employed. This model was chosen for its accuracy and speed, making it suitable for deployment on basic hardware. Traditional bounding box approaches (including oriented bounding boxes) were not selected due to the varied, sometimes non-rectangular, and occasionally skewed shapes of IC chips. Therefore, a segmentation model was preferred to accurately delineate the chip boundaries.

For dataset labeling, Label-Studio was utilized, valued for its ease of use and versatility in image annotation.

The `models/160p500es-seg.pt` model is considered the stable version, performing reliably across most boards. Its naming convention indicates: 160p (160 pictures), 500e (500 epochs), and s-seg (small segment model). The `archived_models` directory contains older or experimental models that may not be stable and could yield poor results.

PaddleOCR was selected as the Optical Character Recognition (OCR) model due to its robust ability to automatically detect image rotation, and it demonstrated superior performance compared to alternatives like EasyOCR and Tesseract.

Flask was chosen as the lightweight web framework for the web application's functionality.

## Future work

The project will continuously improve.

Any suggestions and contributions are highly welcome 😀
