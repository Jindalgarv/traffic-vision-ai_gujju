# TrafficVision AI - Two Roboflow Model Version

This build supports **two Roboflow object-detection models** in the same Streamlit app:

1. **Helmet / no-helmet model** for violation detection.
2. **License plate model** for plate-box detection before OCR.

The app runs both models on the uploaded image, keeps the helmet model output for helmet-rule evaluation, uses the plate model output for OCR, then merges both outputs for annotation and evidence generation.

## Why two models?

Your helmet model can stay focused on motorcycle / rider / helmet / no_helmet detection. Your plate model can be trained separately with one plate class such as:

```text
license_plate
number_plate
plate
```

Use `license_plate` if possible. The OCR code recognizes common variants, but clear class names make debugging easier.

## Model IDs

From a Roboflow model URL like:

```text
https://app.roboflow.com/<workspace>/<project>/models/<project>/2
```

use this as the model ID:

```text
<project>/2
```

Example:

```text
helmet-detection-traffic-cgg9e/2
```

## Setup

```bash
cd traffic-vision-ai-two-models
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

For real OCR instead of mock fallback, also install one OCR engine:

```bash
pip install easyocr
```

or:

```bash
pip install paddleocr paddlepaddle
```

## Local environment configuration

Copy the example file and create your real `.env` file:

```bash
cp .env.example .env
```

Fill it with your Roboflow settings:

```bash
ROBOFLOW_API_KEY=your_roboflow_api_key_here
ROBOFLOW_HELMET_MODEL_ID=your-helmet-model/1
ROBOFLOW_PLATE_MODEL_ID=your-license-plate-model/1
ROBOFLOW_API_URL=https://serverless.roboflow.com
```

The Streamlit sidebar only shows safe Loaded/Missing statuses. It does not show the full API key or ask you to enter model IDs manually.

## Run

```bash
streamlit run app.py
```

## Important integration logic

In `app.py`, the app now does this:

```python
helmet_detections = helmet_detector.detect(image)
plate_detections = plate_detector.detect(image)

# Rules use only helmet_detections.
violations = evaluate_violations(image.size, helmet_detections)

# OCR uses only plate_detections.
plate_results = extract_license_plate_ocr(image, plate_detections, ocr_engine)

# Annotation uses both.
detections = helmet_detections + plate_detections
```

This avoids a common bug where plate boxes accidentally affect helmet-rule logic.

## Evidence generation and human review

When you click **Process and Log Violations**, the app creates one evidence record per violation:

- Unique violation ID, for example `TVAI-20260618-012345-A1B2C3D4`
- Timestamp
- Uploaded image filename
- Violation type
- Detection confidence
- OCR plate number, or `N/A` if no plate was detected/read
- Review status, initially `Pending Review`
- Full annotated evidence image saved under `data/annotated/`
- Cropped evidence image saved under `data/crops/`
- SQLite metadata saved in `data/traffic_violations.db`

Open the **Review Dashboard** tab to mark evidence as **Approve**, **Reject**, or **Needs Manual Check**.

## If plate OCR still shows N/A

Check these first:

1. `.env` has a valid `ROBOFLOW_PLATE_MODEL_ID`.
2. The plate model is trained as **object detection**, not classification.
3. The plate model class name contains `plate`, for example `license_plate`.
4. The plate model actually detects a visible plate in the uploaded image.
5. A real OCR engine is installed. Without EasyOCR/PaddleOCR, the app uses mock fallback for demo behavior.
