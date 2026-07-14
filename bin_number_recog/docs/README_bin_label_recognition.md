# Bin Label Recognition

This `bin_number_recog` folder contains `recognize_bin_labels.py`, a conservative OpenCV + OCR
pipeline for reading printed bin numbers from labels. It is designed for a
robot arm workflow where the camera is aligned above the label before capture.

The script should not guess when the image is unreliable. It returns explicit
status codes so the robot/controller can decide whether to retake the photo,
realign, change exposure, or skip the plastic part.

The current target deployment uses newly printed paper labels, such as the
samples in `纸标签/`. These labels are expected to be much cleaner than the
older worn samples in `bin号识别/`. If the whole label is not visible, the
correct behavior is to reject the frame and retake it rather than trying to
read a partial label.

## Expected Input

- One image file, multiple image files, or a directory of images.
- The complete label should be visible and roughly front-facing.
- The target number is printed as white digits inside a black rectangle.
- The digit count is configurable. It is not assumed to be exactly 3 digits.
- The label may appear rotated by 0/90/180/270 degrees after robot alignment.

The current ROI ratios are tuned for the sample labels in which the black digit
panel is in the lower-left part of the correctly oriented printed label. The
script tests all four right-angle rotations and chooses the most reliable
candidate.

## Dependencies

Python packages:

```bash
pip install opencv-python numpy
```

System OCR dependency:

```bash
# macOS with Homebrew
brew install tesseract

# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y tesseract-ocr

# Conda / Miniconda, useful on this development machine
conda install -c conda-forge tesseract
```

`--engine auto` uses Tesseract. If Tesseract is not installed, the script
returns `ERR_OCR_MISSING` instead of silently falling back to an unsafe guess.

There is also a `--engine template` mode, but it is for development/debugging
only. Do not use it as the production recognizer unless it has been validated
on the real camera data.

## Tesseract Quick Start

Tesseract is the digit-recognition part of the current production path. The
OpenCV code first finds and crops the black digit panel; Tesseract then reads
only digits from that cropped ROI.

Check whether Tesseract is already installed:

```bash
which tesseract
tesseract --version
```

On this development machine, `conda` is available. If `which tesseract` prints
nothing, install Tesseract with:

```bash
conda install -c conda-forge tesseract
```

After installation, verify:

```bash
tesseract --version
```

Then run the script in Tesseract/production mode:

```bash
python3 recognize_bin_labels.py "bin号识别/000045.png" --engine auto
```

Or run the whole sample folder:

```bash
python3 recognize_bin_labels.py "bin号识别" \
  --engine auto \
  --csv bin_label_results.csv \
  --debug-dir debug_bin_labels \
  --min-digits 1 \
  --max-digits 8
```

Expected behavior:

- If Tesseract is installed and OCR succeeds, `ok=True` and `code=OK`.
- If Tesseract is not installed, `code=ERR_OCR_MISSING`.
- If the complete label is clipped by the image boundary, `code=ERR_LABEL_INCOMPLETE`.
- If the black digit panel has strong glare/overexposure, `code=ERR_PANEL_GLARE`.
- If the black digit panel has insufficient contrast, `code=ERR_PANEL_LOW_CONTRAST`.
- If the image is bad or OCR is unsure, the script returns a specific error
  code instead of guessing.

## Local Tesseract Experiment Log

This experiment was run on the development machine before moving the files to
the server.

Environment:

```text
OS/package platform: macOS arm64 with Miniconda
conda: /opt/miniconda3/bin/conda
tesseract: /opt/miniconda3/bin/tesseract
tesseract version: 5.5.2
```

Installation command used:

```bash
conda install -c conda-forge tesseract -y
```

Verification command:

```bash
which tesseract
tesseract --version
```

Single-image test command:

```bash
python3 recognize_bin_labels.py "bin号识别/000045.png" \
  --engine auto \
  --debug-dir debug_bin_labels
```

Observed result after adding white-background OCR preprocessing:

```text
000045.png  ok=False  code=ERR_LOW_CONFIDENCE  value=02  conf=0.35  blur=166.5  label_quality=1.0  rotation=0  source=warp  recognizer=tesseract  box=12,195,147,70
```

Interpretation:

- The OpenCV localization part worked: the label was found, warped, and the
  black digit panel crop was correct.
- The debug crop `debug_bin_labels/000045_panel.png` is visually `052`.
- Tesseract read the crop as `02`, missing the middle digit `5`.
- OpenCV component detection saw 3 digit-like components, while Tesseract
  returned only 2 digits.
- The script therefore downgraded confidence to `0.35` and returned
  `ERR_LOW_CONFIDENCE` instead of accepting the wrong value.

Conclusion:

- Tesseract is installed and callable, but it is not reliable enough yet on the
  current black-background sample.
- The error-code guardrails are working correctly: a wrong partial read does
  not become `OK`.
- Tesseract can still be kept as a lightweight baseline, but the next server
  experiment should try PaddleOCR on the same cropped panel images.

Recommended next server experiment:

1. Copy this project folder to the server.
2. Install normal Python dependencies: `opencv-python` and `numpy`.
3. Run this script with `--debug-dir` to generate `*_panel.png` crops.
4. Test PaddleOCR directly on those `*_panel.png` crops first.
5. If PaddleOCR reads the crops correctly, integrate it as another
   `--engine paddleocr` backend in `recognize_bin_labels.py`.
6. Keep the same validation checks: digit count, confidence, black-panel
   existence, blur, label geometry, and rotation candidate selection.

## OCR Engine Choice

The script currently integrates Tesseract as the OCR backend. Tesseract is the
lightweight option:

- It is mostly a system command-line tool.
- It has fewer Python/runtime dependencies.
- It is easy to install on a server.
- It is a good first choice when the ROI is already tightly cropped and the
  printed digits are clear.

PaddleOCR is another good option for this project, especially if the real robot
images have mild blur, glare, or exposure variation. It is "heavier" than
Tesseract, which means:

- larger install size
- more Python packages/runtime dependencies
- model files must be available
- CPU/GPU package choice may matter
- startup and memory usage may be higher

In exchange, PaddleOCR is often more robust than Tesseract on imperfect images.

Recommended path:

1. Use OpenCV to locate and crop the black digit panel.
2. Try Tesseract first if deployment simplicity matters most.
3. Try PaddleOCR if Tesseract fails too often on real robot-camera images.
4. If both are not stable enough, collect cropped ROI failures and train a small
   digit-only recognition model.

## CNN Digit Recognizer

The CNN path keeps the same conservative OpenCV front end:

1. Detect the complete label.
2. Perspective-warp the label.
3. Try 0/90/180/270 degree orientations.
4. Crop and validate the lower-left black digit panel.
5. Split the panel into single digit crops.
6. Classify each digit crop with a small CNN.

This means the CNN does not see the table, hand, cable, or other background
objects. It only sees cropped white-on-black digit images.

Install runtime dependencies in the vision environment:

```bash
python -m pip install opencv-python numpy torch
```

### 1. Extract digit crops from known full-label images

If a folder contains full label images for bin number `89`:

```bash
python3 extract_digit_crops.py \
  /workspace/huangjie/pure_vision_detection/datasets/bin_number/89 \
  --label 89 \
  --output-dir /workspace/huangjie/pure_vision_detection/datasets/bin_digit_crops \
  --prefix 89_
```

This script reuses the same conservative OpenCV front end as
`recognize_bin_labels.py`: full-label detection, blur check, incomplete-label
check, quadrilateral quality check, black-panel localization, and black-panel
quality checks. If a frame cannot pass those checks, it is rejected instead of
being used as CNN training data.

The output structure is:

```text
bin_digit_crops/
  8/
    89_000000_0_8.png
  9/
    89_000000_1_9.png
```

Repeat this for each known bin number. The CNN training script expects this
`0/..9/` folder layout.

### 2. Train the CNN

```bash
python3 train_digit_cnn.py \
  --data-root /workspace/huangjie/pure_vision_detection/datasets/bin_digit_crops \
  --output /workspace/huangjie/pure_vision_detection/bin_number_recog/models/digit_cnn.pt \
  --epochs 40 \
  --batch-size 64
```

The model is intentionally small and CPU-friendly. Use `--device cuda` if a GPU
environment is available.

### 3. Test the CNN directly on black-panel crops

```bash
python3 infer_digit_cnn.py \
  debug_bin_labels/000045_panel.png \
  --checkpoint models/digit_cnn.pt \
  --mode panel
```

Use `--mode digit` if the input images are already single digit crops.

### 4. Use CNN inside the full conservative recognizer

```bash
python3 recognize_bin_labels.py \
  /workspace/huangjie/pure_vision_detection/datasets/bin_number/89 \
  --engine cnn \
  --cnn-checkpoint /workspace/huangjie/pure_vision_detection/bin_number_recog/models/digit_cnn.pt \
  --csv cnn_results_89.csv \
  --debug-dir debug_cnn_89 \
  --min-digits 1 \
  --max-digits 8 \
  --min-confidence 0.80
```

If `--engine cnn` is selected without a valid checkpoint, the script returns
`ERR_CNN_MISSING`. If the panel is glared, blurry, incomplete, or low-contrast,
the existing quality error codes are returned before the CNN is allowed to
guess.

## Basic Usage

```bash
python3 recognize_bin_labels.py "bin号识别" \
  --engine auto \
  --csv bin_label_results.csv \
  --debug-dir debug_bin_labels \
  --min-digits 1 \
  --max-digits 8
```

Single image:

```bash
python3 recognize_bin_labels.py "bin号识别/000045.png" --engine auto
```

Development-only template mode:

```bash
python3 recognize_bin_labels.py "bin号识别/000045.png" --engine template
```

## Output

The script prints one tab-separated row per image and can also write a CSV.

CSV fields:

- `file`: input file name
- `ok`: `True` only when recognition passed all checks
- `code`: machine-readable status code
- `message`: human-readable status message
- `value`: recognized digit string, empty on most failures
- `confidence`: recognizer confidence
- `source`: usually `warp`
- `recognizer`: `tesseract`, `tesseract-missing`, `template`, or `none`
- `box`: ROI box on the rectified label image, formatted as `x,y,w,h`
- `blur`: Laplacian blur score
- `label_quality`: quadrilateral geometry score
- `rotation`: selected rotation in degrees: `0`, `90`, `180`, or `270`

Example:

```text
000045.png  ok=True   code=OK  value=052  conf=0.706  blur=166.5  label_quality=1.0  rotation=0  source=warp  recognizer=template  box=12,195,147,70
```

## Error Codes

Use `code` as the control signal for the robot/controller.

| Code | Meaning | Suggested Action |
| --- | --- | --- |
| `OK` | Recognition passed all checks. | Use `value`. |
| `ERR_IMAGE_READ` | Image could not be loaded. | Check file path/camera save step. |
| `ERR_BLUR` | Image is too blurry. | Retake photo, adjust focus/exposure, reduce motion. |
| `ERR_LABEL_INCOMPLETE` | Label is clipped by the image boundary. | Realign camera/robot and retake; do not OCR partial labels. |
| `ERR_LABEL_NOT_FOUND` | Label quadrilateral was not found. | Realign camera/robot and retake. |
| `ERR_LABEL_GEOMETRY` | Label shape is too distorted for reliable warp. | Realign camera angle and retake. |
| `ERR_BLACK_PANEL_NOT_FOUND` | Proportional ROI did not contain a valid black digit panel. | Realign, retake, or inspect ROI ratios. |
| `ERR_PANEL_GLARE` | Black digit panel has glare or overexposure. | Retake after changing angle/lighting; do not OCR through glare. |
| `ERR_PANEL_LOW_CONTRAST` | Black digit panel contrast is too low. | Retake, adjust exposure/focus/lighting. |
| `ERR_DIGITS_NOT_FOUND` | Black panel was found, but digits were not detected. | Retake, adjust exposure, inspect OCR preprocessing. |
| `ERR_DIGIT_LENGTH` | Digit count is outside `--min-digits/--max-digits`. | Check expected digit range or retake. |
| `ERR_LOW_CONFIDENCE` | OCR confidence is below threshold. | Retake, adjust lighting/exposure, or use a better OCR engine. |
| `ERR_OCR_MISSING` | Tesseract is not installed but OCR was requested. | Install Tesseract or use a validated OCR backend. |
| `ERR_CNN_MISSING` | CNN was requested but the checkpoint is missing or failed to load. | Provide `--cnn-checkpoint` or use another engine. |

## Debug Images

When `--debug-dir` is provided, the script writes:

- `*_debug.png`: original image with detected label quadrilateral
- `*_warp.png`: selected rectified label orientation
  - red rectangle: coarse proportional ROI
  - blue rectangle: refined black digit panel inside the ROI
- `*_rot0_warp.png`, `*_rot90_warp.png`, `*_rot180_warp.png`, `*_rot270_warp.png`:
  all four orientation candidates with status text
- `*_panel.png`: final black digit panel sent to OCR
- `*_mask.png`: binarized digit mask

The fastest way to diagnose failures is to inspect `*_warp.png`:

- If the green/rectified label is wrong, tune label detection or robot align.
- If the red ROI misses the black panel, tune `DEFAULT_ROI_RATIOS`.
- If the blue box is wrong, tune `refine_black_digit_panel`.
- If boxes are correct but OCR fails, tune OCR/preprocessing.

## Pipeline Summary

1. Read image with OpenCV.
2. Compute blur score using Laplacian variance.
3. Detect bright low-saturation label area.
4. Fit a minimum-area quadrilateral.
5. Reject distorted quadrilaterals using `label_geometry_quality`.
6. Perspective-warp label to `DEFAULT_WARP_SIZE`.
7. Try four right-angle label rotations: 0, 90, 180, and 270 degrees.
8. For each rotation, crop expected bin-number ROI using `DEFAULT_ROI_RATIOS`.
9. Confirm there is a black rectangle inside the ROI.
10. Reject unsafe black panels before OCR if they are glared, overexposed, or too low-contrast.
11. Crop the black rectangle and preprocess it for OCR.
12. Recognize digits with Tesseract.
13. Validate digit count and confidence.
14. Select the best candidate. `OK` candidates win; otherwise the script
    returns the most useful failure code from the candidate closest to success.

## Important Tunables

In `recognize_bin_labels.py`:

- `DEFAULT_WARP_SIZE = (420, 420)`
  - Output size for rectified label images.
- `DEFAULT_ROI_RATIOS = (0.03, 0.61, 0.38, 0.83)`
  - Normalized crop region for the lower-left bin-number area.
  - Format is `(x0, y0, x1, y1)` relative to the warped label.
- `refine_black_digit_panel`
  - Confirms and crops the actual black panel within the coarse ROI.
- `prepare_digit_mask`
  - OCR preprocessing: resize, blur, CLAHE, adaptive threshold.

Command-line thresholds:

```bash
--min-digits 1
--max-digits 8
--min-confidence 0.55
--min-blur 35.0
--min-label-quality 0.55
```

Recommended calibration procedure:

1. Collect 50-200 real robot-camera images.
2. Run with `--debug-dir`.
3. Inspect all non-`OK` cases.
4. Tune `DEFAULT_ROI_RATIOS` only if the red ROI misses the black panel.
5. Tune `--min-blur` from real blur distributions.
6. Tune `--min-label-quality` from real bad-angle examples.
7. Tune OCR preprocessing or replace Tesseract if boxes are correct but OCR is weak.

## Robot Integration Notes

Suggested action mapping:

- `OK`: continue with recognized `value`.
- `ERR_BLUR`: retake without moving, or adjust exposure/focus.
- `ERR_LABEL_INCOMPLETE`: retake after moving the camera/robot so the full label is visible.
- `ERR_LABEL_NOT_FOUND`: rerun align policy and retake.
- `ERR_LABEL_GEOMETRY`: rerun align with angle correction and retake.
- `ERR_BLACK_PANEL_NOT_FOUND`: retake after align; if repeated, skip part or inspect label type.
- `ERR_PANEL_GLARE`: retake after changing camera/lighting angle; this is a normal recoverable vision failure.
- `ERR_PANEL_LOW_CONTRAST`: retake after improving focus/exposure/lighting.
- `ERR_DIGITS_NOT_FOUND`: retake with better exposure/lighting.
- `ERR_LOW_CONFIDENCE`: retake; after repeated failures, skip part.
- `ERR_OCR_MISSING`: server setup issue, not a robot recovery issue.
- `ERR_CNN_MISSING`: model setup issue, not a robot recovery issue.

The script is intentionally conservative. A failure code is better than an
incorrect bin number because the robot can recover by retaking, realigning, or
skipping.

## Known Limitations

- Tesseract confidence is currently simplified because the script calls the
  command-line OCR path. For stronger confidence, switch to TSV output or a
  Python OCR library and use per-character confidence.
- `--engine template` is not production-safe for variable digit length. It can
  over-read noise. Keep it for local debugging only.
- Partial labels are intentionally rejected. This is expected for edge-of-frame
  labels in multi-label overview images.
- If future label layouts change, update `DEFAULT_ROI_RATIOS` or add per-label
  layout detection.
- If robot alignment becomes highly variable, consider adding a small detector
  for the black digit panel rather than relying mainly on proportional ROI.
- If the robot can align to the square label edges but not to text orientation,
  keep the four-rotation candidate search enabled. The `rotation` output tells
  downstream code which orientation was selected.

## Files To Copy To Server

Minimum:

```text
recognize_bin_labels.py
README_bin_label_recognition.md
```

Optional for calibration/debug:

```text
debug_bin_labels/
bin_label_results.csv
```
