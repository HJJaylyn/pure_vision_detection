# Waffle Box Chip Count

`detect_chip_count.py` is a traditional-CV pipeline for counting chips in a
six-slot black waffle box. The black material is only a physical helper; a
table shadow alone cannot select a tray.

## Recognition flow

1. Extract Canny edges and a strict raw-black helper mask.
2. Build tray candidates from two pairs of approximately parallel long Canny
   lines. The outer silhouette may be five-sided because the tray has height:
   the short side-wall/angled edge is allowed, while the two long-direction
   pairs define a perspective quadrilateral for rectification.
3. Score candidates by Canny line support, edge continuity, physical extent,
   and coverage of a continuous tray-sized black component. This prevents a
   strong internal chip divider from being selected as the tray boundary.
4. Expand the selected quadrilateral slightly and perspective-rectify it.
5. Preserve the original-image Canny map by perspective-warping it into the
   rectified tray alongside a fresh local Canny pass. This avoids losing a
   real chip border merely because RGB interpolation softened a small source
   image during rectification.
6. Generate chip candidates primarily from rectified Canny contours. Normal
   closed convex four-sided contours are used directly. If one *local* thick
   Canny outline folds into a non-convex 7/8-point path with tiny contour area,
   its own minimum-area quadrilateral is used as a fallback. This handles a
   visually clear chip outline without combining lines from separate chips.
7. Use a relaxed bright-surface mask only as corroborating evidence. Brightness
   is not a hard requirement because silver chips can reflect dark surroundings
   or be much darker than the white background while remaining brighter than
   the surrounding black slot.
8. Reject a final chip quadrilateral larger than `25%` of the rectified image.
   Remove overlapping candidates using true quadrilateral overlap; one physical
   chip cannot occupy multiple candidates.

The detector does **not** assume a fixed `2 x 3` or `3 x 2` image grid and does
not use a grid cell as a chip boundary.

Single image with a debug image:

```bash
source /workspace/huangjie/miniconda3/bin/activate vision_recog
cd /workspace/huangjie/pure_vision_detection
python chip_count_recog/detect_chip_count.py \
  datasets/chip_count/6/000000.jpg \
  --debug-dir datasets/chip_count_debug_single
```

Evaluate the collected dataset and save debug images only for errors:

```bash
python chip_count_recog/detect_chip_count.py \
  --dataset datasets/chip_count \
  --debug-dir datasets/chip_count_debug \
  --csv datasets/chip_count_eval.csv
```

At the start of a dataset evaluation, the specified `--debug-dir` and `--csv`
are cleared so the output belongs only to that run. The dataset subfolder name
is treated as the expected count. Adjust `--occupancy-threshold` only after
inspecting the generated problem debug images.

The debug image distinguishes the strict raw-black helper mask from the green
selected tray quadrilateral and yellow safe crop margin. It also includes the
rectified Canny chip candidates and bright-surface candidates. Green chip
quadrilaterals are accepted; orange quadrilaterals are rejected by the
occupancy score.

## White Background Dataset

The current validated white-background dataset command is:

```bash
source /workspace/huangjie/miniconda3/bin/activate vision_recog
cd /workspace/huangjie/pure_vision_detection
python chip_count_recog/detect_chip_count.py \
  --dataset datasets/chip_count_white_bg \
  --debug-dir datasets/chip_count_debug_white_bg \
  --csv datasets/chip_count_white_bg_eval.csv
```

This version deliberately favors clear local Canny chip boundaries over an
absolute silver/white threshold. It is robust to normal dim reflections, but
an extremely underexposed image can still lack enough contrast for reliable
counting. Do not broadly relax the detector solely to fit one such outlier.

## Franka Reusable Copy

The Franka workspace keeps an independent copy at
`/workspace/huangjie/Franka/pure_vision_detection/chip_count/detect_chip_count.py`.
Franka task scripts import that copy so they do not depend on this workstation
directory. When this detector changes, copy the updated file there before
synchronizing the Franka workspace:

```bash
cp /workspace/huangjie/pure_vision_detection/chip_count_recog/detect_chip_count.py \
  /workspace/huangjie/Franka/pure_vision_detection/chip_count/detect_chip_count.py
```
