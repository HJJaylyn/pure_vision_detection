# PaddleOCR 前处理链路说明

本文档说明当前 `bin_number_recog` 里，一张原始标签图片在送进 PaddleOCR `TextRecognition` 之前到底经过了哪些处理。

核心入口：

- 批量评估：`evaluate_paddle_text_recognition.py`
- Debug 评估：`debug_paddle_eval.py`
- 标签与黑块检测：`recognize_bin_labels.py`、`extract_digit_crops.py`

## 总流程

输入是一张完整相机图，例如：

```text
datasets/bin_number/304/000002.png
```

最终送进 PaddleOCR 的不是原图，而是裁剪、矫正、去白边、归一化后的黑色数字块：

```text
黑底 + 白色数字，固定尺寸 200x90
```

整体流程如下：

```text
原图
  -> 读图
  -> 模糊检查
  -> 找整张方形标签
  -> 用外框透视变换成 420x420 正方形标签图
  -> 在外框 warp 结果里再次寻找白色标签内容面
  -> 裁掉粉色/黑色外边框，并把白色内容面再次 warp 成 420x420
  -> 尝试 0/90/180/270 四个方向
  -> 在每个方向里裁一个较宽的数字区域 ROI
  -> 在宽 ROI 里找真实黑色数字方块
  -> 黑块裁剪、二次矫正、去白边、归一化到 200x90
  -> 按若干规则选择最可信方向
  -> 再做一次 OCR 输入级去白边和归一化
  -> PaddleOCR TextRecognition
  -> 只保留数字，并做常见 OCR 混淆修正
```

## 1. 读图

代码位置：

```python
cv2.imread(...)
```

如果读取失败：

```text
ERR_IMAGE_READ
```

## 2. 模糊检查

代码位置：

```python
recognize_bin_labels.blur_score()
recognize_bin_labels.preprocess_label_image()
```

方法：

- 转灰度
- 计算 Laplacian 方差
- 默认阈值：`min_blur=35.0`

如果模糊度太低：

```text
ERR_BLUR
```

这个错误表示应该重新拍照，不继续尝试识别。

## 3. 找整张标签

代码位置：

```python
recognize_bin_labels.find_label_quad()
recognize_bin_labels.warp_label()
```

当前假设：

- 标签整体接近正方形。
- 标签主体是白色。
- 白色内容区域外面有一圈粉色或黑色边框。
- 外边框用于帮助定位整张标签，但不应该进入后续数字区域识别。
- 标签内部可能有黄色条、二维码、条码、黑色数字块等结构。

处理方式：

1. 转 HSV、原始灰度。
2. 先用 Canny/边缘闭合找像标签外框的四边形候选：
   - 粉色边框、黑色边框、标签纸边缘都可以提供边线。
   - 边缘候选不要求内部填充很高，因为它可能只是外框线。
   - 候选仍然要满足面积、长宽比、几何角度等基本约束。
3. 对原图灰度做 CLAHE 局部亮度均衡，得到辅助检测灰度。
4. 用 `max(raw_gray, clahe_gray)` 辅助找白色标签主体，作为边缘检测失败时的 fallback：

```python
gray_for_detection = max(gray_raw, gray_clahe)
bright = (gray_for_detection > 125) & (saturation < 175)
```

注意：亮度均衡只用于定位判断，最终透视变换仍然从原始 RGB 图采样。

5. 形态学 close/open，把标签主体连起来。
6. 找外轮廓。
7. 用 `cv2.minAreaRect()` 拟合候选标签。
8. 按面积、填充率、长宽比筛选。
9. 将每个候选临时 warp 成小正方形，再做一次标签特征评分：
   - 中间区域必须像白色标签内容面。
   - 中间区域不能大面积呈现绿色桌面颜色。
   - 外圈如果有粉色边框、黑色边框或明显边缘线，会加分。
   - 边框证据不是硬要求，因为真实边框可能磨损或缺失。
10. 对贴图像边缘的候选降低分数，避免选到旁边被截断的旧标签。

这一步的目的，是让“类似标签外框的四边形边线”成为第一优先级，白色区域只作为 fallback；同时避免绿色桌面反光或低饱和区域只靠“亮色大块”规则骗过标签检测。真正的标签应该同时满足“内部白色内容面”这个强约束，边框/边线作为重要辅助加分。

如果找不到标签：

- 先检查是否像“标签被图像边缘截断”。
- 如果是：

```text
ERR_LABEL_INCOMPLETE
```

- 否则：

```text
ERR_LABEL_NOT_FOUND
```

## 4. 标签完整性和几何质量检查

代码位置：

```python
recognize_bin_labels.label_quad_touches_image_edge()
recognize_bin_labels.label_geometry_quality()
recognize_bin_labels.has_incomplete_label_candidate()
```

如果找到的标签四边形贴到图像边缘：

```text
ERR_LABEL_INCOMPLETE
```

如果标签四边形几何质量太差，例如透视太奇怪、角度不稳定：

```text
ERR_LABEL_GEOMETRY
```

默认阈值：

```text
min_label_quality = 0.55
```

注意：当前策略是保守的。标签不完整、严重贴边、形变太大时，不硬识别。

## 5. 标签两级透视矫正成正方形

代码位置：

```python
recognize_bin_labels.warp_label()
recognize_bin_labels.find_inner_white_label_quad()
recognize_bin_labels.warp_inner_white_label()
```

默认输出尺寸：

```python
DEFAULT_WARP_SIZE = (420, 420)
```

现在使用两级矫正。

第一级：找整张标签外层区域。

- 允许粉色或黑色边框参与定位。
- 目的是先把整张标签拉平成 `420x420`。

第二级：在第一级 warp 后的图里，重新找白色标签内容面。

- 目标是把粉色或黑色外边框切掉。
- 同样会用 CLAHE 辅助找白色内容面，增强阴影下的局部对比。
- 再次透视矫正，把白色内容面重新拉成 `420x420`。
- 后续数字 ROI 比例都基于这个白色内容面，而不是外边框。

这一点很重要：如果只做外层 warp，外边框角点的误差会让后续 ROI 歪掉；现在黑色数字块的 ROI 应该更贴合白色标签内容坐标系。

## 6. 尝试四个旋转方向

代码位置：

```python
recognize_bin_labels.rotate_warp()
evaluate_paddle_text_recognition.best_panel()
```

因为标签在图里可能是任意旋转，所以对已经矫正成正方形的标签图尝试：

```text
0, 90, 180, 270
```

每个方向都会独立做一次宽 ROI 裁剪和黑块检测，最后再选最可信的方向。

## 7. 裁数字区域宽 ROI

代码位置：

```python
extract_digit_crops.WIDE_DIGIT_ROI_RATIOS
recognize_bin_labels.crop_bin_roi_from_warp()
```

当前宽 ROI：

```python
WIDE_DIGIT_ROI_RATIOS = (0.02, 0.58, 0.56, 0.90)
```

含义：

```text
x0 = 标签宽度 * 0.02
y0 = 标签高度 * 0.58
x1 = 标签宽度 * 0.56
y1 = 标签高度 * 0.90
```

这个 ROI 是“搜索区域”，不是最终送进 OCR 的图。

设计原则：

- 只负责大致包住左下角黑色数字块。
- 应该比黑块大一些。
- 不能太高，否则会带进 QTY、条码、二维码等干扰。
- 不能太窄，否则黑块左边缘或右边 padding 会被截断。

如果 debug 图里看到宽 ROI 没有完整包住黑色数字块，就应该优先调整这里。

## 8. 在宽 ROI 里找真实黑色数字块

代码位置：

```python
extract_digit_crops.find_black_digit_panel()
```

处理方式：

1. 转灰度。
2. 生成一张 CLAHE 局部亮度均衡灰度图，用于辅助阴影下的阈值判断。
3. 用原始灰度和 CLAHE 灰度的较亮响应组合做候选判断：

```python
gray = max(gray_raw, gray_clahe)
```

注意：CLAHE 只用于黑块检测和白色笔画判断，不直接改变最终裁出来送进 PaddleOCR 的 RGB panel。

4. 用黑色阈值找暗区域：

```python
dark = gray < 105
```

5. 形态学 close/open，让黑色背景区域更连续。
6. 找轮廓。
7. 对候选黑块做筛选：

主要约束：

- 宽度不能太小。
- 高度不能太小。
- 长宽比要像横向黑色数字条。
- 面积比例要合理。
- 中心位置不能太靠右，避免选到二维码。
- 要有足够黑色背景。
- 要有白色或灰色数字笔画。
- 黑色背景在行/列方向上要比较连续。
- 如果像二维码或背景，拒绝。

单字符数字，比如 `0`，白色笔画占比很小，所以对“强连续黑色长条”有更低的白色笔画阈值。

如果四个旋转方向都找不到黑色数字块：

```text
ERR_BLACK_PANEL_NOT_FOUND
```

## 9. 黑块裁剪、矫正、去白边、归一化

代码位置：

```python
extract_digit_crops.rectify_panel_from_region()
extract_digit_crops.trim_panel_whitespace()
extract_digit_crops.normalize_panel_size()
```

黑块检测得到候选 `(x, y, w, h)` 后，会进一步处理：

1. 在候选框周围加少量 padding。
2. 得到 `region`。
3. 生成两个直接版本：

```text
raw_panel：直接 resize 到 200x90
trimmed_panel：先去白边，再 resize 到 200x90
```

4. 尝试用 `minAreaRect` 对黑块做二次矩形矫正。
5. 二次矫正后再去白边，归一化到：

```python
CANONICAL_PANEL_SIZE = (200, 90)
```

6. 比较不同版本的白色笔画保留情况。

这里有一个折中：

- 如果裁太干净，反光样本可能把白色数字笔画裁掉。
- 如果裁太宽，会带白边。

所以检测阶段可能保留稍宽版本，防止数字丢失。

## 10. 选择最可信旋转方向

代码位置：

```python
evaluate_paddle_text_recognition.best_panel()
extract_digit_crops._slot_split_score()
recognize_bin_labels.assess_digit_panel_quality()
```

每个旋转方向如果找到黑块，会计算一个分数。

分数主要参考：

- 数字笔画投影是否像目标位数。
- 槽位和白色笔画分布是否合理。
- panel 是否低对比、反光等。

如果有质量问题，会保留 `quality_code`，但不会直接让它输掉，因为真实低质量 panel 仍然可能比干净的错误物体更好。

最终选择分数最高的方向。

## 11. OCR 输入级水平裁剪与归一化

代码位置：

```python
evaluate_paddle_text_recognition.prepare_paddle_panel()
evaluate_paddle_text_recognition.trim_horizontal_white_margins()
evaluate_paddle_text_recognition.refine_inner_black_panel()
evaluate_paddle_text_recognition.crop_digit_text_region_horizontally()
extract_digit_crops.CANONICAL_TEXT_RIGHT
```

最终送进 PaddleOCR 前，会做保守的二次清理：

```python
trim_horizontal_white_margins(panel)
normalize_panel_size(...)
refine_inner_black_panel(...)
crop_digit_text_region_horizontally(...)
```

原因：

- 检测阶段已经选择并矫正了黑色数字块。
- 如果在 OCR 前再做一次强上下左右白边裁剪，容易把白色数字笔画、反光、局部亮斑误当成边界，导致只截到数字局部。
- 但是如果黑色方块左右漏进了白色标签边，PaddleOCR 容易把 `0` 识别成 `01` 等多余字符。
- 如果第一次黑块检测把标签外侧黑色边框也截进来了，后续基于投影的裁剪会误把外框当成黑色方块左边缘。
- PaddleOCR 需要看到完整黑块的横向结构，而不是被二次裁坏的局部碎片。

第一步仅根据黑色背景列去掉黑色方块外面的左右白边：

- 不裁上下边。
- 不按白色数字笔画裁。
- 保留黑色方块本身的右侧黑色 padding。
- 如果计算出的裁剪太激进，会回退到原 panel。

第二步在固定大小 panel 里，再找一次真实黑色数字矩形：

- 目标是去掉误截入的标签黑色/粉色外框、侧边黑条、白色缝隙。
- 用 CLAHE 亮度均衡图辅助判断暗区，让泛黄白边和阴影白边更容易被排除。
- 最终裁剪仍然从原始 RGB panel 上取，不直接把 CLAHE 图送给 OCR。
- 要求候选是横向连续黑色矩形，而不是竖向边框。
- 如果候选基本已经覆盖整张 panel，就不重复裁剪，避免插值损伤。
- 裁完后再做一次严格去白边和归一化。

第三步暂时不再裁数字串，直接把完整黑色数字块送进 PaddleOCR：

- 之前的槽位/投影裁剪还不够可靠，会把完整黑块裁成无数字区域。
- 当前先保留完整黑色长方块，避免破坏已经正确识别出的黑块。
- 真正的槽位/缝隙裁剪需要后续单独实现：先根据完整黑块坐标建立固定槽位，再只在槽位边界附近寻找可靠黑色缝隙。

实际送进 PaddleOCR 的图，是去掉左右白色漏边、剥掉误截入边框后的完整黑色数字块，统一保持在：

```text
200x90 黑色数字块
```

如果命令里用了：

```bash
--save-panel-dir ...
```

保存下来的 panel 就是这一步之后的 OCR 输入图。

## 12. PaddleOCR TextRecognition

代码位置：

```python
evaluate_paddle_text_recognition.recognize_panel()
```

当前用的是：

```python
from paddleocr import TextRecognition
recognizer = TextRecognition()
recognizer.predict(panel_path)
```

注意：当前不用完整的 `PaddleOCR().ocr()`，因为之前在本机 CPU 环境下遇到过底层 oneDNN/PIR 相关错误；`TextRecognition().predict(...)` 可以直接跑裁好的黑块。

## 13. OCR 输出归一化

代码位置：

```python
evaluate_paddle_text_recognition.normalize_ocr_digits()
```

PaddleOCR 可能把数字识别成形状相似的字母，例如：

```text
D/O/Q -> 0
I/l/| -> 1
S/s -> 5
B -> 8
Z/z -> 2
```

然后只保留数字字符：

```python
re.sub(r"\D", "", text)
```

例如 Paddle 原始输出：

```text
raw_text = D
```

会被归一化成：

```text
pred = 0
```

CSV 里会同时保存：

```text
pred      归一化后的数字结果
raw_text  PaddleOCR 原始文本
score     PaddleOCR 分数
```

## Debug 输出含义

脚本：

```bash
debug_paddle_eval.py
```

默认输出目录：

```text
/workspace/huangjie/pure_vision_detection/datasets/bin_number_debug
```

主要内容：

```text
bin_number_paddle_eval.csv
bin_number_paddle_errors.tsv
summary.md
errors_by_code/
debug_images/
paddle_panels/
```

其中：

- `paddle_panels/`：最终送进 PaddleOCR 的输入图。
- `debug_images/ERR_BLACK_PANEL_NOT_FOUND/`：四个旋转方向的宽 ROI 拼图。
  - 绿色框：该旋转方向成功通过规则的黑块。
  - 红框：该旋转方向最终没有找到可用黑块。
  - 橙色框：即使最终失败，也会画出“最像黑块的原始候选”，并标注 `raw_score` 和主要 reject reason，方便判断是阈值问题还是 ROI 问题。
  - 原图下面会附一张标签检测用 CLAHE 辅助图。
  - 每个宽 ROI 也会附对应的 CLAHE 辅助图，用于判断局部亮度均衡是否帮忙或制造伪影。
- `debug_images/ERR_LABEL_NOT_FOUND/`：标签检测失败前的原图/quad 等信息。
- `debug_images/ERR_LABEL_INCOMPLETE/`：标签被认为不完整时的原图/quad/质量信息。
- `debug_images/OCR_MISMATCH/`：OCR 识别错误时，保存最终 OCR 输入图，并标注 expected/pred/raw/score。
  - 原图会画出外层标签框和内白面框。
  - 选中的旋转宽 ROI 会画出最终选中的黑色方块框。
  - 选中宽 ROI 下方会附对应的 CLAHE 辅助图。
  - 同时保存“检测阶段黑色 panel”和“最终 PaddleOCR 输入 panel”，用于判断是黑块检测问题还是 OCR 本身问题。

## 当前各阶段对应的典型错误

### ERR_BLUR

原图太糊。应该重新拍照。

### ERR_LABEL_NOT_FOUND

没有找到完整标签四边形。可能原因：

- 标签太远。
- 标签太暗。
- 标签被遮挡。
- 标签和背景对比不够。
- 亮色检测没有把标签主体连起来。

### ERR_LABEL_INCOMPLETE

认为标签贴边或被图像边缘截断。可能原因：

- 目标标签确实不完整。
- 图中还有其他被截断的旧标签干扰。
- 边缘亮色物体被误判成标签残片。

### ERR_LABEL_GEOMETRY

标签四边形找到，但几何质量太差。可能原因：

- 拍摄角度过斜。
- 标签局部反光。
- 角点定位不稳定。

### ERR_BLACK_PANEL_NOT_FOUND

整张标签找到了，但在四个旋转方向的宽 ROI 里都没有找到可信黑色数字块。可能原因：

- 宽 ROI 没包住黑块。
- 黑块反光严重。
- 数字块被阈值误判成二维码/背景。
- 黑色背景不连续。

### OCR_MISMATCH

黑色数字块成功裁出并送进 PaddleOCR，但识别结果和文件夹名不同。此时优先查看：

```text
debug_images/OCR_MISMATCH/
paddle_panels/
```

如果 OCR 输入图已经很干净，那是 OCR 本身的问题；如果输入图仍有白边、缺字、变形，则应该回到前处理继续调。

## 调参优先级

如果识别失败，建议按这个顺序看：

1. `debug_images/ERR_LABEL_*`：标签是否找对。
2. `debug_images/ERR_BLACK_PANEL_NOT_FOUND`：宽 ROI 是否完整覆盖黑色数字块。
3. `paddle_panels/`：最终 OCR 输入是否是干净黑底白字。
4. `debug_images/OCR_MISMATCH`：Paddle 原始输出和分数。

常见调参位置：

```python
WIDE_DIGIT_ROI_RATIOS = (0.02, 0.58, 0.56, 0.90)
```

如果宽 ROI 太高，往下调 `y0/y1`。

如果黑块左边被截断，减小 `x0`。

如果黑块右侧 padding 被截断，增大 `x1`。

如果黑块底部被截断，增大 `y1`。
