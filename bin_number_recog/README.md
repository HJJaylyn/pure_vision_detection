# Bin Number Recognition

当前定版的标签数字识别工具集。运行脚本保留在根目录，方便直接执行；详细说明和历史产物分开保存。

## 目录结构

### 核心识别流程

- `recognize_bin_labels.py`: 标签定位、四边形矫正、旋转方向选择和基础识别接口。
- `extract_digit_crops.py`: 黑色数字方块检测、四边形边缘、透视矫正、槽位/投影处理。
- `evaluate_paddle_text_recognition.py`: PaddleOCR 批量评估和 OCR 输入预处理。

### Debug 与训练工具

- `debug_paddle_eval.py`: 按错误类别生成完整中间图、统计和 TSV 日志。
- `debug_digit_segmentation.py`: 数字区域分割和槽位调试。
- `cnn_digit_model.py`: 数字 CNN 模型定义。
- `train_digit_cnn.py`: CNN 训练脚本。
- `infer_digit_cnn.py`: CNN 推理脚本。

### 说明文档

- `docs/README_bin_label_recognition.md`: 标签识别使用说明。
- `docs/PADDLE_OCR_PREPROCESS_PIPELINE.md`: 原图到 PaddleOCR 输入的完整处理链路。

### 历史产物

- `artifacts/bin_label_results.csv`: 旧版识别结果。
- `artifacts/bin_label_results_smoke.csv`: 旧版 smoke test 结果。
- `artifacts/debug_bin_labels/`: 旧版 Tesseract/OpenCV debug 产物。

新的评估结果建议放在数据集目录下，例如：

```bash
/workspace/huangjie/miniconda3/envs/vision_recog/bin/python \
  debug_paddle_eval.py \
  --dataset-dir /workspace/huangjie/pure_vision_detection/datasets/bin_number \
  --output-dir /workspace/huangjie/pure_vision_detection/datasets/bin_number_debug
```

脚本之间使用同目录模块导入，因此核心 `.py` 文件暂时不要继续移动到子目录，否则直接运行命令会找不到依赖。

当前版本基线已包含：

- 白色标签四边形定位与二次白色内容面矫正；
- 四方向旋转 ROI 搜索；
- Canny/连续边缘黑色四边形检测；
- 过曝压暗灰度和自适应阈值辅助图；
- 保留首位数字的透视裁剪保护；
- 上下白边和左右白边分开处理；
- 槽位/投影裁剪只作用于最终数字字符串区域；
- PaddleOCR 错误分类与中间结果保存。

单张图片识别

使用最终 PaddleOCR 流程识别单张图片：

```bash
cd /workspace/huangjie/pure_vision_detection/bin_number_recog
/workspace/huangjie/miniconda3/envs/vision_recog/bin/python \
  recognize_single_bin_image.py \
  /workspace/huangjie/Franka/data/img/right_tcp_20260715_212141_294.jpg
```

脚本会输出 JSON。成功时 `code=OK`，数字在 `bin_number` 字段；标签定位、黑色方块定位或 OCR 未完成时，会输出对应错误码。若已知数字位数，可增加 `--digit-count 4`，帮助槽位评分，但不是必需参数。
