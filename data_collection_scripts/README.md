# RealSense RGB-D Image Collector

用于给纯视觉检测模型采集图片数据。脚本会从 Intel RealSense 同时读取 RGB 和深度帧，并按相同 6 位编号分开保存：

```text
dataset_name/
  rgb/
    000000.png
    000001.png
  depth/
    000000.png
    000001.png
  depth_vis/
    000000.jpg
  depth_overlay/
    000000.jpg
  camera_info.json
  metadata.csv
  metadata.jsonl
```

`rgb/000000.png` 和 `depth/000000.png` 是同一时刻采到的一组帧。深度图保存为 16-bit PNG，像素值是 RealSense 原始深度单位；实际米制距离可以用 `metadata.csv` 里的 `depth_scale_m_per_unit` 换算：

```text
distance_meter = depth_pixel_value * depth_scale_m_per_unit
```

`camera_info.json` 会保存相机序列号、RGB/Depth 分辨率、fps、depth scale、RGB/Depth 内参，以及深度是否对齐到 RGB。

如果只采 RGB，可以使用 `--no-depth --flat-rgb`，这时图片会直接放在输出目录根下：

```text
digits_rgb_001/
  000000.jpg
  000001.jpg
  camera_info.json
  metadata.csv
  metadata.jsonl
```

## 采集电脑环境检查

目标采集电脑是 `huangjie@10.0.10.39`。脚本复制过去后，先确认 RealSense 和 Python 环境。

### 1. 检查 USB 设备

```bash
lsusb | grep -i -E 'intel|realsense|8086'
```

如果有 RealSense，通常能看到 Intel 设备。也可以看系统日志：

```bash
dmesg | tail -n 50
```

RealSense 最好接 USB 3.0 口。USB 2.0 可能导致深度流/RGB-D 同时开失败。

### 2. 检查 RealSense 命令行工具

如果装了 librealsense 工具：

```bash
rs-enumerate-devices
realsense-viewer
```

`rs-enumerate-devices` 能看到相机信息，说明系统层面基本没问题。`realsense-viewer` 可以用来确认 RGB 和 Depth 画面。

### 3. 检查 Python 包

进入准备用来采集的 Python 环境后：

```bash
python3 - <<'PY'
import cv2
import numpy as np
import pyrealsense2 as rs
print("cv2", cv2.__version__)
print("numpy", np.__version__)
print("pyrealsense2 ok")
print("devices", len(rs.context().query_devices()))
PY
```

如果 `pyrealsense2` import 失败，需要安装 RealSense Python binding。常见方式：

```bash
python3 -m pip install pyrealsense2 opencv-python numpy
```

如果 pip 装不上或找不到设备，需要安装 Intel librealsense 系统包，优先参考采集电脑系统版本对应的 librealsense 安装方式。

### 4. 用脚本列出设备

```bash
python3 collect_realsense_rgbd.py --list-devices
```

能看到 serial 后，就可以开始采集。多相机时用 `--serial` 指定。

## Run

先进入有 `pyrealsense2` 和 `opencv-python` 的 Python 环境，然后运行：

```bash
cd /path/to/pure_vision_detection/data_collection_scripts

python3 collect_realsense_rgbd.py \
  --output-dir /path/to/datasets/test_001
```

常用参数：

```bash
# 指定相机序列号
--serial 123456789

# 控制保存频率，默认 0.2 秒一张；0 表示每帧都存
--interval 0.1

# 只采 RGB
--no-depth

# 只采 RGB 时，把图片直接存在 output-dir 下，不再套 rgb/ 文件夹
--flat-rgb

# RGB 用 jpg，深度仍然固定为 png
--rgb-ext jpg

# 不弹 OpenCV 窗口，从终端读取按键
--no-preview

# 查看相机列表，不采集
--list-devices

# 额外保存给人看的深度伪彩色图
--save-depth-vis

# 额外保存 RGB/depth 混合图，用来检查对齐
--save-depth-overlay
```

## Controls

- `r`: 开始连续保存
- `s`: 停止连续保存
- `c`: 停止状态下单独保存一帧
- `q` 或 `ESC`: 退出

如果输出目录里已经有旧图片，脚本默认会自动从已有最大编号继续往后存。

## 推荐采集命令

如果需要预览窗口，并按 `r/s/c/q` 控制：

```bash
python3 collect_realsense_rgbd.py \
  --output-dir /home/huangjie/pure_vision_detection_datasets/plastic_stack_001 \
  --width 640 \
  --height 480 \
  --fps 30 \
  --interval 0.2 \
  --save-depth-vis \
  --save-depth-overlay
```

如果通过 SSH 跑、没有图形界面，用终端按键：

```bash
python3 collect_realsense_rgbd.py \
  --output-dir /home/huangjie/pure_vision_detection_datasets/plastic_stack_001 \
  --no-preview \
  --interval 0.2
```

如果只训练 RGB 检测模型，不需要深度：

```bash
python3 collect_realsense_rgbd.py \
  --output-dir /home/huangjie/pure_vision_detection_datasets/digits_rgb_001 \
  --no-depth \
  --flat-rgb \
  --rgb-ext jpg \
  --interval 0.2
```

这样保存出来就是：

```text
/home/huangjie/pure_vision_detection_datasets/digits_rgb_001/
  000000.jpg
  000001.jpg
  ...
  camera_info.json
  metadata.csv
  metadata.jsonl
```

## 从本机同步到采集电脑

如果从当前机器复制脚本到采集电脑：

```bash
rsync -av --info=progress2 \
  /workspace/huangjie/pure_vision_detection/ \
  huangjie@10.0.10.39:/home/huangjie/pure_vision_detection/
```

采完数据后，从采集电脑同步回当前机器示例：

```bash
rsync -av --info=progress2 \
  huangjie@10.0.10.39:/home/huangjie/pure_vision_detection_datasets/ \
  /workspace/huangjie/pure_vision_detection/datasets/
```

## 注意事项

- RGB 和 depth 是分开保存的，但同编号是一组样本。
- `--flat-rgb` 只适合和 `--no-depth` 一起用；RGB-D 采集仍然建议保留 `rgb/`、`depth/` 子文件夹。
- 默认会把 depth 对齐到 RGB；如果对齐导致启动失败，可加 `--no-align`。
- depth PNG 是 16-bit 原始深度单位，不是彩色可视化图。看图用 `--save-depth-vis` 生成的 `depth_vis/*.jpg`。
- 如果要检查 RGB 和 depth 是否对齐，用 `--save-depth-overlay` 生成的 `depth_overlay/*.jpg`。
- 如果输出目录已有图片，会自动从最大 6 位编号继续保存，避免覆盖。
- 如果按键没反应，确认 OpenCV 预览窗口是当前焦点；SSH 无窗口时用 `--no-preview`。
