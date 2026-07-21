#!/usr/bin/env python3
"""Collect RGB and depth images from an Intel RealSense camera.

Controls:
  r: start continuous recording
  s: stop recording
  c: capture one frame while stopped
  q or ESC: quit
"""

from __future__ import annotations

import argparse
import csv
import json
import select
import sys
import termios
import time
import tty
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

try:
    import cv2
except ImportError as exc:
    raise SystemExit(
        "Cannot import cv2. Please run this script in an environment with opencv-python installed."
    ) from exc

try:
    import numpy as np
except ImportError as exc:
    raise SystemExit("Cannot import numpy. Please run this script in an environment with numpy installed.") from exc

try:
    import pyrealsense2 as rs
except ImportError as exc:
    raise SystemExit(
        "Cannot import pyrealsense2. Please run this script in an environment "
        "with Intel RealSense SDK Python bindings installed."
    ) from exc


@dataclass
class CapturePaths:
    root: Path
    rgb_dir: Path
    depth_dir: Path | None
    depth_vis_dir: Path | None
    depth_overlay_dir: Path | None
    metadata_csv: Path
    metadata_jsonl: Path


class TerminalKeyReader:
    """Non-blocking single-key reader for no-preview mode."""

    def __init__(self) -> None:
        self._fd: int | None = None
        self._old_settings: list[int | bytes] | None = None

    def __enter__(self) -> "TerminalKeyReader":
        if sys.stdin.isatty():
            self._fd = sys.stdin.fileno()
            self._old_settings = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
        return self

    def __exit__(self, *_args: object) -> None:
        if self._fd is not None and self._old_settings is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_settings)

    def read_key(self) -> str | None:
        if self._fd is None:
            return None
        readable, _, _ = select.select([sys.stdin], [], [], 0)
        if readable:
            return sys.stdin.read(1).lower()
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect aligned RGB/depth frames from an Intel RealSense camera."
    )
    parser.add_argument(
        "--output-dir",
        required=False,
        type=Path,
        help="Dataset output directory. It will contain rgb/, depth/, and metadata files.",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List connected RealSense devices and exit.",
    )
    parser.add_argument(
        "--serial",
        default=None,
        help="RealSense serial number. If omitted, the first connected camera is used.",
    )
    parser.add_argument("--width", type=int, default=640, help="RGB/depth width.")
    parser.add_argument("--height", type=int, default=480, help="RGB/depth height.")
    parser.add_argument("--fps", type=int, default=30, help="Camera FPS.")
    parser.add_argument(
        "--interval",
        type=float,
        default=0.2,
        help="Seconds between saved frames while recording. Use 0 to save every frame.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=None,
        help="Starting 6-digit index. By default, continues after existing files.",
    )
    parser.add_argument(
        "--rgb-ext",
        choices=("png", "jpg"),
        default="png",
        help="RGB image format. Depth is always saved as 16-bit PNG.",
    )
    parser.add_argument(
        "--no-depth",
        action="store_true",
        help="Only save RGB images. Useful if the depth stream cannot start.",
    )
    parser.add_argument(
        "--flat-rgb",
        action="store_true",
        help="With --no-depth, save RGB images directly under output-dir instead of output-dir/rgb/.",
    )
    parser.add_argument(
        "--no-align",
        action="store_true",
        help="Do not align depth to RGB. By default, depth is aligned to color.",
    )
    parser.add_argument(
        "--save-depth-vis",
        action="store_true",
        help="Also save colorized depth preview images under depth_vis/.",
    )
    parser.add_argument(
        "--save-depth-overlay",
        action="store_true",
        help="Also save RGB/depth blended preview images under depth_overlay/.",
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Disable OpenCV preview window and read keys from terminal.",
    )
    return parser.parse_args()


def list_realsense_devices() -> list[dict[str, str]]:
    devices = []
    for dev in rs.context().query_devices():
        item = {
            "name": dev.get_info(rs.camera_info.name),
            "serial": dev.get_info(rs.camera_info.serial_number),
        }
        if dev.supports(rs.camera_info.firmware_version):
            item["firmware"] = dev.get_info(rs.camera_info.firmware_version)
        if dev.supports(rs.camera_info.usb_type_descriptor):
            item["usb"] = dev.get_info(rs.camera_info.usb_type_descriptor)
        devices.append(item)
    return devices


def prepare_output_dirs(
    root: Path,
    save_depth: bool,
    save_depth_vis: bool,
    save_depth_overlay: bool,
    flat_rgb: bool,
) -> CapturePaths:
    root.mkdir(parents=True, exist_ok=True)
    rgb_dir = root if flat_rgb else root / "rgb"
    depth_dir = root / "depth" if save_depth else None
    depth_vis_dir = root / "depth_vis" if save_depth and save_depth_vis else None
    depth_overlay_dir = root / "depth_overlay" if save_depth and save_depth_overlay else None
    rgb_dir.mkdir(parents=True, exist_ok=True)
    if depth_dir is not None:
        depth_dir.mkdir(parents=True, exist_ok=True)
    if depth_vis_dir is not None:
        depth_vis_dir.mkdir(parents=True, exist_ok=True)
    if depth_overlay_dir is not None:
        depth_overlay_dir.mkdir(parents=True, exist_ok=True)
    return CapturePaths(
        root=root,
        rgb_dir=rgb_dir,
        depth_dir=depth_dir,
        depth_vis_dir=depth_vis_dir,
        depth_overlay_dir=depth_overlay_dir,
        metadata_csv=root / "metadata.csv",
        metadata_jsonl=root / "metadata.jsonl",
    )


def find_next_index(paths: CapturePaths, rgb_ext: str, explicit_start: int | None) -> int:
    if explicit_start is not None:
        return explicit_start

    indices: list[int] = []
    folders: list[tuple[Path, str]] = [(paths.rgb_dir, f"*.{rgb_ext}")]
    if paths.depth_dir is not None:
        folders.append((paths.depth_dir, "*.png"))
    for folder, pattern in folders:
        for path in folder.glob(pattern):
            if path.stem.isdigit():
                indices.append(int(path.stem))
    return (max(indices) + 1) if indices else 0


def append_metadata_header(path: Path) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "index",
                "timestamp_unix",
                "rgb_path",
                "depth_path",
                "depth_scale_m_per_unit",
                "width",
                "height",
                "fps",
                "serial",
            ]
        )


def video_stream_intrinsics(profile: rs.pipeline_profile, stream: rs.stream) -> dict[str, float | int | str]:
    stream_profile = profile.get_stream(stream).as_video_stream_profile()
    intr = stream_profile.get_intrinsics()
    return {
        "width": intr.width,
        "height": intr.height,
        "ppx": intr.ppx,
        "ppy": intr.ppy,
        "fx": intr.fx,
        "fy": intr.fy,
        "model": str(intr.model),
        "coeffs": [float(x) for x in intr.coeffs],
    }


def write_camera_info(
    paths: CapturePaths,
    profile: rs.pipeline_profile,
    args: argparse.Namespace,
    depth_scale: float | None,
    serial: str,
) -> None:
    info = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "serial": serial,
        "width": args.width,
        "height": args.height,
        "fps": args.fps,
        "rgb_ext": args.rgb_ext,
        "save_depth": not args.no_depth,
        "flat_rgb": args.flat_rgb,
        "rgb_layout": "output_dir" if args.flat_rgb else "rgb_subdir",
        "align_depth_to_color": not args.no_depth and not args.no_align,
        "depth_scale_m_per_unit": depth_scale,
        "color_intrinsics": video_stream_intrinsics(profile, rs.stream.color),
    }
    if not args.no_depth:
        info["depth_intrinsics"] = video_stream_intrinsics(profile, rs.stream.depth)
    (paths.root / "camera_info.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def create_pipeline(args: argparse.Namespace) -> tuple[rs.pipeline, rs.pipeline_profile, float | None]:
    devices = list_realsense_devices()
    if not devices:
        raise RuntimeError("No Intel RealSense camera found.")

    print("Connected RealSense devices:")
    for dev in devices:
        print("  - " + json.dumps(dev, ensure_ascii=False))

    pipeline = rs.pipeline()
    config = rs.config()
    if args.serial:
        config.enable_device(args.serial)
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    if not args.no_depth:
        config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)

    profile = pipeline.start(config)
    depth_scale = None
    if not args.no_depth:
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = float(depth_sensor.get_depth_scale())
        print(f"Depth scale: {depth_scale} meter/unit")
    return pipeline, profile, depth_scale


def colorize_depth(depth_image: np.ndarray) -> np.ndarray:
    """Convert a raw uint16 depth frame into a human-readable color image."""
    valid = depth_image[depth_image > 0]
    if valid.size == 0:
        depth_8u = np.zeros(depth_image.shape, dtype=np.uint8)
    else:
        low = float(np.percentile(valid, 1))
        high = float(np.percentile(valid, 99))
        if high <= low:
            high = low + 1.0
        normalized = (depth_image.astype(np.float32) - low) * (255.0 / (high - low))
        depth_8u = np.clip(normalized, 0, 255).astype(np.uint8)
        depth_8u[depth_image == 0] = 0
    colorized = cv2.applyColorMap(depth_8u, cv2.COLORMAP_JET)
    colorized[depth_image == 0] = (0, 0, 0)
    return colorized


def draw_status(preview: np.ndarray, recording: bool, frame_index: int, saved_count: int) -> np.ndarray:
    status = "REC" if recording else "STOP"
    color = (0, 0, 255) if recording else (255, 255, 255)
    text = f"{status} | next {frame_index:06d} | saved {saved_count} | r start, s stop, c one, q quit"
    cv2.putText(preview, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
    return preview


def save_sample(
    paths: CapturePaths,
    index: int,
    color_image: np.ndarray,
    depth_image: np.ndarray | None,
    args: argparse.Namespace,
    depth_scale: float | None,
    serial: str,
) -> None:
    stem = f"{index:06d}"
    rgb_path = paths.rgb_dir / f"{stem}.{args.rgb_ext}"
    depth_path = paths.depth_dir / f"{stem}.png" if depth_image is not None and paths.depth_dir is not None else None

    cv2.imwrite(str(rgb_path), color_image)
    if depth_image is not None and depth_path is not None:
        cv2.imwrite(str(depth_path), depth_image)
        if paths.depth_vis_dir is not None:
            depth_vis = colorize_depth(depth_image)
            cv2.imwrite(str(paths.depth_vis_dir / f"{stem}.jpg"), depth_vis)
            if paths.depth_overlay_dir is not None:
                overlay = cv2.addWeighted(color_image, 0.55, depth_vis, 0.45, 0)
                cv2.imwrite(str(paths.depth_overlay_dir / f"{stem}.jpg"), overlay)

    timestamp = time.time()
    row = [
        index,
        f"{timestamp:.6f}",
        str(rgb_path.relative_to(paths.root)),
        str(depth_path.relative_to(paths.root)) if depth_path else "",
        "" if depth_scale is None else f"{depth_scale:.12g}",
        args.width,
        args.height,
        args.fps,
        serial,
    ]
    with paths.metadata_csv.open("a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)

    record = {
        "index": index,
        "timestamp_unix": timestamp,
        "rgb_path": str(rgb_path.relative_to(paths.root)),
        "depth_path": str(depth_path.relative_to(paths.root)) if depth_path else None,
        "depth_scale_m_per_unit": depth_scale,
        "width": args.width,
        "height": args.height,
        "fps": args.fps,
        "serial": serial,
    }
    with paths.metadata_jsonl.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def get_key_from_preview(delay_ms: int = 1) -> str | None:
    key = cv2.waitKey(delay_ms) & 0xFF
    if key == 255:
        return None
    if key == 27:
        return "q"
    return chr(key).lower()


def handle_key(key: str | None, recording: bool) -> tuple[bool, bool, bool]:
    """Return recording, save_one, should_quit."""
    if key == "r":
        return True, False, False
    if key == "s":
        return False, False, False
    if key == "c":
        return recording, True, False
    if key == "q":
        return recording, False, True
    return recording, False, False


def main() -> int:
    args = parse_args()
    if args.list_devices:
        devices = list_realsense_devices()
        if not devices:
            print("No Intel RealSense camera found.")
            return 1
        print(json.dumps(devices, ensure_ascii=False, indent=2))
        return 0

    if args.output_dir is None:
        raise SystemExit("--output-dir is required unless --list-devices is used.")
    if args.flat_rgb and not args.no_depth:
        raise SystemExit("--flat-rgb requires --no-depth. RGB-D datasets keep rgb/ and depth/ subfolders.")
    if args.no_depth and (args.save_depth_vis or args.save_depth_overlay):
        raise SystemExit("--save-depth-vis and --save-depth-overlay require depth. Remove them or omit --no-depth.")

    paths = prepare_output_dirs(
        args.output_dir.expanduser().resolve(),
        not args.no_depth,
        args.save_depth_vis or args.save_depth_overlay,
        args.save_depth_overlay,
        args.flat_rgb,
    )
    append_metadata_header(paths.metadata_csv)
    frame_index = find_next_index(paths, args.rgb_ext, args.start_index)

    pipeline: rs.pipeline | None = None
    try:
        pipeline, profile, depth_scale = create_pipeline(args)
        serial = profile.get_device().get_info(rs.camera_info.serial_number)
        write_camera_info(paths, profile, args, depth_scale, serial)
        align = None if args.no_depth or args.no_align else rs.align(rs.stream.color)

        print(f"Saving dataset to: {paths.root}")
        print(f"Starting index: {frame_index:06d}")
        print("Controls: r=start continuous save, s=stop, c=single frame, q=quit")

        # Let auto exposure settle before saving the first frames.
        for _ in range(max(5, args.fps // 2)):
            pipeline.wait_for_frames()

        recording = False
        saved_count = 0
        last_save_time = 0.0
        window_name = "realsense rgbd collector"

        terminal_context = TerminalKeyReader() if sys.stdin.isatty() else nullcontext()
        with terminal_context as terminal_keys:
            while True:
                frames = pipeline.wait_for_frames()
                if align is not None:
                    frames = align.process(frames)

                color_frame = frames.get_color_frame()
                depth_frame = None if args.no_depth else frames.get_depth_frame()
                if not color_frame or (not args.no_depth and not depth_frame):
                    print("Skipped an incomplete frame.")
                    continue

                color_image = np.asanyarray(color_frame.get_data())
                depth_image = None
                if depth_frame is not None:
                    depth_image = np.asanyarray(depth_frame.get_data())

                key = None
                if not args.no_preview:
                    preview = color_image.copy()
                    if depth_image is not None:
                        depth_preview = colorize_depth(depth_image)
                        preview = np.hstack((preview, depth_preview))
                    cv2.imshow(window_name, draw_status(preview, recording, frame_index, saved_count))
                    key = get_key_from_preview()

                if key is None and terminal_keys is not None:
                    key = terminal_keys.read_key()

                recording, save_one, should_quit = handle_key(key, recording)
                if should_quit:
                    break

                now = time.time()
                due_by_interval = args.interval <= 0 or now - last_save_time >= args.interval
                if save_one or (recording and due_by_interval):
                    save_sample(paths, frame_index, color_image, depth_image, args, depth_scale, serial)
                    print(f"Saved {frame_index:06d}")
                    frame_index += 1
                    saved_count += 1
                    last_save_time = now

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        if pipeline is not None:
            pipeline.stop()
        if not args.no_preview:
            cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
