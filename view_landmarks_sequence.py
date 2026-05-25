#!/usr/bin/env python3
"""
view_landmarks_sequence.py
--------------------------
逐帧可视化一个目录下大量 (N, 3) 形状的 .npy 关键点 / 点云序列文件。
适用于 MediaPipe FaceMesh、人体骨骼、Depth landmark 等按帧导出的数据。

设计目标:
    - 一行命令打开目录, 自动把文件按文件名排序当作时间序列;
    - 支持降采样 / 起止区间, 7000+ 帧也能流畅播放;
    - 处理图像坐标系 (Y 向下, Z 向前) 与 matplotlib 3D 坐标系的差异,
      避免脸 / 物体倒立;
    - 全局包围盒固定, 视图不会逐帧抖动缩放;
    - 仅依赖 numpy + matplotlib, 不需要 GUI 工具链。

用法示例:
    python view_landmarks_sequence.py "Y:\\path\\to\\Landmarks_dir"
    python view_landmarks_sequence.py "Y:\\path\\to\\Landmarks_dir" --fps 60 --step 3
    python view_landmarks_sequence.py "Y:\\path\\to\\Landmarks_dir" --start 1000 --end 2000

交互:
    Space      暂停 / 播放
    Left/Right 上一帧 / 下一帧 (自动暂停)
    Home/End   跳到第一帧 / 最后一帧
    q          退出
鼠标拖拽可旋转 3D 视角, 滚轮缩放。
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  注册 3D projection


# ---------------------------------------------------------------------------
# 命令行参数
# ---------------------------------------------------------------------------

@dataclass
class FlipConfig:
    """三轴翻转开关。"""
    x: bool = False
    y: bool = True   # 默认翻转 Y, 修正 MediaPipe / 图像坐标系倒立
    z: bool = False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="逐帧可视化目录下 (N,3) 形状的 .npy 关键点序列。",
    )
    p.add_argument("directory", help="包含 .npy 文件的目录")
    p.add_argument("--pattern", default="*.npy",
                   help="文件匹配模式 (默认 *.npy)")
    p.add_argument("--fps", type=float, default=30.0,
                   help="播放帧率 (默认 30)")
    p.add_argument("--step", type=int, default=1,
                   help="每隔多少帧取一个, 用于降采样 (默认 1)")
    p.add_argument("--start", type=int, default=0,
                   help="起始帧索引 (默认 0)")
    p.add_argument("--end", type=int, default=-1,
                   help="结束帧索引, -1 表示最后 (默认 -1)")
    p.add_argument("--point-size", type=float, default=4.0,
                   help="点大小 (默认 4)")
    p.add_argument("--point-color", default="red",
                   help="点颜色, matplotlib 颜色字符串 (默认 red)")
    p.add_argument("--no-equal-axis", action="store_true",
                   help="不强制 x/y/z 等比例, 用各自范围撑满视图")

    # 坐标轴翻转: 默认开启 --flip-y, 让 MediaPipe / 图像坐标的脸立起来
    p.add_argument("--flip-y", dest="flip_y", action="store_true", default=True,
                   help="翻转 Y 轴 (默认开启, 让 MediaPipe 坐标的脸立起来)")
    p.add_argument("--no-flip-y", dest="flip_y", action="store_false",
                   help="关闭 Y 轴翻转, 使用原始坐标")
    p.add_argument("--flip-x", action="store_true", default=False,
                   help="翻转 X 轴 (左右镜像)")
    p.add_argument("--flip-z", action="store_true", default=False,
                   help="翻转 Z 轴 (前后)")

    return p.parse_args()


# ---------------------------------------------------------------------------
# 数据加载与几何工具
# ---------------------------------------------------------------------------

def apply_flip(pts: np.ndarray, flip: FlipConfig) -> np.ndarray:
    """对 (N,3) 点云按需做轴翻转, 返回新数组。"""
    out = pts.astype(np.float64, copy=True)
    if flip.x:
        out[:, 0] = -out[:, 0]
    if flip.y:
        out[:, 1] = -out[:, 1]
    if flip.z:
        out[:, 2] = -out[:, 2]
    return out


def load_frame(path: str, flip: FlipConfig) -> np.ndarray:
    """加载单个 .npy 帧并做翻转, 返回 (N,3) 数组。"""
    data = np.load(path, allow_pickle=True)
    if data.ndim != 2 or data.shape[1] < 3:
        raise ValueError(f"文件 {path} 形状不符: {data.shape}, 期望 (N,>=3)")
    return apply_flip(data[:, :3], flip)


def load_file_list(directory: str, pattern: str,
                   step: int, start: int, end: int) -> List[str]:
    """扫描目录, 按文件名排序后按 start/end/step 切片。"""
    files = sorted(glob.glob(os.path.join(directory, pattern)))
    if not files:
        print(f"[错误] 在 {directory} 下找不到匹配 {pattern} 的文件",
              file=sys.stderr)
        sys.exit(1)

    if end == -1 or end > len(files):
        end = len(files)
    files = files[start:end:step]
    print(f"[信息] 共 {len(files)} 帧将被可视化 "
          f"(start={start}, end={end}, step={step})")
    return files


def compute_global_bounds(files: List[str], flip: FlipConfig,
                          sample: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """从抽样文件估算全局 xyz 包围盒, 避免逐帧自动缩放抖动。"""
    n = min(sample, len(files))
    idx = np.linspace(0, len(files) - 1, num=n).astype(int)
    mins = np.full(3, np.inf)
    maxs = np.full(3, -np.inf)
    for i in idx:
        try:
            pts = load_frame(files[i], flip)
        except ValueError:
            continue
        mins = np.minimum(mins, pts.min(axis=0))
        maxs = np.maximum(maxs, pts.max(axis=0))

    if not np.all(np.isfinite(mins)):
        mins = np.array([-1.0, -1.0, -1.0])
        maxs = np.array([1.0, 1.0, 1.0])
    pad = (maxs - mins) * 0.05
    return mins - pad, maxs + pad


# ---------------------------------------------------------------------------
# 播放器
# ---------------------------------------------------------------------------

class LandmarksPlayer:
    """基于 matplotlib FuncAnimation 的逐帧 3D 点云播放器。"""

    def __init__(self, files: List[str], flip: FlipConfig,
                 fps: float, point_size: float, point_color: str,
                 equal_axis: bool):
        self.files = files
        self.flip = flip
        self.fps = fps
        self.idx = 0
        self.playing = True

        bounds_min, bounds_max = compute_global_bounds(files, flip)
        print(f"[信息] 全局包围盒 min={bounds_min}, max={bounds_max}  "
              f"(flip_x={flip.x}, flip_y={flip.y}, flip_z={flip.z})")

        first_pts = load_frame(files[0], flip)

        self.fig = plt.figure(figsize=(8, 8))
        self.ax = self.fig.add_subplot(111, projection="3d")

        self.scat = self.ax.scatter(
            first_pts[:, 0], first_pts[:, 1], first_pts[:, 2],
            s=point_size, c=point_color, marker="o",
        )
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        self.ax.set_xlim(bounds_min[0], bounds_max[0])
        self.ax.set_ylim(bounds_min[1], bounds_max[1])
        self.ax.set_zlim(bounds_min[2], bounds_max[2])
        if equal_axis:
            try:
                # matplotlib >= 3.3, 让 3D 坐标轴按数据比例显示, 不挤压
                self.ax.set_box_aspect(tuple(bounds_max - bounds_min))
            except Exception:
                pass

        self.title = self.ax.set_title("")

        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

        interval_ms = max(1.0, 1000.0 / fps)
        # cache_frame_data=False: 大序列下避免艺术家对象被全量缓存
        self.anim = FuncAnimation(
            self.fig, self._update, interval=interval_ms,
            blit=False, cache_frame_data=False,
        )
        self._render(0)

    @property
    def n(self) -> int:
        return len(self.files)

    def _render(self, i: int) -> None:
        path = self.files[i]
        try:
            pts = load_frame(path, self.flip)
        except ValueError as e:
            print(f"[警告] 跳过 {path}: {e}", file=sys.stderr)
            return
        # 3D scatter 没有 set_data, 直接更新 _offsets3d
        self.scat._offsets3d = (pts[:, 0], pts[:, 1], pts[:, 2])
        self.title.set_text(f"[{i + 1}/{self.n}] {os.path.basename(path)}")
        self.fig.canvas.draw_idle()

    def _update(self, _frame):
        if self.playing:
            self.idx = (self.idx + 1) % self.n
            self._render(self.idx)
        return (self.scat,)

    def _on_key(self, event):
        key = event.key
        if key == " ":
            self.playing = not self.playing
            print("[播放]" if self.playing else "[暂停]")
        elif key == "right":
            self.playing = False
            self.idx = (self.idx + 1) % self.n
            self._render(self.idx)
        elif key == "left":
            self.playing = False
            self.idx = (self.idx - 1) % self.n
            self._render(self.idx)
        elif key == "home":
            self.playing = False
            self.idx = 0
            self._render(self.idx)
        elif key == "end":
            self.playing = False
            self.idx = self.n - 1
            self._render(self.idx)
        elif key == "q":
            plt.close(self.fig)

    def run(self) -> None:
        print("交互: Space 暂停/播放, Left/Right 单帧, "
              "Home/End 跳首尾, q 退出")
        plt.show()


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    if not os.path.isdir(args.directory):
        print(f"[错误] 目录不存在: {args.directory}", file=sys.stderr)
        sys.exit(1)

    files = load_file_list(args.directory, args.pattern,
                           args.step, args.start, args.end)

    flip = FlipConfig(x=args.flip_x, y=args.flip_y, z=args.flip_z)

    player = LandmarksPlayer(
        files=files,
        flip=flip,
        fps=args.fps,
        point_size=args.point_size,
        point_color=args.point_color,
        equal_axis=not args.no_equal_axis,
    )
    player.run()


if __name__ == "__main__":
    main()
