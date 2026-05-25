
<p align="center">
  <img src="npyviewer_128x128.png" alt="screenshot">
</p>

# NPYViewer 1.28
###  A simple GUI tool that provides multiple ways to load and view the contents of .npy files containing 2D and 1D NumPy arrays.

#### Plot 3-column 2D numpy arrays containing 3D coordinates as 3D point clouds
![screenshot](screenshots/ScreenShot1.png)
#### Plot 2D numpy arrays as grayscale images
![screenshot](screenshots/ScreenShot2.png)
#### Visualize heightmaps stored as 2D numpy arrays
![screenshot](screenshots/ScreenShot3.png) 
![screenshot](screenshots/ScreenShot4.png)
#### Visualize time series data stored as 1D numpy arrays
![screenshot](screenshots/ScreenShot5.png)
#### Visualize adjacency matrices (saved in .npy arrays) as directional edge weighted graphs
![screenshot](screenshots/ScreenShot7.png)
#### Print numpy arrays in terminal
![screenshot](screenshots/ScreenShot6.png)


### Installation:
* Original development in Ubuntu 20.04 and Python 3.8.8
* Also tested on Windows 10 and Ubuntu 22.04
* pip3 install -r requirements.txt


### Execution:
* python3 NPYViewer.py


### Current Features:
* Open and view .npy files that contain 2D NumPy arrays and lists, as spreadsheets
* Convert .npy files to .csv format
* Convert .csv files to .npy format
* Export .npy files as .mat files (compatible with MATLAB and Octave)
* Plot 2D numpy arrays as grayscale images
* Plot 2D numpy arrays containing 3D coordinates as 3D point clouds
* Visualize heightmaps stored as 2D numpy arrays
* Visualize time series data stored as 1D numpy arrays
* Supports loading .npy files as command line arguments (e.g., python3 NPYViewer.py sample_npy_files/timeseries.npy)
* Visualize adjacency matrices (saved in .npy arrays) as directional edge weighted graphs
* Print numpy arrays in terminal through the use of the -noGUI argument (e.g., python NPYViewer.py sample_npy_files/timeseries.npy -noGUI)
* GUI developed using PyQT5


### TODO:
* Add/Remove Rows & Columns
* Copy/Paste Rows & Columns
* Data search and filtering
* Modify content datatypes 
* Handle data with more than 2 dimensions



### Changes since last version:
* Added application icon
* Fixed Bug: "View as Time Series" option was hidden in the "Functionalities" menu

---

## Landmark Sequence Viewer (`view_landmarks_sequence.py`)

NPYViewer 一次只能查看单个 `.npy` 文件。为了播放**目录下成千上万帧**的 `(N, 3)` 关键点 / 点云序列（例如 MediaPipe FaceMesh 478 点、人体骨骼、Depth landmark），仓库里附带了一个独立的轻量播放脚本 `view_landmarks_sequence.py`。

它仅依赖 `numpy` + `matplotlib`（不需要 PyQt5），适合快速浏览动捕 / 面部追踪导出的逐帧数据。

### 功能特性

* 自动按文件名排序，把目录下所有 `.npy` 当作时间序列循环播放
* 支持降采样 (`--step`) 与起止区间 (`--start` / `--end`)，7000+ 帧也能流畅播放
* 全局包围盒预估，视图范围固定，不会逐帧抖动缩放
* 内置坐标轴翻转 (`--flip-x/y/z`)，自动修正 MediaPipe / 图像坐标系（Y 向下）下的脸部倒立问题
* 键盘 + 鼠标交互：暂停、单帧步进、跳首尾、旋转视角

### 安装

```bash
pip install numpy matplotlib
```

### 用法

```bash
# 默认 30 fps 播放整个目录
python view_landmarks_sequence.py "Y:\path\to\Landmarks_dir"

# 60 fps + 每 3 帧抽 1 帧（适合 7000+ 帧的长序列）
python view_landmarks_sequence.py "Y:\path\to\Landmarks_dir" --fps 60 --step 3

# 只播放第 1000 ~ 2000 帧
python view_landmarks_sequence.py "Y:\path\to\Landmarks_dir" --start 1000 --end 2000

# 关闭默认 Y 翻转，使用原始坐标
python view_landmarks_sequence.py "Y:\path\to\Landmarks_dir" --no-flip-y
```

### 交互按键

| 按键 | 作用 |
|---|---|
| `Space` | 暂停 / 继续播放 |
| `←` / `→` | 上一帧 / 下一帧（自动暂停） |
| `Home` / `End` | 跳到第一帧 / 最后一帧 |
| `q` | 退出 |
| 鼠标拖拽 | 旋转 3D 视角 |
| 滚轮 | 缩放 |

### 命令行参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `directory` | — | 包含 `.npy` 文件的目录（必填，位置参数）|
| `--pattern` | `*.npy` | 文件匹配模式 |
| `--fps` | `30` | 播放帧率 |
| `--step` | `1` | 每隔多少帧取一个，用于降采样 |
| `--start` | `0` | 起始帧索引 |
| `--end` | `-1` | 结束帧索引（`-1` 表示最后） |
| `--point-size` | `4` | 散点大小 |
| `--point-color` | `red` | 点颜色（matplotlib 颜色字符串） |
| `--no-equal-axis` | off | 关闭 x/y/z 等比例，用各轴范围撑满视图 |
| `--flip-y` / `--no-flip-y` | **on** | Y 轴翻转。默认开启以修正图像坐标系下脸部倒立 |
| `--flip-x` | off | X 轴翻转（左右镜像） |
| `--flip-z` | off | Z 轴翻转（前后） |

### 输入数据约定

* 目录下任意数量的 `.npy` 文件，按文件名升序当作帧顺序
* 每个文件形状须为 `(N, 3)` 或 `(N, ≥3)`（只取前 3 列作为 XYZ）
* `N` 在帧之间可以不一致；`dtype` 任意数值类型即可

### 朝向不对怎么调？

脚本默认开启 `--flip-y`，因为 MediaPipe / 图像坐标系是 Y 向下，否则 3D 视图里脸会倒立。如果实际数据朝向仍不对：

| 现象 | 加这个参数 |
|---|---|
| 脸 / 物体上下颠倒 | （默认已开 `--flip-y`，若关掉了再加回来）|
| 左右反了（镜像） | `--flip-x` |
| 朝向后面（前后反了） | `--flip-z` |
| 想看完全原始坐标 | `--no-flip-y` |

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

