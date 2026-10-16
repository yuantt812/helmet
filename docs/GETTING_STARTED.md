# 新手入门：从下载项目到完成第一次安全帽检测

这份教程面向第一次接触 Python 或目标检测的读者。你会先用现成模型完成图片检测，再尝试视频和自己的素材。最后一节介绍如何重新训练，第一次使用时可以先跳过。

仓库已经包含训练好的权重和演示视频，**运行检测不需要下载完整数据集，也不需要先训练模型**。没有独立显卡也能用 CPU 运行。

## 1. 先知道程序会做什么

你提供一张图片或一个视频，模型寻找画面中能看清的头部，输出两种类别：

| 类别 | 含义 | 结果中的颜色 |
| --- | --- | --- |
| `helmet` | 戴安全帽 | 绿色 |
| `no_helmet` | 未戴安全帽 | 红色 |

![项目已经生成的检测示例](../deliverables/demo_image.jpg)

这里的“模型权重”是训练得到的 `.pt` 文件，保存了模型学习到的参数。“推理”就是用这份权重检测新图片；“训练”则需要带标注的数据，让模型更新参数。

先按第 2～5 节完成一次图片检测，再继续第 6 节的视频检测。

## 2. 准备 Python 和 Git

需要 **Python 3.12** 和 **Git**。Python 用来运行代码，Git 用来下载项目及其固定版本的依赖源码。可以从 [Python 官网](https://www.python.org/downloads/)和 [Git 官网](https://git-scm.com/downloads)安装；Windows 安装 Python 时勾选添加到 PATH。

macOS 打开“终端”，Windows 打开 PowerShell，Linux 打开终端。下面的命令都在这些窗口中输入，一次执行一条；代码框中的内容可以直接复制。

macOS / Linux 检查版本：

```bash
python3.12 --version
git --version
```

Windows 检查版本：

```powershell
py -3.12 --version
git --version
```

能看到 `Python 3.12.x` 和 `git version ...` 即可继续。如果显示“找不到命令”，先检查是否安装成功，并关闭终端后重新打开。

## 3. 下载项目

在你希望保存项目的目录中执行：

```bash
git clone --recurse-submodules https://github.com/yuantt812/helmet.git
cd helmet
```

第一条命令下载项目，第二条进入项目目录。后面的所有命令都在这里运行。看到目录中有 `README.md`、`scripts/` 和 `requirements-train.txt`，就说明位置正确。

`--recurse-submodules` 会一并下载 `ultralytics/` 中的依赖源码，它被固定在本次实验使用的版本。若之前使用普通克隆下载过项目，在项目目录补执行：

```bash
git submodule update --init --recursive
```

GitHub 的“Download ZIP”不包含子模块源码，本教程使用 Git 克隆。

## 4. 创建环境并安装依赖

“虚拟环境”是专门放这个项目所需 Python 包的目录，名字为 `.venv`。它可以减少不同项目之间的依赖冲突。

### macOS / Linux

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

### Windows PowerShell

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

如果 PowerShell 提示禁止运行激活脚本，可以跳过激活，将后续命令开头的 `python` 换成 `.\.venv\Scripts\python.exe`。例如安装依赖时执行 `.\.venv\Scripts\python.exe -m pip install -r requirements-train.txt`。

### 安装与检查

激活后先确认 Python 路径：

```bash
python -c "import sys; print(sys.executable)"
```

输出应包含当前项目下的 `.venv`。然后安装依赖：

```bash
python -m pip install -r requirements-train.txt
```

这一步会下载 PyTorch、OpenCV 等包，并安装子模块中的 Ultralytics，下载量较大，需要等待。即使你暂时只做推理，也使用这份依赖文件安装。

安装完成后检查：

```bash
python -c "import torch, cv2; from ultralytics import YOLO; print('Environment OK'); print('PyTorch:', torch.__version__)"
```

看到 `Environment OK` 表示主要依赖可以导入。依赖文件记录的是原实验环境版本；CUDA 用户可能需要匹配显卡与驱动的 PyTorch 构建，首次运行可以先使用 CPU。

图片上的中文需要字体。macOS 和 Windows 会自动查找常用系统字体；Ubuntu 可先执行：

```bash
sudo apt install fonts-noto-cjk
```

其他系统可以通过后面介绍的 `--font` 指定支持中文的字体文件。

## 5. 完成第一次图片检测

### 提取演示视频的第一帧

仓库中已有的 `demo_image.jpg` 是画好检测框的成品。这里从自带视频提取一张没有叠加检测框的画面作为输入，不需要准备自己的图片。

在项目目录复制并执行下面这一整行命令：

```bash
python -c "from pathlib import Path; import cv2; Path('outputs').mkdir(exist_ok=True); cap=cv2.VideoCapture('deliverables/demo_input.mp4'); ok, frame=cap.read(); cap.release(); assert ok, 'Cannot read demo video'; assert cv2.imwrite('outputs/first_frame.jpg', frame); print('Saved outputs/first_frame.jpg')"
```

看到 `Saved outputs/first_frame.jpg` 后，可以在文件管理器中打开 `outputs/` 文件夹，查看刚保存的原始画面。

### 执行检测

```bash
python scripts/monitor_helmet.py --source outputs/first_frame.jpg --output outputs/first_result.jpg --device cpu
```

三个参数分别表示：

| 参数 | 作用 |
| --- | --- |
| `--source` | 输入文件的位置 |
| `--output` | 结果文件的位置，必须是尚不存在的新文件 |
| `--device cpu` | 使用 CPU，便于先确认流程能正常运行 |

执行结束后会输出一行 JSON，例如：

```json
{"output": "outputs/first_result.jpg", "frames": 1, "counts": [2, 0]}
```

这表示保存了 1 张图片，检出 2 个戴帽目标、0 个未戴帽目标。`counts` 的顺序始终是 `[戴帽, 未戴帽]`；具体数量以你的运行结果为准。

打开 `outputs/first_result.jpg`：如果看到了检测框、类别文字和顶部统计条，就已经完成第一次检测。第一帧只有绿色框也正常，并非每一帧都有两类目标。

**再次运行时请换一个输出名**，例如 `outputs/first_result_02.jpg`。程序会拒绝覆盖已有结果。

## 6. 检测视频

先处理演示视频的前 24 帧，确认本机能正常导出视频：

```bash
python scripts/monitor_helmet.py --source deliverables/demo_input.mp4 --output outputs/video_short.mp4 --device cpu --max-frames 24
```

成功后，终端输出中应出现 `frames` 和 `decoded`，两者均为 `24`。`decoded` 表示程序保存后重新读取成功的帧数。

然后可以处理完整视频：

```bash
python scripts/monitor_helmet.py --source deliverables/demo_input.mp4 --output outputs/video_full.mp4 --device cpu
```

完整演示视频是 15 秒、24 fps，共 360 帧。正常完成后 `frames` 和 `decoded` 应均为 `360`。处理耗时取决于设备，保存的视频保留输入帧率，**没有声音**。

视频导出需要 OpenCV 支持 H.264 编码。如果图片成功、视频出现 `H.264 encoder unavailable`，说明当前 OpenCV 构建不具备所需编码器；可以继续使用图片检测，视频导出需换用支持 H.264 编码的 OpenCV 构建。单独安装 `ffmpeg` 命令并不会自动改变已安装 OpenCV 的编码能力。

有图形桌面时，可以在命令末尾加 `--show` 边处理边预览，按 Q 退出。CPU 的处理速度可能低于原视频播放速度，预览不保证实时。

## 7. 换成自己的图片或视频

在项目目录新建 `inputs/` 文件夹，把素材放进去。文件名建议先使用简单的英文，例如 `site.jpg` 或 `site.mp4`。`inputs/` 存放自己的输入素材，`outputs/` 存放生成结果，这两个文件夹都已加入 Git 忽略规则。

图片示例：

```bash
python scripts/monitor_helmet.py --source inputs/site.jpg --output outputs/site_result.jpg --device cpu
```

视频示例：

```bash
python scripts/monitor_helmet.py --source inputs/site.mp4 --output outputs/site_result.mp4 --device cpu
```

路径包含空格时，用英文双引号包起来，例如 `--source "inputs/my site.jpg"`。输入图片可使用 JPG、PNG 等格式；视频输出需以 `.mp4` 结尾。

确认 CPU 能跑通后，可以去掉 `--device cpu`，让程序自动选择 Apple MPS、NVIDIA CUDA 或 CPU。也可以用 `--device mps` 或 `--device 0` 显式选择设备，前提是相应硬件和 PyTorch 环境可用。

摄像头是可选功能，未参与本次实验验收。有摄像头的电脑可以尝试：

```bash
python scripts/monitor_helmet.py --camera --source 0 --output outputs/camera_result.mp4 --show --max-frames 120
```

`0` 表示默认摄像头；系统可能要求授予终端摄像头访问权限。该命令最多处理 120 帧，也可按 Q 提前结束。

## 8. 怎样理解检测结果

框旁边的数字是模型对该次分类的置信度。例如 `0.87` 表示模型对这个预测较有把握，不能把它理解为“此人有 87% 的概率符合安全规定”。默认只显示置信度达到 **0.40** 的预测，该阈值由验证集选择。

顶部人数只统计当前帧中的检测框。同一个人连续出现在 100 帧中，不会被累计成 100 个不同的人；系统也没有跨帧追踪和去重计数功能。

“当前帧未检出未戴帽目标”表示模型没有检出该类目标。远处的小头部、遮挡和模糊画面都可能造成漏检，不能据此认定画面中所有人都戴了安全帽。

本次固定测试集上，阈值 0.40 时的结果如下：

| 类别 | Precision（查准率） | Recall（召回率） |
| --- | ---: | ---: |
| 戴帽 | 94.65% | 95.13% |
| 未戴帽 | 92.52% | 89.19% |

以未戴帽类为例：查准率衡量“模型报出的未戴帽目标中，有多少报对了”，召回率衡量“标注中的未戴帽目标中，有多少被找到了”。这些数值描述本次测试集，不能直接当作其他工地的效果保证。

`config/monitor.json` 保存了本次验证过的阈值、权重路径和 SHA256 校验值。入门阶段保持该配置即可，效果分析见[最终报告](../deliverables/工地安全帽佩戴检测实验报告.docx)和[实验结果分析](../runs/evaluation/yolo26n_100e_b8_20261015_145934_test_metrics/ANALYSIS.md)。

## 9. 常见问题

| 现象或报错 | 处理方法 |
| --- | --- |
| `can't open file ...scripts/monitor_helmet.py`，或找不到 requirements | 当前目录不对。进入包含 `README.md` 的 `helmet` 项目目录后重试。 |
| `No module named pip` | 在已选好的虚拟环境里执行 `python -m ensurepip --upgrade`，再安装依赖。 |
| 创建环境时提示 `ensurepip is not available` | Debian / Ubuntu 环境通常需要安装对应版本的 venv 支持包，例如 `python3.12-venv`，再重新创建虚拟环境。 |
| `No module named cv2`、`torch` 或无法导入 `YOLO` | 检查 `sys.executable` 是否指向 `.venv`，确认子模块已下载，再执行依赖安装命令。 |
| 依赖安装提示某个版本找不到 | 确认 Python 为 3.12；若在使用包镜像，检查镜像是否提供该版本。依赖快照不保证覆盖所有操作系统和硬件组合。 |
| `Chinese font required` | 安装中文字体，或在检测命令末尾加 `--font "字体文件的完整路径"`，可使用支持中文的 `.ttf` 或 `.ttc`。 |
| `Refusing to overwrite ...` | 输出已经存在，换一个输出文件名；训练和评价也需要使用新的运行名。 |
| `FileNotFoundError` 指向输入图片或视频 | 检查路径、扩展名和当前目录；含空格的路径加双引号。 |
| `Weight checksum differs from the validated model` | 权重与冻结配置不匹配。恢复仓库原始 `best.pt` 和配置，重新训练的模型需另外配置与验证。 |
| `H.264 encoder unavailable` | 先使用图片检测。视频导出需要支持 H.264 编码的 OpenCV 构建。 |
| 使用 MPS / CUDA 时报错 | 加 `--device cpu` 先排除加速环境问题。CPU 跑通后，再检查设备对应的 PyTorch 安装。 |
| 没有弹出窗口 | 默认只保存文件。到 `outputs/` 查看结果；视频需要预览时添加 `--show`。 |
| CPU 很慢或长视频迟迟没结束 | 先用 `--max-frames 24` 测试，确认流程正常后再处理完整视频。 |

查看程序支持的全部参数：

```bash
python scripts/monitor_helmet.py --help
```

## 10. 以后每次怎么启动

环境只需创建一次。以后打开新的终端，进入项目目录，再激活已有环境：

macOS / Linux：

```bash
cd helmet
source .venv/bin/activate
```

Windows PowerShell：

```powershell
cd helmet
.\.venv\Scripts\Activate.ps1
```

这里的 `cd helmet` 假设你当前在项目的上一级目录；如果项目保存在其他地方，应换成实际路径。激活后即可运行检测，无需每次重新安装依赖。使用结束可执行 `deactivate` 退出环境。

## 11. 进阶：重新训练与评价

这一节会下载完整数据并进行较长时间的计算。只使用现成模型做检测，可以停留在前面的步骤。

### 恢复数据

按[数据恢复说明](../datasets/hard-hat-workers/README.md#恢复数据与重新生成划分)下载 Kaggle 镜像版本 1，将 COCO 原始数据放到指定目录，然后运行：

```bash
python scripts/prepare_dataset.py audit
python scripts/prepare_dataset.py prepare
python scripts/prepare_dataset.py verify
```

`audit` 检查图片与标注，`prepare` 清洗并生成 YOLO 格式的数据，`verify` 核对生成结果。最终保留 7026 张图片：训练集 4916 张、验证集 1406 张、测试集 704 张。训练集用于学习，验证集用于选择模型和阈值，测试集用于最后评价。

已有 `yolo/` 时，`prepare` 会拒绝覆盖；不要重复运行它来尝试修复其他报错。完整处理规则见[预处理记录](../datasets/hard-hat-workers/PREPROCESSING.md)。

### 启动新训练

```bash
python scripts/train_helmet.py --run-name my_first_training
```

初次训练会尝试下载上游预训练权重 `yolo26n.pt`，需要能访问下载源。脚本固定设置最多 100 轮、batch=8、输入尺寸 640、seed=42，连续 20 轮未改进时提前结束。当前入口没有 `--epochs` 或 `--batch` 参数，如需改动，应先查看 `scripts/train_helmet.py` 中的 `options`。

训练结果保存在 `runs/helmet/my_first_training/`。其中 `weights/best.pt` 是新训练选出的最佳权重，`logs/status.json` 记录状态。运行名已经存在时，换一个名字。本次原实验在 MPS 上耗时约 3 小时 56 分钟，不能按这个时间估算其他设备，CPU 训练通常更慢。

### 评价新模型

```bash
python scripts/evaluate_helmet.py --run-name my_first_test --weights runs/helmet/my_first_training/weights/best.pt
```

结果写入 `runs/evaluation/my_first_test/`，包括 `metrics.json` 和曲线。**省略 `--weights` 会评价仓库自带的原实验权重**，不会自动改成你刚训练的模型。

新训练也不会自动替换检测程序的冻结模型。`monitor_helmet.py` 默认仍读取 `config/monitor.json`；若要切换模型，需要准备单独的配置、填入对应的权重路径和 SHA256，并重新验证适用阈值，再通过 `--config` 使用它。当前 `validate_monitor.py` 针对原实验权重编写，没有切换权重的命令行参数。

新生成的 `outputs/`、数据集和训练运行目录默认不会被 Git 提交。理解复现流程后，可以继续阅读[项目 README](../README.md)中的实验指标及[第三方来源说明](../THIRD_PARTY_NOTICES.md)。
