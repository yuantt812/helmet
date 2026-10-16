# 工地安全帽检测

使用 YOLO26n 检测“戴安全帽”和“未戴安全帽”，支持图片、视频和可选的摄像头输入。

- [训练流程说明](docs/GETTING_STARTED.md)：了解这个项目如何准备数据、训练和评价。
- [数据准备](docs/DATASET.md)：需要重新训练时阅读。

## 运行

需要 Python 3.12 和 Git。在项目根目录执行：

```bash
git clone --recurse-submodules https://github.com/yuantt812/helmet.git
cd helmet
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/monitor_helmet.py --source examples/demo_input.mp4 --output outputs/demo.mp4
```

Windows PowerShell 使用 `py -3.12 -m venv .venv` 创建环境，用 `.venv\Scripts\Activate.ps1` 激活。已普通克隆的项目需补执行 `git submodule update --init --recursive`。

仓库包含训练好的 `weights/best.pt` 和演示输入，不用先训练。更换 `--source` 即可检测自己的素材；图片输出用 `.jpg` 或 `.png`，视频输出用 `.mp4`。输出文件需使用新名称，程序会拒绝覆盖。

默认自动选择 MPS、CUDA 或 CPU，也可加 `--device cpu`。中文标注需要系统字体，缺少时用 `--font` 指定；Ubuntu 可安装 `fonts-noto-cjk`。视频保存需要支持 H.264 编码的 OpenCV 构建。

## 工程结构

```text
scripts/        数据处理、训练、评价和检测入口
config/         推理配置、数据清洗决定
docs/           项目说明与来源许可
examples/       演示输入
weights/        训练好的检测权重
ultralytics/    固定版本的框架子模块
requirements.txt
```

数据集和新产生的训练结果保存在本地，不纳入仓库。

## 重新训练

先按[数据说明](docs/DATASET.md)恢复数据，然后执行：

```bash
python scripts/train_helmet.py --run-name my_training
python scripts/evaluate_helmet.py --run-name my_test --weights runs/helmet/my_training/weights/best.pt
```

训练参数为最多 100 轮、batch=8、输入尺度 640。结果写入 `runs/`，运行名不可重复。检测程序默认使用仓库自带权重，重新训练不会自动替换它。

## 结果与许可

原实验在第 80 轮提前停止，第 60 轮最佳。测试集 mAP50 为 96.51%，mAP50–95 为 67.15%。演示阈值 0.40 由验证集选择，其他场景的效果仍需验证。

项目代码使用 [AGPL-3.0](LICENSE)。模型、数据和视频来源见[第三方说明](docs/THIRD_PARTY_NOTICES.md)。
