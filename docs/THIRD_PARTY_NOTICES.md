# 第三方来源与许可证

## Ultralytics 与模型

上游：https://github.com/ultralytics/ultralytics

实验版本为 8.4.177，固定提交 `00cd4b8b187b6c7bc34776a6fe25a000f08b5fcc`，以 Git 子模块引用，未修改上游代码。上游使用 AGPL-3.0，项目代码的许可文本见 [LICENSE](../LICENSE)。

训练初始化使用上游 `yolo26n.pt`。仓库中的 `weights/best.pt` 是在 Hard Hat Workers 数据上微调得到的权重；路径与 SHA256 见 [monitor.json](../config/monitor.json)。使用模型时也应遵守 Ultralytics 的适用许可条款。

## Hard Hat Workers 数据

原项目：https://universe.roboflow.com/joseph-nelson/hard-hat-workers

下载镜像：https://www.kaggle.com/datasets/ammarnassanalhajali/hard-hat-workers

原始数据说明署名 Northeastern University - China，并标注 Public Domain；Kaggle 镜像的元数据许可证字段为 Unknown。原说明见 [DATASET_LICENSE.txt](DATASET_LICENSE.txt)，准备步骤见 [DATASET.md](DATASET.md)。完整数据不随仓库分发。

## 演示视频

源仓库：https://github.com/cosmo-wander-ai/cosmo-edge

源文件为 `data/test-video/Safety Helmet.mp4`。本项目裁剪 x=400、y=350、宽 1050、高 650 的作业区域，移除音轨，生成 24 fps、15 秒的 `examples/demo_input.mp4`，用于定性演示。

来源仓库许可证文本原样保存在 [VIDEO_LICENSE.txt](VIDEO_LICENSE.txt)。未发现针对该视频的单独媒体许可证；保留来源说明与仓库许可文本不代表已独立确认媒体权利归属。本项目代码许可证不重新授权第三方数据和媒体。
