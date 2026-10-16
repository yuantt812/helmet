# 第三方来源与许可证

## Ultralytics 与 YOLO26

- 上游：https://github.com/ultralytics/ultralytics
- 实验版本：8.4.177，提交 `00cd4b8b187b6c7bc34776a6fe25a000f08b5fcc`
- 许可证：AGPL-3.0，完整文本见 [LICENSE](LICENSE) 及子模块中的 `ultralytics/LICENSE`。
- 仓库以 Git 子模块引用上游，未修改上游代码。
- 训练初始化使用上游 `yolo26n.pt`。最终 `best.pt` 在 Hard Hat Workers 数据上微调得到，路径与 SHA256 见 [config/monitor.json](config/monitor.json)。使用模型时也应遵守 Ultralytics 的适用许可条款。

项目代码按根目录 AGPL-3.0 许可证分发；本说明不重新授权第三方数据与媒体。

## Hard Hat Workers 数据

原项目：https://universe.roboflow.com/joseph-nelson/hard-hat-workers

下载镜像：https://www.kaggle.com/datasets/ammarnassanalhajali/hard-hat-workers

原始数据说明署名 Northeastern University - China，并标注 Public Domain；Kaggle 镜像的元数据许可证字段为 Unknown。保留原始说明及元数据，不将镜像字段解读为新的授权。来源、哈希和处理步骤见 [datasets/hard-hat-workers/README.md](datasets/hard-hat-workers/README.md)。完整原始数据和生成的图片/标签不纳入仓库；演示检测图片来自该数据集。

## 演示视频

源仓库：https://github.com/cosmo-wander-ai/cosmo-edge

源文件：`data/test-video/Safety Helmet.mp4`。本项目裁剪画面、移除音轨并生成检测叠加视频，处理参数和哈希见 [deliverables/acceptance.json](deliverables/acceptance.json)。

来源仓库许可证文本原样保存在 [deliverables/VIDEO_LICENSE.txt](deliverables/VIDEO_LICENSE.txt)。未发现针对该视频的单独媒体许可证；保留来源说明与仓库许可文本不代表已独立确认媒体权利归属。
