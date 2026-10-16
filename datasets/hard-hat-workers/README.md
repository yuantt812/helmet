# Hard Hat Workers 原始数据

- 下载日期：2026-10-15
- 原项目：https://universe.roboflow.com/joseph-nelson/hard-hat-workers
- 下载镜像：https://www.kaggle.com/datasets/ammarnassanalhajali/hard-hat-workers
- Kaggle 镜像版本：1；压缩包内 Roboflow 版本：v2 raw
- 原导出日期：2020-04-30；格式：COCO JSON
- 原说明声明仅进行 EXIF 自动方向调整，未做图像增强。
- 原 README.dataset.txt 声明 Public Domain；Kaggle 元数据许可证字段为 Unknown，原始说明均已保留。

## 已验证的文件统计

| 集合 | 图片 | helmet 框 | head 框 | person 框 |
| --- | ---: | ---: | ---: | ---: |
| train | 5269 | 14884 | 4874 | 473 |
| test | 1766 | 4863 | 1803 | 142 |
| 合计 | 7035 | 19747 | 6677 | 615 |

压缩包 CRC 检查通过，COCO 记录的全部图片均存在。原说明称 7041 张，实际压缩包和标注均为 7035 张，以实际统计为准。类别表中的 Workers 是无标注的父类。

## 文件与后续处理

本地 `raw/` 保留原图、COCO 标注及原说明；Git 仓库仅包含来源说明，不包含完整原始数据。下载压缩包在完整解压并校验后已清理。manifest.json 保留原始下载时的文件清单、SHA256、来源和类别统计，其中 ZIP 条目仅作历史记录；source-metadata.json 保留下载来源的元数据。

已完成图片解码检查、重复与近似图片检查、抽样语义核对、类别转换及分组重新划分。保留 7026 张，输出位于 `yolo/`，训练入口为 `yolo/data.yaml`。完整方法、统计、复现命令及抽样和相似检测的局限见 [预处理记录](PREPROCESSING.md)。生成的 `yolo/`、机器记录 `audit/records.json` 和审查拼图不纳入 Git；人工审查、划分清单及汇总指标保留。

## 恢复数据与重新生成划分

1. 从上述 Kaggle 镜像下载 **版本 1、COCO JSON 格式**，保留原始文件名。其他版本或 YOLO 导出格式不能直接复现本实验。
2. 解压到 `datasets/hard-hat-workers/raw/`，确认目录结构如下（如果压缩包带外层目录，需要移除该层）：

   ```text
   raw/
     train/
       _annotations.coco.json
       *.jpg
     test/
       _annotations.coco.json
       *.jpg
   ```

3. 在项目根目录、激活虚拟环境后运行：

   ```bash
   python -m pip install -r requirements-preprocess.txt
   python scripts/prepare_dataset.py audit
   python scripts/prepare_dataset.py prepare
   python scripts/prepare_dataset.py verify
   ```

`audit` 重新生成 `records.json`，`prepare` 读取已提交的 `audit/manual_review.json`，复用人工排除与场景归组决定，并生成与 YAML 位置相对的数据路径。`verify` 应通过 7026 张图片的检查；最终划分为 train 4916、val 1406、test 704。可用 `git diff -- datasets/hard-hat-workers/audit/` 对照冻结记录。

`prepare` 会拒绝覆盖已存在的 `yolo/`；本地已有数据时无需重复处理。要重建，应先把该目录移到备份位置，再执行上述步骤。完整推理演示只使用仓库中的权重和视频，不需要此数据恢复流程。
