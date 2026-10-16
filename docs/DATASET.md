# 数据准备

使用 Hard Hat Workers 的 [Kaggle 镜像](https://www.kaggle.com/datasets/ammarnassanalhajali/hard-hat-workers)，本实验对应版本 1、COCO JSON 格式。原项目见 [Roboflow](https://universe.roboflow.com/joseph-nelson/hard-hat-workers)。

## 恢复原始数据

下载后解压到以下位置，保留原始图片文件名。如果有额外的外层目录，需要去掉该层：

```text
datasets/hard-hat-workers/raw/
  train/
    _annotations.coco.json
    *.jpg
  test/
    _annotations.coco.json
    *.jpg
```

原始数据共 7035 张。本项目使用 `helmet → 0 helmet`、`head → 1 no_helmet` 两类，不训练 `person` 和父类 `Workers`。原始说明标注 Public Domain，镜像元数据标注 Unknown；原说明见 [DATASET_LICENSE.txt](DATASET_LICENSE.txt)。

## 生成训练数据

先按 README 安装 `requirements.txt`，在项目根目录运行：

```bash
python scripts/prepare_dataset.py audit
python scripts/prepare_dataset.py prepare
python scripts/prepare_dataset.py verify
```

- `audit`：读取 COCO 标注，检查解码、尺寸和重复情况，生成检查记录。
- `prepare`：读取 [dataset_review.json](../config/dataset_review.json) 中的人工排除和场景归组决定，转换标签并划分数据。
- `verify`：核对导出图片和标签，检查坐标、类别及已识别相似组是否跨集合。

生成的训练入口为 `datasets/hard-hat-workers/yolo/data.yaml`。图片保持原始字节，框转换为归一化的 YOLO 标签；尺寸适配和训练增强由框架执行。

本次固定种子为 42，排除 9 张，剩余 7026 张划分如下：

| 集合 | 图片 | 戴帽框 | 未戴帽框 |
| --- | ---: | ---: | ---: |
| train | 4916 | 13799 | 4677 |
| val | 1406 | 3938 | 1334 |
| test | 704 | 1973 | 666 |

划分前联合 pHash、dHash 检查相似图片，按组分配，减少已识别的同场景图片跨集合。原始数据缺少可靠的场景 ID，不能保证识别所有相关场景。

`prepare` 会拒绝覆盖已有的 `yolo/`。本地已有生成数据时无需重复处理；确需重建，应先将已有输出移到其他位置。检查记录写入本地 `audit/`，训练结果写入本地 `runs/`，均不纳入 Git。
