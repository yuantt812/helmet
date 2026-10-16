# 发布到 GitHub

项目根目录是独立 Git 仓库；`ultralytics/` 是固定提交的上游子模块。以下命令均在项目根目录执行。

## 上传范围

提交代码、README、配置、来源与划分记录、最终报告、演示媒体、最终 best.pt、训练曲线和标准评价指标。

`.gitignore` 排除虚拟环境、`.config/`、密钥文件、原始图片与 COCO 标注、生成的 YOLO 数据、缓存、预训练权重及新的运行输出。仅放行本次冻结实验的指定结果。忽略规则不删除本地文件。若需要更新正式实验，先明确调整该白名单和配置中的权重哈希。

## 本地检查与首次提交

```bash
git status --short
git diff --cached --stat
git submodule status
python -m compileall -q scripts
git add .
git diff --cached --check
git diff --cached --stat
git commit -m "Initial helmet detection experiment"
```

子模块应显示提交 `00cd4b8b187b6c7bc34776a6fe25a000f08b5fcc`；不要把上游仓库的 origin 当作本项目的上传地址。数据集和 `.venv/` 不应出现在待提交清单。AGPL-3.0 适用于项目代码；第三方数据与媒体来源见 [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md)。

## 创建仓库并推送

在 GitHub 创建一个空仓库（例如 `helmet`），可按需要选择公开或私有，不勾选自动添加 README、LICENSE 或 `.gitignore`。将下列地址中的 `OWNER/helmet` 替换成实际仓库路径：

```bash
git remote add origin https://github.com/OWNER/helmet.git
git push -u origin main
```

也可以在已登录 GitHub CLI 的环境中，用以下命令创建公开仓库并推送；`OWNER/helmet` 同样需要替换：

```bash
gh repo create OWNER/helmet --public --source=. --remote=origin --push
```

两种创建方式选择一种即可。仓库默认分支为 `main`，当前文件无需 Git LFS。

## 克隆验收

```bash
git clone --recurse-submodules https://github.com/OWNER/helmet.git
cd helmet
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-train.txt
python scripts/monitor_helmet.py --source deliverables/demo_input.mp4 --output outputs/check.mp4
```

字体与 H.264 编码环境要求见 README。GitHub Actions 在 Ubuntu / Python 3.12 上检查语法、冻结模型哈希和 CPU 首帧推理；不运行完整训练、数据评价或视频编码验收。
