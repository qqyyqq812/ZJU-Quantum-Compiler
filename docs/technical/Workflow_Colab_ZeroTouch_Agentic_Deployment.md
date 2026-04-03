# Zero-Touch Colab Agentic Deployment Workflow
## 零接触式 Colab Agent 自动化部署工作流范式

> **受众说明**：本规范文档面向未来接管、维护或拓展各类科研实验训练池的 AI Agent。用于规范 Agent 与人类 User 之间在 Cloud GPU (如 Colab) 训练任务中的交接与协作模式。

### 📌 核心理念：User 的“零接触” (Zero-Touch)
在复杂的深度学习或离线训练任务中，**严禁要求 User 直接在 Colab 中复制粘贴、修改核心配置参数或书写 Python 脚本片段。** 
因为 Cloud 运行时的易失性极高，任何未沉淀至 Git 代码库的手动修改都极易引发版本灾难和数据污染（例如：不知不觉读取了旧版本的模型权重）。

User 的唯一职责是：**点开链接、点击 `Run All` (全部运行) 、查看图表、提供算力与授权。**
Agent 的核心职责是：**接管一切底层的脏活累活，将实验设计打包成自动化容器或引导脚本（Jupyter Notebook API），通过 GitHub 实现云端分发。**

---

### 📦 Agent 标准分发工作流四步曲

#### 1. 逻辑与表现的绝对分离 (Decoupling)
- **底层脚本 (`.py` 和 `.sh`)**：所有超参数设定（如 RL 的 `penalty`, `obs_dim`, `soft-mask`）、网络架构和核心训练循环，必须全部写死在仓库内的 python 或 bash 脚本中。
- **启动层 (`.ipynb`)**：仅仅作为一个**薄层的云端引导发射台**。本文件内的源码应当具有极强的防御性：
  - 自动注入浏览器 JS 防断联代码 (`setInterval click`)。
  - 自动检测 GPU 型号并拉取对应版本的 `torch-scatter/sparse` 库支持。
  - **强制环境清洗**：在执行前，必须包含 `%cd /content` 和 `!rm -rf Repo` 的强杀命令，并重新 `git clone`，确保读到永远是 Agent 最新推上 GitHub 的干净代码。

#### 2. 云盘硬链接机制 (Safe Checkpoints)
- 模型在执行时必须使用防崩溃设计。而在 Colab 中，必须解决 `/content` 重启即销毁的问题。
- **强制要求**：Agent 必须在 Notebook 中编写 Google Drive 挂载语句 (`drive.mount`)。
- **规避污染**：遇到新版本的重大技术革新（如 V10 切换 V11），Agent **绝不可** 复用旧版的目录（如 `models/v9_tokyo20`）。必须在挂载驱动器后创建一个全新的子目录（如 `v11_dqn_tokyo20`）。
- **软链接欺骗**：通过 `!ln -s /content/drive/MyDrive/.../models/v11 models_local`，让底层脚本仍以为自己在写本地文件，从而实现解耦。

#### 3. 动态画图截断 (Interrupt-Plot-Resume)
- 训练过程长达数小时，User 的不安感会急剧上升。绝不可让图表代码必须等到 50000 局结束才画出来。
- **解决范式**：
  1. 在底层的训练脚本里（如 `train.py`），设置高频写盘机制（每 100 步将短效 `history.json` 覆盖更新写入云盘目录）。
  2. 在 Notebook 的最下方建立一个独立的 `Plot` 画图格子。
  3. 引号提示 User：“您可以随时手动终止上面的训练（Stop），运行这里的格子看图，看完图回去重新点播放（Run），利用底层脚本读取最新 *.pt 文件的断点续传机制，进度不会丢失一丝一毫。”

#### 4. Repo 级一键推流 (One-Shot Push)
- 当 User 提出改进要求或修正 bug 后：
  1. Agent 在本地 (`/home/` 空间) 修改逻辑代码、修正 `Colab_Master.ipynb` 文件。
  2. Agent 自行执行 `git add . && git commit && git push`。
  3. Agent 会在回复里通知 User：“代码已全线接管并推送，您只需刷新云端的那个页面重新跑即可。”

---

### 🛡️ 给后来 Agent 的警戒
**不要相信任何云端的缓存。** 
1. 曾经的一个重大 Bug（模型在 40~70 次 SWAP 高位震荡），即使本地改了 `.sh` 文件，因为模型启动读取了存在云盘里的上一版脏数据的 checkpoint，导致所有验证数据全盘作废。
2. 永远记住为不同架构变更开辟全新的隔离 `SAVE_DIR` 进行物理斩断。
