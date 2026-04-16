#!/bin/bash
# ============================================================
# AutoDL 磁盘清理 + V13 环境部署一键脚本
# ============================================================
# 用法: 在 AutoDL GPU 服务器上执行
#   bash cleanup_and_setup.sh
# ============================================================
set -euo pipefail

# 使用说明: 在 AutoDL GPU 服务器上执行此脚本
# 用法: bash cleanup_and_setup.sh [project_root]
# 例如: bash cleanup_and_setup.sh /root/quantum
PROJECT_ROOT="${1:-$(pwd)}"

echo "=========================================="
echo "  Step 1: 磁盘空间诊断"
echo "=========================================="
df -h /
echo ""
echo "Top 10 占用:"
du -sh /root/* 2>/dev/null | sort -rh | head -10
echo ""

echo "=========================================="
echo "  Step 2: 清理 pip 缓存"
echo "=========================================="
pip cache purge 2>/dev/null || true
rm -rf /root/.cache/pip /tmp/pip-* 2>/dev/null || true
echo "pip 缓存已清理"

echo "=========================================="
echo "  Step 3: 清理 conda 缓存"
echo "=========================================="
conda clean --all -y 2>/dev/null || true
echo "conda 缓存已清理"

echo "=========================================="
echo "  Step 4: 清理旧的训练日志和临时文件"
echo "=========================================="
rm -rf /root/.local/share/Trash 2>/dev/null || true
rm -rf /tmp/* 2>/dev/null || true
rm -f /root/install.log 2>/dev/null || true
# 清理旧的 nohup 输出
rm -f /root/nohup.out 2>/dev/null || true
echo "临时文件已清理"

echo "=========================================="
echo "  Step 5: 清理 PyTorch 编译缓存"
echo "=========================================="
rm -rf /root/.cache/torch_extensions 2>/dev/null || true
rm -rf /root/.triton 2>/dev/null || true
echo "PyTorch 缓存已清理"

echo "=========================================="
echo "  Step 6: 查看清理后的空间"
echo "=========================================="
df -h /
echo ""

echo "=========================================="
echo "  Step 7: 安装依赖 (轻量版)"
echo "=========================================="
cd "$PROJECT_ROOT"

# 先确保基础包可用，跳过已有的大包
pip install --no-cache-dir gymnasium==1.2.3 2>&1 | tail -3
pip install --no-cache-dir qiskit==2.3.1 2>&1 | tail -3
pip install --no-cache-dir qiskit-aer==0.17.2 2>&1 | tail -3
pip install --no-cache-dir torch_geometric==2.7.0 2>&1 | tail -3
pip install --no-cache-dir networkx==3.6.1 rustworkx 2>&1 | tail -3
pip install --no-cache-dir scipy matplotlib pillow 2>&1 | tail -3

echo ""
echo "=========================================="
echo "  Step 8: 验证环境"
echo "=========================================="
python -c "
import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')
import gymnasium; print(f'Gymnasium: {gymnasium.__version__}')
import qiskit; print(f'Qiskit: {qiskit.__version__}')
import torch_geometric; print(f'PyG: {torch_geometric.__version__}')
import networkx; print(f'NetworkX: {networkx.__version__}')
print('所有依赖验证通过!')
"

echo ""
echo "=========================================="
echo "  Step 9: 拉取最新代码"
echo "=========================================="
cd "$PROJECT_ROOT"
git pull 2>&1 || echo "git pull 失败，请检查远程仓库"

echo ""
echo "=========================================="
echo "  清理完成! 磁盘空间:"
echo "=========================================="
df -h /
echo ""
echo "接下来运行训练:"
echo "  cd "$PROJECT_ROOT" && bash run_train_v13.sh"
