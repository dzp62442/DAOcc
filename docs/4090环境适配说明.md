# 环境配置指南（Ubuntu 22 + RTX 4090）

## 环境说明

| 项目 | 版本 |
|------|------|
| OS | Ubuntu 22.04 |
| GPU | RTX 4090 (sm_89, Ada Lovelace) |
| CUDA Toolkit | 11.8（conda 内安装，提供 nvcc，支持 sm_89） |
| PyTorch | 1.13.1+cu117（自带 CUDA 11.7 运行时） |
| Python | 3.8 |
| mmcv-full | 1.7.2 |
| mmdet | 2.28.2 |

### 版本选型说明

**为什么 nvcc 用 11.8 而不是 11.7？**
sm_89（RTX 4090）是 CUDA 11.8 才引入的架构，nvcc 11.7 无法编译 sm_89 内核。
参考：[CUDA Toolkit Release Notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/)（11.8 New Features 列出 Ada Lovelace/sm_89）

**为什么 PyTorch 用 cu117 而不是 cu118？**
cu118 是 mmcv-full 1.x 系列不支持的版本，cu117 是其支持的最高版本。
PyTorch cu117 自带 CUDA 11.7 运行时库，编译自定义算子时调用的是 conda 里的 nvcc 11.8，两者互不冲突。

---

## 第一步：创建 conda 环境

```shell
conda create -n daocc python=3.8 -y
conda activate daocc
```

---

## 第二步：安装 PyTorch 1.13.1+cu117

```shell
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 -f https://download.pytorch.org/whl/torch_stable.html
```

验证：

```shell
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# 期望输出：1.13.1+cu117 True
```

---

## 第三步：安装 mmcv-full 1.7.2

```shell
pip install mmcv-full==1.7.2 -f https://download.openmmlab.com/mmcv/dist/cu117/torch1.13.0/index.html
```

---

## 第四步：安装其他依赖

```shell
pip install -r requirements.txt
```

---

## 第五步：为 RTX 4090 添加 sm_89 编译支持

`setup.py` 的 nvcc gencode 列表缺少 sm_89，需手动添加一行（nvcc 11.8 支持此架构）。

编辑 `setup.py`，在 `sm_86` 那行之后添加：

```python
"-gencode=arch=compute_89,code=sm_89",  # RTX 4090 (Ada Lovelace)
```

修改后该段应为：

```python
extra_compile_args["nvcc"] = extra_args + [
    "-D__CUDA_NO_HALF_OPERATORS__",
    "-D__CUDA_NO_HALF_CONVERSIONS__",
    "-D__CUDA_NO_HALF2_OPERATORS__",
    "-gencode=arch=compute_70,code=sm_70",
    "-gencode=arch=compute_75,code=sm_75",
    "-gencode=arch=compute_80,code=sm_80",
    "-gencode=arch=compute_86,code=sm_86",
    "-gencode=arch=compute_89,code=sm_89",  # RTX 4090
]
```

---

## 第六步：编译安装 DAOcc

```shell
FORCE_CUDA=1 python setup.py develop
```

---

## 验证安装

```shell
python -c "
import torch, mmcv, mmdet
print('torch   :', torch.__version__)
print('mmcv    :', mmcv.__version__)
print('mmdet   :', mmdet.__version__)
print('CUDA    :', torch.cuda.is_available())
print('GPU     :', torch.cuda.get_device_name(0))
"
```

期望输出：

```
torch   : 1.13.1+cu117
mmcv    : 1.7.2
mmdet   : 2.28.2
CUDA    : True
GPU     : NVIDIA GeForce RTX 4090
```

---

## 常见问题

**Q: `nvcc: command not found`**
确认已激活 conda 环境并执行了第二步的环境变量配置：
```shell
which nvcc  # 路径应包含 conda 环境名
```

**Q: `setup.py develop` 报 `sm_89` 不支持**
说明 `CUDA_HOME` 指向了系统的旧版 CUDA，而非 conda 环境：
```shell
echo $CUDA_HOME   # 应包含 conda 路径，而非 /usr/local/cuda
nvcc --version    # 应为 11.8
```

**Q: mmcv-full 安装时找不到对应 wheel**
确认 torch 版本确实是 `1.13.1+cu117`，wheel 索引对版本敏感：
```shell
python -c "import torch; print(torch.__version__)"
```

**Q: RTX 4090 上 PyTorch 算子首次运行很慢**
正常现象。PyTorch cu117 不含 sm_89 预编译内核，首次运行会 JIT 编译并缓存，后续正常。
自定义 CUDA 扩展（spconv、bev_pool 等）已通过第六步原生编译为 sm_89，无此问题。
