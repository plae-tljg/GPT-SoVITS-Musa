# UVR5 MUSA GPU 支持

Note this use the musa-4.0.0 SDK and the latest  torch_musa Release v2.0.0 built package.  

Note that some part which I encoutner problem converting into running by gpu (code doesnt have problem but just stuck somewhere withotu warning or error).  

**check out the conda_create_env.txt for required conda environment setup. I use anaconda because my computer ffmpeg has trouble running the funasr-related things, so need ffmpeg inside conda.**  

**Extra Note: please note the code patches zip file which contains the code to replace for pytorch lightning and related things in your virtual environment.**  

本文档说明如何在GPT-SoVITS的UVR5模块中使用MUSA GPU进行人声分离和音频处理。

## 概述

UVR5 (Ultimate Vocal Remover 5) 是一个强大的音频分离工具，可以分离人声和伴奏。现在支持使用MUSA GPU来加速处理过程。

## 支持的设备类型

- **MUSA GPU**: 支持MUSA架构的GPU设备
- **CUDA GPU**: 支持NVIDIA CUDA的GPU设备  
- **CPU**: 作为后备选项

## 安装要求

### 1. 安装torch_musa

确保已安装支持MUSA的PyTorch版本：

```bash
# 安装torch_musa
pip install torch_musa
```

### 2. 验证安装

运行测试脚本验证MUSA支持：

```bash
python test_musa_uvr5.py
```

## 使用方法

### 1. 通过主WebUI启动

1. 启动主WebUI：
   ```bash
   python webui.py
   ```

2. 在WebUI界面中点击"人声分离WebUI"按钮启动UVR5

3. 系统会自动检测可用的GPU设备：
   - 优先使用MUSA GPU（如果可用）
   - 其次使用CUDA GPU（如果可用）
   - 最后回退到CPU

### 2. 直接启动UVR5 WebUI

```bash
# 使用MUSA GPU
python tools/uvr5/webui.py "musa:0" True 9873 False

# 使用CUDA GPU
python tools/uvr5/webui.py "cuda:0" True 9873 False

# 使用CPU
python tools/uvr5/webui.py "cpu" False 9873 False
```

参数说明：
- 第1个参数：设备类型 (`musa:0`, `cuda:0`, `cpu`)
- 第2个参数：是否使用半精度 (`True`/`False`)
- 第3个参数：WebUI端口号
- 第4个参数：是否分享 (`True`/`False`)

## 设备检测逻辑

系统会按以下优先级检测设备：

1. **MUSA GPU**: 如果检测到MUSA设备，优先使用
2. **CUDA GPU**: 如果没有MUSA设备，使用CUDA设备
3. **CPU**: 如果GPU都不可用，使用CPU

### 设备检测示例

```python
# 在config.py中的设备检测逻辑
if torch_musa.is_available():
    # 使用MUSA设备
    device = torch.device("musa:0")
elif torch.cuda.is_available():
    # 使用CUDA设备
    device = torch.device("cuda:0")
else:
    # 使用CPU
    device = torch.device("cpu")
```

## 性能优化

### 1. 半精度支持

MUSA GPU支持半精度（float16）计算，可以显著提升性能：

```python
# 启用半精度
is_half = True
model = model.half().to(device)
```

### 2. 内存管理

处理大文件时注意内存使用：

```python
# 清理GPU内存
if torch_musa.is_available():
    torch_musa.empty_cache()
elif torch.cuda.is_available():
    torch.cuda.empty_cache()
```

## 故障排除

### 1. MUSA设备不可用

如果遇到MUSA设备不可用的错误：

1. 检查torch_musa是否正确安装：
   ```bash
   python -c "import torch_musa; print(torch_musa.is_available())"
   ```

2. 检查MUSA驱动是否正确安装

3. 系统会自动回退到CPU模式

### 2. 内存不足

如果遇到内存不足错误：

1. 减少批处理大小
2. 使用CPU模式处理
3. 清理GPU内存

### 3. 模型加载失败

如果模型加载失败：

1. 检查模型文件是否存在
2. 确保模型文件完整
3. 尝试使用CPU模式加载

## 支持的模型类型

UVR5支持以下模型类型：

1. **BS-Roformer模型**: 基于Transformer的分离模型
2. **Mel-Band-Roformer模型**: 基于梅尔频谱的分离模型
3. **VR模型**: 传统的分离模型
4. **MDX-Net模型**: 去混响模型

## 注意事项

1. **兼容性**: 确保使用的模型与设备兼容
2. **内存**: MUSA GPU内存使用情况需要监控
3. **性能**: 不同设备类型的性能差异较大
4. **稳定性**: 建议在处理重要文件前先进行小规模测试

## 更新日志

- **v1.0**: 初始MUSA GPU支持
- 添加了MUSA设备检测
- 支持MUSA半精度计算
- 添加了设备回退机制

## 技术支持

如果遇到问题，请：

1. 运行测试脚本检查设备状态
2. 查看日志文件获取详细错误信息
3. 在GitHub上提交issue

---

**注意**: 使用MUSA GPU需要确保硬件和软件环境都支持MUSA架构。 
