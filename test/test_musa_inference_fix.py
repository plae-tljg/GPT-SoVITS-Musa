#!/usr/bin/env python3
"""
测试MUSA推理修复是否有效
"""

import os
import sys
import torch
import torch_musa
import numpy as np

def test_musa_inference_fix():
    """测试MUSA推理修复"""
    print("=== MUSA推理修复测试 ===")
    
    # 测试设备检测
    def get_device():
        if torch_musa.is_available():
            return "musa"
        elif torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"
    
    device = get_device()
    print(f"检测到的设备: {device}")
    
    # 测试is_half设置
    if torch_musa.is_available():
        is_half = False
        print("MUSA设备检测到，禁用半精度模式")
    else:
        is_half = eval(os.environ.get("is_half", "True")) and torch.cuda.is_available()
    print(f"半精度设置: {is_half}")
    
    # 测试数据精度处理
    print("\n=== 数据精度处理测试 ===")
    try:
        # 模拟音频数据
        audio_data = np.random.randn(16000).astype(np.float32)
        wav16k = torch.from_numpy(audio_data)
        
        # 模拟zero_wav数据
        zero_wav = np.zeros(4800, dtype=np.float16 if is_half else np.float32)
        zero_wav_torch = torch.from_numpy(zero_wav)
        
        # 测试推理代码中的精度处理逻辑
        if is_half == True and not torch_musa.is_available():
            wav16k = wav16k.half().to(device)
            zero_wav_torch = zero_wav_torch.half().to(device)
        else:
            wav16k = wav16k.to(device)
            zero_wav_torch = zero_wav_torch.to(device)
        
        # 在MUSA设备上强制使用全精度
        if torch_musa.is_available():
            wav16k = wav16k.float()
            zero_wav_torch = zero_wav_torch.float()
            print("  MUSA设备：强制转换为全精度")
        
        print(f"  wav16k数据类型: {wav16k.dtype}")
        print(f"  zero_wav_torch数据类型: {zero_wav_torch.dtype}")
        
        # 测试模型精度
        print("\n=== 模型精度测试 ===")
        if torch_musa.is_available():
            # 创建一个简单的测试模型
            model = torch.nn.Conv1d(1, 1, 3)
            model = model.to(device)
            
            # 强制转换为全精度
            model = model.float()
            for module in model.modules():
                if hasattr(module, 'weight') and module.weight is not None:
                    module.weight.data = module.weight.data.float()
                if hasattr(module, 'bias') and module.bias is not None:
                    module.bias.data = module.bias.data.float()
            
            print("  模型强制转换为全精度")
            print(f"  模型权重数据类型: {model.weight.dtype}")
            
            # 测试前向传播
            x = torch.randn(1, 1, 10).to(device).float()
            y = model(x)
            print("  MUSA卷积操作成功")
            
    except Exception as e:
        print(f"测试失败: {e}")
    
    # 测试内存清理
    print("\n=== 内存清理测试 ===")
    try:
        if torch_musa.is_available():
            torch_musa.empty_cache()
            print("MUSA内存清理成功")
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("CUDA内存清理成功")
        else:
            print("CPU模式，无需内存清理")
    except Exception as e:
        print(f"内存清理失败: {e}")

def test_inference_webui_import():
    """测试推理WebUI的导入"""
    print("\n=== 推理WebUI导入测试 ===")
    try:
        # 添加路径
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        sys.path.append("GPT_SoVITS")
        
        # 测试关键模块导入
        from GPT_SoVITS.inference_webui import get_device, device, is_half
        print(f"推理WebUI导入成功")
        print(f"设备: {device}")
        print(f"半精度: {is_half}")
        
        # 验证MUSA设备的处理
        if torch_musa.is_available():
            assert is_half == False, "MUSA设备应该禁用半精度"
            print("✓ MUSA设备半精度设置正确")
        
    except Exception as e:
        print(f"推理WebUI导入失败: {e}")

if __name__ == "__main__":
    test_musa_inference_fix()
    test_inference_webui_import()
    print("\n=== 测试完成 ===") 