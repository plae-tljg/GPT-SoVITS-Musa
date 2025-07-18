#!/usr/bin/env python3
"""
测试推理代码的MUSA GPU支持
"""

import os
import sys
import torch
import torch_musa

def test_musa_inference_setup():
    """测试推理代码的MUSA设置"""
    print("=== 推理代码MUSA支持测试 ===")
    
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
    
    # 测试模型加载到设备
    print("\n=== 模型加载测试 ===")
    try:
        # 创建一个简单的测试模型
        model = torch.nn.Linear(10, 1)
        
        if device == "musa":
            model = model.to("musa:0")
            if is_half:
                model = model.half()
            print("模型成功加载到MUSA设备")
            
            # 测试前向传播
            x = torch.randn(5, 10).to("musa:0")
            if is_half:
                x = x.half()
            y = model(x)
            print("MUSA前向传播成功")
            
        elif device == "cuda":
            model = model.to("cuda:0")
            if is_half:
                model = model.half()
            print("模型成功加载到CUDA设备")
            
            # 测试前向传播
            x = torch.randn(5, 10).to("cuda:0")
            if is_half:
                x = x.half()
            y = model(x)
            print("CUDA前向传播成功")
            
        else:
            model = model.to("cpu")
            print("模型成功加载到CPU设备")
            
            # 测试前向传播
            x = torch.randn(5, 10)
            y = model(x)
            print("CPU前向传播成功")
            
    except Exception as e:
        print(f"模型加载失败: {e}")
    
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
    
    # 测试随机种子设置
    print("\n=== 随机种子测试 ===")
    try:
        import random
        import numpy as np
        
        seed = 42
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        if torch_musa.is_available():
            torch_musa.manual_seed(seed)
            print("MUSA随机种子设置成功")
        elif torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            print("CUDA随机种子设置成功")
        else:
            print("CPU随机种子设置成功")
            
    except Exception as e:
        print(f"随机种子设置失败: {e}")

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
        
    except Exception as e:
        print(f"推理WebUI导入失败: {e}")

if __name__ == "__main__":
    test_musa_inference_setup()
    test_inference_webui_import()
    print("\n=== 测试完成 ===") 