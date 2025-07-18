#!/usr/bin/env python3
"""
测试MUSA GPU训练支持
"""

import os
import sys
import torch
import torch_musa

def test_musa_training_setup():
    """测试MUSA训练设置"""
    print("=== MUSA训练设置测试 ===")
    
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
    
    # 测试设备数量
    if torch_musa.is_available():
        n_gpus = torch_musa.device_count()
        print(f"MUSA设备数量: {n_gpus}")
    elif torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        print(f"CUDA设备数量: {n_gpus}")
    else:
        n_gpus = 1
        print(f"CPU设备数量: {n_gpus}")
    
    # 测试模型加载到设备
    print("\n=== 模型加载测试 ===")
    try:
        # 创建一个简单的测试模型
        model = torch.nn.Linear(10, 1)
        
        if device == "musa":
            model = model.to("musa:0")
            print("模型成功加载到MUSA设备")
            
            # 测试前向传播
            x = torch.randn(5, 10).to("musa:0")
            y = model(x)
            print("MUSA前向传播成功")
            
        elif device == "cuda":
            model = model.to("cuda:0")
            print("模型成功加载到CUDA设备")
            
            # 测试前向传播
            x = torch.randn(5, 10).to("cuda:0")
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
    
    # 测试autocast
    print("\n=== Autocast测试 ===")
    try:
        if device == "musa":
            autocast_device = "musa"
        elif device == "cuda":
            autocast_device = "cuda"
        else:
            autocast_device = "cpu"
        
        with torch.amp.autocast(device_type=autocast_device, enabled=True):
            x = torch.randn(5, 10)
            if device != "cpu":
                x = x.to(f"{device}:0")
            y = torch.nn.functional.linear(x, torch.randn(1, 10).to(x.device))
            print(f"{device.upper()} autocast成功")
            
    except Exception as e:
        print(f"Autocast失败: {e}")

def test_distributed_setup():
    """测试分布式训练设置"""
    print("\n=== 分布式训练设置测试 ===")
    
    try:
        import torch.distributed as dist
        
        # 测试后端选择
        if torch_musa.is_available():
            backend = "gloo"  # MUSA暂时使用gloo后端
        elif torch.cuda.is_available():
            backend = "nccl"
        else:
            backend = "gloo"
        
        print(f"选择的分布式后端: {backend}")
        
        # 注意：这里只是测试配置，不实际初始化分布式训练
        print("分布式训练配置测试完成")
        
    except Exception as e:
        print(f"分布式训练设置失败: {e}")

if __name__ == "__main__":
    print("MUSA GPU训练支持测试")
    print("=" * 50)
    
    test_musa_training_setup()
    test_distributed_setup()
    
    print("\n" + "=" * 50)
    print("测试完成") 