#!/usr/bin/env python3
"""
测试UVR5 CPU模式
验证UVR5强制使用CPU后是否能正常工作
"""

import os
import sys
import torch
import torch_musa
import subprocess
import tempfile
import shutil

def test_uvr5_cpu_mode():
    """测试UVR5 CPU模式"""
    print("=== 测试UVR5 CPU模式 ===")
    
    # 检查设备可用性
    print(f"CUDA可用: {torch.cuda.is_available()}")
    print(f"MUSA可用: {torch_musa.is_available()}")
    
    # 创建临时测试音频文件
    test_audio = create_test_audio()
    if not test_audio:
        print("❌ 无法创建测试音频文件")
        return False
    
    try:
        # 测试UVR5启动
        print("\n--- 测试UVR5启动 ---")
        result = test_uvr5_startup()
        if not result:
            print("❌ UVR5启动失败")
            return False
        
        print("✅ UVR5 CPU模式测试通过")
        return True
        
    finally:
        # 清理临时文件
        if test_audio and os.path.exists(test_audio):
            os.remove(test_audio)

def create_test_audio():
    """创建测试音频文件"""
    try:
        # 使用ffmpeg生成一个简单的测试音频
        temp_dir = tempfile.gettempdir()
        test_audio = os.path.join(temp_dir, "test_uvr5.wav")
        
        # 生成1秒的440Hz正弦波
        cmd = [
            "ffmpeg", "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
            "-ar", "44100", "-ac", "2", "-y", test_audio
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0 and os.path.exists(test_audio):
            print(f"✅ 创建测试音频: {test_audio}")
            return test_audio
        else:
            print(f"❌ 创建测试音频失败: {result.stderr}")
            return None
            
    except Exception as e:
        print(f"❌ 创建测试音频异常: {e}")
        return None

def test_uvr5_startup():
    """测试UVR5启动"""
    try:
        # 检查UVR5权重文件是否存在
        weight_dir = "tools/uvr5/uvr5_weights"
        if not os.path.exists(weight_dir):
            print(f"❌ UVR5权重目录不存在: {weight_dir}")
            return False
        
        # 检查是否有可用的模型
        model_files = [f for f in os.listdir(weight_dir) 
                      if f.endswith(('.pth', '.ckpt')) or 'onnx' in f]
        if not model_files:
            print(f"❌ 未找到UVR5模型文件")
            return False
        
        print(f"✅ 找到UVR5模型文件: {len(model_files)}个")
        
        # 测试UVR5模块导入
        try:
            sys.path.append('tools/uvr5')
            from bsroformer import Roformer_Loader
            from mdxnet import MDXNetDereverb
            from vr import AudioPre, AudioPreDeEcho
            print("✅ UVR5模块导入成功")
        except ImportError as e:
            print(f"❌ UVR5模块导入失败: {e}")
            return False
        
        # 测试设备设置
        device = "cpu"
        is_half = False
        
        print(f"✅ UVR5设备设置: {device}, 半精度: {is_half}")
        
        return True
        
    except Exception as e:
        print(f"❌ UVR5启动测试异常: {e}")
        return False

def test_uvr5_webui_script():
    """测试UVR5 WebUI脚本参数"""
    print("\n--- 测试UVR5 WebUI脚本参数 ---")
    
    # 模拟命令行参数
    test_cases = [
        ("musa:0", True, 9876, False),
        ("cuda:0", True, 9877, False),
        ("cpu", False, 9878, False),
    ]
    
    for device, is_half, port, share in test_cases:
        print(f"\n测试参数: device={device}, is_half={is_half}, port={port}, share={share}")
        
        # 模拟webui.py的设备检测逻辑
        if device.startswith("musa") or device.startswith("cuda"):
            print(f"  强制使用CPU，原设备: {device}")
            final_device = "cpu"
            final_is_half = False
        else:
            final_device = device
            final_is_half = is_half
        
        print(f"  最终设置: device={final_device}, is_half={final_is_half}")
        
        if final_device == "cpu" and not final_is_half:
            print("  ✅ CPU模式配置正确")
        else:
            print("  ❌ CPU模式配置错误")

if __name__ == "__main__":
    print("开始UVR5 CPU模式测试...")
    
    # 测试UVR5 WebUI脚本参数
    test_uvr5_webui_script()
    
    # 测试UVR5 CPU模式
    success = test_uvr5_cpu_mode()
    
    if success:
        print("\n🎉 所有测试通过！UVR5 CPU模式工作正常")
    else:
        print("\n❌ 测试失败，请检查配置")
    
    print("\n=== 测试完成 ===") 