#!/usr/bin/env python3
"""
测试SV模型的MUSA FFT修复
"""

import os
import sys
import torch
import torch_musa
import numpy as np

def test_musa_sv_fix():
    """测试SV模型的MUSA FFT修复"""
    print("=== SV模型MUSA FFT修复测试 ===")
    
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
    
    # 测试SV模型导入
    print("\n=== SV模型导入测试 ===")
    try:
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        sys.path.append("GPT_SoVITS")
        
        # 检查SV模型文件是否存在
        sv_path = "GPT_SoVITS/pretrained_models/sv/pretrained_eres2netv2w24s4ep4.ckpt"
        if os.path.exists(sv_path):
            print(f"SV模型文件存在: {sv_path}")
        else:
            print(f"SV模型文件不存在: {sv_path}")
            return
        
        # 测试SV模型导入
        from GPT_SoVITS.sv import SV
        print("SV模型导入成功")
        
        # 测试SV模型初始化
        try:
            sv_model = SV(device, is_half)
            print("SV模型初始化成功")
            
            # 测试音频数据
            print("\n=== 音频数据处理测试 ===")
            # 创建模拟音频数据
            audio_data = torch.randn(1, 16000)  # 1秒的音频数据
            if device != "cpu":
                audio_data = audio_data.to(device)
            
            print(f"音频数据形状: {audio_data.shape}")
            print(f"音频数据设备: {audio_data.device}")
            print(f"音频数据类型: {audio_data.dtype}")
            
            # 测试compute_embedding3函数
            print("\n=== compute_embedding3测试 ===")
            try:
                sv_emb = sv_model.compute_embedding3(audio_data)
                print("SV嵌入计算成功")
                print(f"嵌入形状: {sv_emb.shape}")
                print(f"嵌入设备: {sv_emb.device}")
                print(f"嵌入类型: {sv_emb.dtype}")
            except Exception as e:
                print(f"SV嵌入计算失败: {e}")
                import traceback
                traceback.print_exc()
                
        except Exception as e:
            print(f"SV模型初始化失败: {e}")
            import traceback
            traceback.print_exc()
            
    except Exception as e:
        print(f"SV模型导入失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 测试FFT操作
    print("\n=== FFT操作测试 ===")
    try:
        # 创建测试数据
        test_data = torch.randn(1024)
        if device != "cpu":
            test_data = test_data.to(device)
        
        # 测试FFT操作
        if torch_musa.is_available():
            print("MUSA设备：测试FFT回退到CPU")
            # 在MUSA设备上，FFT应该回退到CPU
            test_data_cpu = test_data.cpu()
            fft_result = torch.fft.rfft(test_data_cpu)
            fft_result = fft_result.to(device)
            print("FFT操作成功（CPU回退）")
        else:
            fft_result = torch.fft.rfft(test_data)
            print("FFT操作成功")
            
    except Exception as e:
        print(f"FFT操作失败: {e}")

def test_inference_with_sv():
    """测试推理过程中的SV模型使用"""
    print("\n=== 推理SV模型测试 ===")
    try:
        # 模拟推理过程中的SV模型使用
        if torch_musa.is_available():
            print("测试MUSA设备上的SV模型推理")
            
            # 模拟音频数据
            wav = torch.randn(1, 16000)
            if torch_musa.is_available():
                wav = wav.to("musa:0")
            
            # 模拟SV模型处理
            if torch_musa.is_available():
                # 将数据移到CPU进行FFT计算
                wav_cpu = wav.cpu()
                print("数据已移到CPU进行FFT计算")
                # 这里应该调用Kaldi.fbank，但我们只是模拟
                feat = torch.randn(1, 80, 100)  # 模拟特征
                # 将结果移回MUSA设备
                feat = feat.to("musa:0")
                print("特征已移回MUSA设备")
                print("SV模型推理模拟成功")
            
    except Exception as e:
        print(f"推理SV模型测试失败: {e}")

if __name__ == "__main__":
    test_musa_sv_fix()
    test_inference_with_sv()
    print("\n=== 测试完成 ===") 