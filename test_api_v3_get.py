#!/usr/bin/env python3
"""
GPT-SoVITS API v3 使用示例
"""

import requests
import base64
import json
import soundfile as sf
from io import BytesIO

# API配置
API_BASE_URL = "http://localhost:9880"

def test_health():
    """测试健康检查"""
    print("=== 健康检查 ===")
    response = requests.get(f"{API_BASE_URL}/health")
    print(f"状态码: {response.status_code}")
    print(f"响应: {response.json()}")
    print()

def test_tts():
    """测试TTS合成"""
    print("=== TTS合成测试 ===")
    
    # 准备请求数据
    tts_request = {
        "text": "每个都有过人之处，每个都有他们独门绝招，斗志和耐性更是技惊四座，秘密武器更给你意外的惊喜呀！",
        "text_lang": "中文",
        "ref_audio_path": "./完成高难行动.wav",  # 请确保这个文件存在
        "prompt_text": "终有一天，我们可以点起火焰，燃尽一切腐朽。",
        "prompt_lang": "中文",
        "how_to_cut": "按标点符号切",
        "top_k": 20,
        "top_p": 0.6,
        "temperature": 0.6,
        "speed": 1.0,
        "pause_second": 0.3,
        "ref_free": False,
        "if_freeze": False,
        "sample_steps": 8,
        "if_sr": False
    }
    
    try:
        response = requests.post(f"{API_BASE_URL}/tts", json=tts_request)
        print(f"状态码: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"状态: {result['status']}")
            print(f"消息: {result['message']}")
            
            # 保存音频文件
            if result.get('audio_data'):
                audio_bytes = base64.b64decode(result['audio_data'])
                audio_io = BytesIO(audio_bytes)
                audio, sr = sf.read(audio_io)
                
                output_path = "api_output.wav"
                sf.write(output_path, audio, sr)
                print(f"音频已保存到: {output_path}")
                print(f"音频长度: {len(audio)/sr:.2f}秒")
        else:
            print(f"错误: {response.text}")
            
    except Exception as e:
        print(f"请求失败: {e}")
    
    print()

def test_model_switch():
    """测试模型切换"""
    print("=== 模型切换测试 ===")
    
    # 切换GPT模型
    gpt_path = "GPT_weights_v2Pro/theresa-e15.ckpt"
    response = requests.get(f"{API_BASE_URL}/set_gpt_weights?weights_path={gpt_path}")
    print(f"GPT模型切换: {response.json()}")
    
    # 切换SoVITS模型
    sovits_path = "SoVITS_weights_v2Pro/theresa_e8_s160.pth"
    response = requests.get(f"{API_BASE_URL}/set_sovits_weights?weights_path={sovits_path}")
    print(f"SoVITS模型切换: {response.json()}")
    
    print()

def test_control():
    """测试控制命令"""
    print("=== 控制命令测试 ===")
    
    # 获取状态
    response = requests.get(f"{API_BASE_URL}/control?command=status")
    print(f"状态: {response.json()}")
    
    # 重新加载模型
    response = requests.get(f"{API_BASE_URL}/control?command=restart")
    print(f"重新加载: {response.json()}")
    
    print()

def test_tts_simple(text):
    """测试TTS合成"""
    print("=== TTS合成测试 ===")
    
    # 准备请求数据
    tts_request = {
        "text": text
    }
    
    try:
        response = requests.post(f"{API_BASE_URL}/tts", json=tts_request)
        print(f"状态码: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"状态: {result['status']}")
            print(f"消息: {result['message']}")
            
            # 保存音频文件
            if result.get('audio_data'):
                audio_bytes = base64.b64decode(result['audio_data'])
                audio_io = BytesIO(audio_bytes)
                audio, sr = sf.read(audio_io)
                
                output_path = "api_output.wav"
                sf.write(output_path, audio, sr)
                print(f"音频已保存到: {output_path}")
                print(f"音频长度: {len(audio)/sr:.2f}秒")
        else:
            print(f"错误: {response.text}")
            
    except Exception as e:
        print(f"请求失败: {e}")
    
    print()

def main():
    """主函数"""
    print("GPT-SoVITS API v3 使用示例")
    print("=" * 50)
    
    # 测试健康检查
    test_health()
    
    # 测试模型切换
    test_model_switch()
    
    # 测试控制命令
    test_control()
    
    # 测试TTS合成
    test_tts()
    
    print("测试完成！")

if __name__ == "__main__":
    # main() 
    test_tts_simple("每个都有过人之处，每个都有他们独门绝招，斗志和耐性更是技惊四座，秘密武器更给你意外的惊喜呀！")