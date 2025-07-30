#!/usr/bin/env python3
"""
GPT-SoVITS API v3 with Lazy Loading
基于inference_test.py的API服务，实现模型懒加载
"""

import os
import sys
import json
import logging
import traceback
import argparse
import signal
import numpy as np
import soundfile as sf
from fastapi import FastAPI, Response, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
from io import BytesIO
from typing import Optional, List
from pydantic import BaseModel
import threading
import time

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 添加项目路径
now_dir = os.getcwd()
sys.path.append(now_dir)
sys.path.append("%s/GPT_SoVITS" % (now_dir))

# 导入推理相关模块
from GPT_SoVITS.inference_musa_cli_simplified import (
    get_tts_wav,
    change_gpt_weights,
    change_sovits_weights,
    device,
    model_version,
    version,
    gpt_path,
    sovits_path,
    dict_language,
    cut1, cut2, cut3, cut4, cut5
)

# 创建FastAPI应用
APP = FastAPI(title="GPT-SoVITS API v3 Lazy", version="3.0.0")

# 全局变量
current_gpt_path = gpt_path
current_sovits_path = sovits_path
models_loaded = False
models_lock = threading.Lock()

# 文本切分方法映射
cut_methods = {
    "凑四句一切": cut1,
    "凑50字一切": cut2,
    "按中文句号。切": cut3,
    "按英文句号.切": cut4,
    "按标点符号切": cut5,
    "不切": lambda x: x
}

class TTSRequest(BaseModel):
    """TTS请求模型"""
    text: str
    text_lang: str = "中文"
    ref_audio_path: str = "./完成高难行动.wav"
    prompt_text: Optional[str] = "终有一天，我们可以点起火焰，燃尽一切腐朽。"
    prompt_lang: Optional[str] = "中文"
    how_to_cut: Optional[str] = "按标点符号切"
    top_k: Optional[int] = 20
    top_p: Optional[float] = 0.6
    temperature: Optional[float] = 0.6
    speed: Optional[float] = 1.0
    pause_second: Optional[float] = 0.3
    ref_free: Optional[bool] = False
    if_freeze: Optional[bool] = False
    sample_steps: Optional[int] = 8
    if_sr: Optional[bool] = False

class TTSResponse(BaseModel):
    """TTS响应模型"""
    status: str
    message: str
    audio_data: Optional[str] = None  # base64编码的音频数据

def load_models_if_needed():
    """懒加载模型"""
    global models_loaded, current_gpt_path, current_sovits_path
    
    with models_lock:
        if not models_loaded:
            logger.info("开始加载模型...")
            try:
                # 忽略第二个yield的异常（UI更新相关）
                try:
                    next(change_sovits_weights(current_sovits_path))
                except:
                    pass
                # 加载GPT模型
                change_gpt_weights(current_gpt_path)
                models_loaded = True
                logger.info("模型加载完成")
            except Exception as e:
                logger.error(f"模型加载失败: {e}")
                raise HTTPException(status_code=500, detail=f"模型加载失败: {str(e)}")

def ensure_models_loaded():
    """确保模型已加载"""
    if not models_loaded:
        load_models_if_needed()

@APP.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "version": "v3",
        "device": device,
        "model_version": model_version,
        "models_loaded": models_loaded,
        "current_gpt_path": current_gpt_path,
        "current_sovits_path": current_sovits_path
    }

@APP.get("/set_gpt_weights")
async def set_gpt_weights(weights_path: str):
    """切换GPT模型"""
    global current_gpt_path, models_loaded
    
    try:
        current_gpt_path = weights_path
        models_loaded = False  # 标记需要重新加载
        logger.info(f"GPT模型路径已更新为: {weights_path}")
        return {"status": "success", "message": f"GPT模型路径已更新为: {weights_path}"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"设置GPT模型失败: {str(e)}")

@APP.get("/set_sovits_weights")
async def set_sovits_weights(weights_path: str):
    """切换SoVITS模型"""
    global current_sovits_path, models_loaded
    
    try:
        current_sovits_path = weights_path
        models_loaded = False  # 标记需要重新加载
        logger.info(f"SoVITS模型路径已更新为: {weights_path}")
        return {"status": "success", "message": f"SoVITS模型路径已更新为: {weights_path}"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"设置SoVITS模型失败: {str(e)}")

@APP.post("/tts")
async def tts_synthesis(request: TTSRequest):
    """TTS合成接口"""
    try:
        # 确保模型已加载
        ensure_models_loaded()
        
        # 验证输入
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="文本不能为空")
        
        if not os.path.exists(request.ref_audio_path):
            raise HTTPException(status_code=400, detail="参考音频文件不存在")
        
        # 验证语言
        if request.text_lang not in dict_language:
            raise HTTPException(status_code=400, detail=f"不支持的语言: {request.text_lang}")
        
        if request.prompt_lang not in dict_language:
            raise HTTPException(status_code=400, detail=f"不支持的提示语言: {request.prompt_lang}")
        
        logger.info(f"开始TTS合成: {request.text[:50]}...")
        
        # 执行TTS合成
        result_generator = get_tts_wav(
            ref_wav_path=request.ref_audio_path,
            prompt_text=request.prompt_text,
            prompt_language=request.prompt_lang,
            text=request.text,
            text_language=request.text_lang,
            how_to_cut=request.how_to_cut,
            top_k=request.top_k,
            top_p=request.top_p,
            temperature=request.temperature,
            ref_free=request.ref_free,
            speed=request.speed,
            if_freeze=request.if_freeze,
            inp_refs=None,
            sample_steps=request.sample_steps,
            if_sr=request.if_sr,
            pause_second=request.pause_second
        )
        
        # 获取最终结果
        sr, audio = None, None
        for sr, audio in result_generator:
            pass
        
        if audio is None:
            raise HTTPException(status_code=500, detail="音频生成失败")
        
        # 将音频数据转换为base64
        import base64
        audio_bytes = BytesIO()
        sf.write(audio_bytes, audio, sr, format='WAV')
        audio_bytes.seek(0)
        audio_base64 = base64.b64encode(audio_bytes.read()).decode('utf-8')
        
        logger.info(f"TTS合成完成，音频长度: {len(audio)/sr:.2f}秒")
        
        return TTSResponse(
            status="success",
            message="TTS合成成功",
            audio_data=audio_base64
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"TTS合成失败: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"TTS合成失败: {str(e)}")

@APP.get("/control")
async def control(command: str):
    """命令控制"""
    if command == "restart":
        # 重新加载模型
        global models_loaded
        models_loaded = False
        load_models_if_needed()
        return {"status": "success", "message": "模型已重新加载"}
    elif command == "status":
        return {
            "status": "success",
            "models_loaded": models_loaded,
            "device": device,
            "model_version": model_version
        }
    else:
        raise HTTPException(status_code=400, detail=f"未知命令: {command}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="GPT-SoVITS API v3 with Lazy Loading")
    parser.add_argument("-a", "--bind_addr", type=str, default="0.0.0.0", help="绑定地址，默认: 0.0.0.0")
    parser.add_argument("-p", "--port", type=int, default=9880, help="绑定端口，默认: 9880")
    parser.add_argument("--lazy_load", action="store_true", help="启用懒加载模式")
    
    args = parser.parse_args()
    
    if not args.lazy_load:
        # 预加载模型
        logger.info("预加载模型...")
        load_models_if_needed()
    
    logger.info(f"启动API服务: {args.bind_addr}:{args.port}")
    logger.info(f"懒加载模式: {'启用' if args.lazy_load else '禁用'}")
    
    uvicorn.run(
        APP,
        host=args.bind_addr,
        port=args.port,
        log_level="info"
    )

if __name__ == "__main__":
    main() 