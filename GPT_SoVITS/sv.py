import sys
import os
import torch
import torch_musa  # 添加MUSA支持

sys.path.append(f"{os.getcwd()}/GPT_SoVITS/eres2net")
sv_path = "GPT_SoVITS/pretrained_models/sv/pretrained_eres2netv2w24s4ep4.ckpt"
from ERes2NetV2 import ERes2NetV2
import kaldi as Kaldi


class SV:
    def __init__(self, device, is_half):
        pretrained_state = torch.load(sv_path, map_location="cpu", weights_only=False)
        embedding_model = ERes2NetV2(baseWidth=24, scale=4, expansion=4)
        embedding_model.load_state_dict(pretrained_state)
        embedding_model.eval()
        self.embedding_model = embedding_model
        if is_half == False:
            self.embedding_model = self.embedding_model.to(device)
        else:
            self.embedding_model = self.embedding_model.half().to(device)
        self.is_half = is_half
        self.device = device

    def compute_embedding3(self, wav):
        with torch.no_grad():
            if self.is_half == True:
                wav = wav.half()
            
            # 在MUSA设备上，FFT操作需要在CPU上进行
            if torch_musa.is_available():
                # 将数据移到CPU进行FFT计算
                wav_cpu = wav.cpu()
                feat = torch.stack(
                    [Kaldi.fbank(wav0.unsqueeze(0), num_mel_bins=80, sample_frequency=16000, dither=0) for wav0 in wav_cpu]
                )
                # 将结果移回原设备
                feat = feat.to(self.device)
            else:
                feat = torch.stack(
                    [Kaldi.fbank(wav0.unsqueeze(0), num_mel_bins=80, sample_frequency=16000, dither=0) for wav0 in wav]
                )
            
            sv_emb = self.embedding_model.forward3(feat)
        return sv_emb
