# 成功解析测试结果
# Best checkpoint: models/fakesv/weights/best_model_sv_0.8616.pth

# 测试结果:
# Test Accuracy: 0.8616
# Weighted Precision: 0.8651
# Weighted Recall: 0.8616
# Weighted F1: 0.8600
# Fake - Precision: 0.8398, Recall: 0.9309, F1: 0.8830
# Real - Precision: 0.8976, Recall: 0.7731, F1: 0.8307

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import os
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, accuracy_score

import json
import pickle
import h5py
import csv
import subprocess
import sys
from torch.utils.data import Dataset, DataLoader


# ======================================
class MultiModalNewsDataset(Dataset):
    def __init__(self,
                 split_txt: str,
                 json_file: str,
                 feature_info: dict):
        """
        split_txt:   每行一个 video_id 的 txt 文件
        json_file:   完整的 JSONL，带 annotation
        feature_info: dict, 每个模态配置项
        """
        # --- 1. 读 split 列表 & label dict ---
        with open(split_txt, 'r', encoding='utf-8') as f:
            vids = [l.strip() for l in f if l.strip()]
        self.label_dict = self._load_labels(json_file)
        # 只保留那些既在 split 又有标签的 id
        self.video_ids = [v for v in vids if v in self.label_dict]

        self.feature_info = feature_info
        self.modalities = list(feature_info.keys())

        # --- 2. 预加载所有特征到内存（numpy）---
        # 数据结构: self.features[modality] = { video_id: np.ndarray }
        self.features = {}
        for mod, info in feature_info.items():
            t = info['type']
            if t == 'pkl':
                self.features[mod] = self._load_pkl(info['path'])
            elif t == 'h5':
                self.features[mod] = self._load_h5(
                    info['path'],
                    info['feat_name'],
                    info['id_name']
                )
            elif t == 'hdf5':
                self.features[mod] = self._load_hdf5(
                    info['path'],
                    info['feat_name']
                )
            else:
                raise ValueError(f"Unknown feature type {t} for modality {mod}")

    def _load_labels(self, json_file):
        label_map = {"假": 0, "真": 1, "辟谣": 2}
        d = {}
        with open(json_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    vid, ann = obj.get("video_id"), obj.get("annotation")
                    if vid and ann in label_map:
                        d[vid] = label_map[ann]
                except:
                    pass
        return d

    def _load_pkl(self, path):
        """加载.pkl文件，确保返回dict格式{video_id: feature}"""
        print(f"加载PKL文件: {path}")
        raw = pickle.load(open(path, 'rb'))
        
        result_dict = {}
        
        # 处理音频特征的特殊格式 (VGGish)
        if path.endswith('audio_vggish.pkl'):
            print("检测到audio_vggish.pkl特殊格式")
            for item in raw:
                if isinstance(item, dict) and 'video_id' in item and 'features' in item:
                    vid = item['video_id']
                    feat = item['features']
                    result_dict[vid] = np.array(feat, dtype=np.float32)
            return result_dict
        
        # 处理标准(id, feature)格式
        if isinstance(raw, list) and len(raw) > 0 and isinstance(raw[0], tuple) and len(raw[0]) == 2:
            print(f"标准(id, feature)格式，共{len(raw)}条")
            return {item[0]: np.array(item[1], dtype=np.float32) for item in raw}
        
        # 如果是字典格式
        if isinstance(raw, dict):
            print(f"字典格式，键数量: {len(raw.keys())}")
            return {k: np.array(v, dtype=np.float32) if not isinstance(v, np.ndarray) else v 
                    for k, v in raw.items()}
        
        raise ValueError(f"不支持的PKL格式: {type(raw)}")

    def _load_h5(self, path, feat_name, id_name):
        """加载.h5文件，返回dict格式{video_id: feature}"""
        print(f"加载H5文件: {path}")
        d = {}
        with h5py.File(path, 'r') as f:
            if id_name in f and feat_name in f:
                ids = f[id_name][()]
                feats = f[feat_name][()]
                print(f"IDs形状: {ids.shape}, 特征形状: {feats.shape}")
                
                for i, vid in enumerate(ids):
                    if isinstance(vid, bytes):
                        vid = vid.decode('utf-8')
                    d[vid] = feats[i].astype(np.float32)
        
        print(f"从H5加载了{len(d)}条特征")
        return d

    def _load_hdf5(self, path, feat_name):
        """加载.hdf5文件，返回dict格式{video_id: feature}"""
        print(f"加载HDF5文件: {path}")
        d = {}
        with h5py.File(path, 'r') as f:
            # 每个video_id是一个组
            video_ids = [k for k in f.keys() if isinstance(f[k], h5py.Group)]
            print(f"发现视频ID作为组, 数量: {len(video_ids)}")
            
            for vid in video_ids:
                if feat_name in f[vid]:
                    d[vid] = np.array(f[vid][feat_name][()], dtype=np.float32)
                elif 'c3d_features' in f[vid]:  # 尝试替代名称
                    d[vid] = np.array(f[vid]['c3d_features'][()], dtype=np.float32)
        
        print(f"从HDF5加载了{len(d)}条特征")
        return d

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, idx):
        vid = self.video_ids[idx]
        y = self.label_dict[vid]
        
        # 获取所有模态特征
        feats = {}
        for mod in self.modalities:
            if vid in self.features[mod]:
                feats[mod] = torch.from_numpy(self.features[mod][vid])
            else:
                print(f"警告: 视频ID {vid} 在模态 {mod} 中不存在")
                # 使用默认形状的零张量
                if mod.startswith('text_bert'):
                    shape = (1536,)
                elif mod.startswith('text_xclip'):
                    shape = (1024,)
                elif mod.startswith('audio_hubert'):
                    shape = (1024,)
                elif mod.startswith('audio_vggish'):
                    shape = (1, 128)
                elif mod.startswith('video_c3d'):
                    shape = (1, 4096)
                elif mod.startswith('video_xclip'):
                    shape = (8, 512)
                else:
                    shape = (1, 128)
                
                feats[mod] = torch.zeros(shape, dtype=torch.float32)
        
        return {'video_id': vid, 'modalities': feats, 'label': torch.tensor(y)}


# 自定义批处理函数，处理不同长度的序列
def custom_collate_fn(batch):
    """处理变长序列的自定义collate函数
    
    批次中每个样本的特征可能有不同的时序长度
    我们将返回一个包含所有特征和一个长度列表的字典
    """
    video_ids = [item['video_id'] for item in batch]
    labels = torch.stack([item['label'] for item in batch])
    
    # 处理每个模态
    modalities_dict = {}
    for mod in batch[0]['modalities'].keys():
        # 获取该模态下所有样本的特征
        mod_feats = [item['modalities'][mod] for item in batch]
        
        # 检查第一个特征是否是二维序列（时序）
        if mod_feats[0].dim() > 1:
            # 对于序列特征，我们要存储原始长度
            seq_lengths = [feat.size(0) for feat in mod_feats]
            
            # 对序列进行填充（使用零填充到最大长度）
            max_len = max(seq_lengths)
            feat_dim = mod_feats[0].size(-1)
            
            # 创建填充后的张量
            padded_feats = torch.zeros(len(mod_feats), max_len, feat_dim)
            for i, feat in enumerate(mod_feats):
                padded_feats[i, :feat.size(0)] = feat
            
            modalities_dict[mod] = {
                'features': padded_feats,
                'lengths': torch.tensor(seq_lengths)
            }
        else:
            # 对于非序列特征，直接堆叠
            modalities_dict[mod] = {
                'features': torch.stack(mod_feats),
                'lengths': None
            }
    
    return {
        'video_id': video_ids,
        'modalities': modalities_dict,
        'label': labels
    }


# 序列到固定维度转换模块
class SequenceToFixedDim(nn.Module):
    """
    使用动态路由将可变长度序列转换为固定维度特征
    输入: 变长序列特征 [B, seq_len, feat_dim] 或 [B, feat_dim]
    输出: 固定维度特征 [B, fixed_seq_len, fixed_dim]
    """
    def __init__(self, input_dim, fixed_seq_len=32, fixed_dim=128, routing_iterations=3):
        super(SequenceToFixedDim, self).__init__()
        self.input_dim = input_dim
        self.fixed_seq_len = fixed_seq_len
        self.fixed_dim = fixed_dim
        self.routing_iterations = routing_iterations
        
        # 维度投影层
        self.dim_proj = nn.Linear(input_dim, fixed_dim)
        
        # 主要capsule转换矩阵
        self.primary_transform = nn.Parameter(torch.randn(fixed_seq_len, fixed_dim, fixed_dim))
        nn.init.xavier_uniform_(self.primary_transform)
        
        # 时序注意力层，用于处理变长序列
        self.attention = nn.MultiheadAttention(fixed_dim, num_heads=8, batch_first=True)
        
        # 序列长度投影层，用于调整时序长度
        self.seq_proj = nn.Linear(fixed_seq_len, fixed_seq_len)
        
        # LayerNorm用于特征归一化
        self.layer_norm = nn.LayerNorm(fixed_dim)

    def forward(self, x, lengths=None):
        """
        x: 输入特征，可以是 [B, seq_len, feat_dim] 或 [B, feat_dim]
        lengths: 序列的实际长度（对于填充序列）
        返回: [B, fixed_seq_len, fixed_dim]
        """
        batch_size = x.size(0)
        
        # 处理非序列输入
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [B, 1, feat_dim]
        
        # 提取序列维度信息
        seq_len = x.size(1)
        feat_dim = x.size(2)
        
        # 如果输入维度与期望不匹配，使用线性变换
        if feat_dim != self.input_dim:
            proj = nn.Linear(feat_dim, self.input_dim).to(x.device)
            x = proj(x)
        
        # 将特征维度投影到fixed_dim
        x = self.dim_proj(x)  # [B, seq_len, fixed_dim]
        
        # 使用注意力机制处理变长序列，并创建注意力掩码
        mask = None
        if lengths is not None:
            # 创建注意力掩码
            mask = torch.zeros(batch_size, seq_len, device=x.device, dtype=torch.bool)
            for i, length in enumerate(lengths):
                mask[i, length:] = True
        
        # 使用自注意力进行序列编码
        x, _ = self.attention(x, x, x, key_padding_mask=mask)
        
        # 调整序列长度至fixed_seq_len
        if seq_len > self.fixed_seq_len:
            # 如果序列过长，使用自适应池化
            x = F.adaptive_avg_pool1d(x.transpose(1, 2), self.fixed_seq_len).transpose(1, 2)
        elif seq_len < self.fixed_seq_len:
            # 如果序列过短，使用插值扩展
            x = F.interpolate(x.transpose(1, 2), size=self.fixed_seq_len, mode='linear').transpose(1, 2)
        
        # 应用层归一化
        x = self.layer_norm(x)
        
        # 应用动态路由获得最终表示
        x = self._dynamic_routing(x)
        
        return x
    
    def _dynamic_routing(self, primary_caps):
        """实现动态路由算法"""
        batch_size = primary_caps.size(0)
        
        # 初始化路由系数
        routing_logits = torch.zeros(batch_size, self.fixed_seq_len, self.fixed_seq_len).to(primary_caps.device)
        
        # 动态路由迭代
        for r in range(self.routing_iterations):
            # 计算路由权重
            routing_weights = F.softmax(routing_logits, dim=2)
            
            # 更新表示
            if r < self.routing_iterations - 1:
                # 计算每个capsule的"投票"
                votes = torch.matmul(primary_caps.unsqueeze(2), 
                                      self.primary_transform.unsqueeze(0))
                votes = votes.squeeze(2)
                
                # 更新路由系数
                similarity = torch.bmm(primary_caps, votes.transpose(1, 2))
                routing_logits = routing_logits + similarity
        
        # 最终路由权重
        final_weights = F.softmax(routing_logits, dim=2)
        
        # 生成最终表示
        output = torch.bmm(final_weights, primary_caps)
        
        return output


class CoLaVID(nn.Module):
    """
    改进版CoLaVID模型：保留时序信息，并统一特征维度为(batch,32,128)
    
    输入特征:
      - text_bert: 文本BERT特征, 张量形状 [B, 1536]
      - text_xclip: 文本XCLIP特征, 张量形状 [B, 1024]
      - audio_hubert: 音频HuBERT特征, 张量形状 [B, 1024]
      - audio_vggish: 音频VGGish特征, 张量形状 [B, T, 128]
      - video_c3d: 视频C3D特征, 张量形状 [B, t, 4096]
      - video_xclip: 视频XCLIP特征, 张量形状 [B, 8, 512]
      
    输出:
      - logits: 每个样本的未归一化预测分数（真/假）[B, 2]
    """
    def __init__(self, 
                 fixed_seq_len=32, 
                 fixed_dim=128, 
                 nhead=8, 
                 dropout=0.6,
                 # === 新增的可调参数 ===
                 conv_out=128,      # 对应原本的 out_channels=128
                 kernel_size=3,     # 对应原本的 kernel_size=3
                 cls_hidden=56,     # 对应原本的 Linear(128, 56) 中的 56
                 cls_dropout=0.5    # 对应原本的 Dropout(0.5)
                 ):
        super(CoLaVID, self).__init__()
        # 特征维度常量
        self.text_bert_dim = 1536
        self.text_xclip_dim = 1024
        self.audio_hubert_dim = 1024
        self.audio_vggish_dim = 128
        self.video_c3d_dim = 4096
        self.video_xclip_dim = 512

        self.intert_bert_dim = 768
        self.intert_xclip_dim = 1024
        
        self.fixed_seq_len = fixed_seq_len
        self.fixed_dim = fixed_dim

        self.claim_bert_converter = SequenceToFixedDim(
            input_dim=768,  # BERT 默认 pooler_output 是 768
            fixed_seq_len=fixed_seq_len,
            fixed_dim=fixed_dim
        )

        self.claim_xclip_converter = SequenceToFixedDim(
            input_dim=512,  # XCLIP 默认是 512
            fixed_seq_len=fixed_seq_len,
            fixed_dim=fixed_dim
)

        
        ## 1. 序列到固定维度转换层
        # 文本特征转换
        self.text_bert_converter = SequenceToFixedDim(
            input_dim=self.text_bert_dim, 
            fixed_seq_len=fixed_seq_len, 
            fixed_dim=fixed_dim
        )
        self.text_xclip_converter = SequenceToFixedDim(
            input_dim=self.text_xclip_dim, 
            fixed_seq_len=fixed_seq_len, 
            fixed_dim=fixed_dim
        )

        #Internet
        self.internet_bert_converter = SequenceToFixedDim(
            input_dim=self.intert_bert_dim, 
            fixed_seq_len=fixed_seq_len, 
            fixed_dim=fixed_dim
        )
        self.internet_xclip_converter = SequenceToFixedDim(
            input_dim=self.intert_xclip_dim, 
            fixed_seq_len=fixed_seq_len, 
            fixed_dim=fixed_dim
        )
        self.internet_claim_xclip_converter = SequenceToFixedDim(
            input_dim=self.intert_bert_dim, 
            fixed_seq_len=fixed_seq_len, 
            fixed_dim=fixed_dim
        )        
        self.internet_claim_bert_converter = SequenceToFixedDim(
            input_dim=self.intert_bert_dim, 
            fixed_seq_len=fixed_seq_len, 
            fixed_dim=fixed_dim
        )
        #non_internet

        self.non_internet_bert_converter = SequenceToFixedDim(
            input_dim=768, 
            fixed_seq_len=fixed_seq_len, 
            fixed_dim=fixed_dim
        )
        self.non_internet_xclip_converter = SequenceToFixedDim(
            input_dim=512, 
            fixed_seq_len=fixed_seq_len, 
            fixed_dim=fixed_dim
        )

        # 音频特征转换
        self.audio_hubert_converter = SequenceToFixedDim(
            input_dim=self.audio_hubert_dim, 
            fixed_seq_len=fixed_seq_len, 
            fixed_dim=fixed_dim
        )
        self.audio_vggish_converter = SequenceToFixedDim(
            input_dim=self.audio_vggish_dim, 
            fixed_seq_len=fixed_seq_len, 
            fixed_dim=fixed_dim
        )
        # 视频特征转换
        self.video_c3d_converter = SequenceToFixedDim(
            input_dim=self.video_c3d_dim, 
            fixed_seq_len=fixed_seq_len, 
            fixed_dim=fixed_dim
        )
        self.video_xclip_converter = SequenceToFixedDim(
            input_dim=self.video_xclip_dim, 
            fixed_seq_len=fixed_seq_len, 
            fixed_dim=fixed_dim
        )
        
        ## 2. 模态内部融合层 
        # 拼接后维度为 [B, fixed_seq_len, 2*fixed_dim]，不进行降维保留信息
        self.text_fusion = nn.Identity()
        self.audio_fusion = nn.Identity() 
        self.video_fusion = nn.Identity()
        
        ## 3. 单模态自注意力编码器
        d_model = 2 * fixed_dim  # 融合后维度为 2*fixed_dim = 256
        self.self_attn = TransformerBlock(d_model=d_model, nhead=nhead, dropout=dropout)
       
        ## 4. 跨模态Co-Attention模块
        self.co_attn = CoAttentionBlock(d_model=d_model, nhead=nhead, dropout=dropout)
        
        ## 5. 交互注意力后的特征融合层
        self.va_proj = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.ReLU(),
            nn.LayerNorm(d_model),
            nn.Dropout(0.3),
            nn.Linear(d_model, d_model)  # 再次映射保持维度
        )

        # 自适应权重学习模块
        self.num_modalities = 6  # vat_comb, text_emb, internet_emb, non_internet_emb, audio_emb, video_emb
        self.modality_weights = nn.Parameter(torch.ones(self.num_modalities))
        self.softmax_weights = nn.Softmax(dim=0)

        # ## 6. 最终分类器 (参数化修改版)
        self.temporal_conv = nn.Conv1d(
            in_channels=2 * fixed_dim, # 动态计算输入维度 (256 if dim=128, 512 if dim=256)
            out_channels=conv_out,     # 变量: 128
            kernel_size=kernel_size,   # 变量: 3
            padding=kernel_size // 2   # 动态padding: 3->1, 5->2, 保证序列长度不变
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(conv_out, cls_hidden), # 变量: 128 -> 56
            nn.ReLU(),
            nn.LayerNorm(cls_hidden),        # 变量: 56
            nn.Dropout(cls_dropout),         # 变量: 0.5
            nn.Linear(cls_hidden, 2)         # 变量: 56 -> 2
        )
        
        # ## 6. 最终分类器
        # self.classifier = FinalClassifier(d_model=256*4, num_classes=2)
        
        # 用于最后fused_feat的transformer
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=256*4, nhead=8, batch_first=True),
            num_layers=2
        )

        # 初始化权重
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """初始化模型权重"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, batch, device):
        """
        前向传播处理原始模态特征
        输入:
          - batch: 包含所有模态特征和长度的字典
        输出:
          - logits: 未归一化的预测分数 [B, 2]
        """
        modalities = batch['modalities']
        
        # 1. 提取特征和长度信息
        text_bert = modalities['text_bert']['features'].to(device)
        text_xclip = modalities['text_xclip']['features'].to(device)
        
        audio_hubert = modalities['audio_hubert']['features'].to(device)
        audio_vggish = modalities['audio_vggish']['features'].to(device)
        audio_vggish_lens = modalities['audio_vggish']['lengths'].to(device) if modalities['audio_vggish']['lengths'] is not None else None
        
        video_c3d = modalities['video_c3d']['features'].to(device)
        video_c3d_lens = modalities['video_c3d']['lengths'].to(device) if modalities['video_c3d']['lengths'] is not None else None
        
        video_xclip = modalities['video_xclip']['features'].to(device)

        internet_bert = modalities['internet_bert']['features'].to(device) #768
        internet_xclip = modalities['internet_xclip']['features'].to(device) #512

        internet_claim_bert = modalities['internet_bert']['features'].to(device) #768
        internet_claim_xclip = modalities['internet_xclip']['features'].to(device) #512


        non_internet_bert = modalities['non_internet_summary_bert']['features'].to(device)#768
        non_internet_xclip = modalities['non_internet_summary_xclip']['features'].to(device)#512

        
        # 2. 将每种特征转换为统一的(batch, 32, 128)格式
        text_bert_std = self.text_bert_converter(text_bert)
        text_xclip_std = self.text_xclip_converter(text_xclip)
        
        audio_hubert_std = self.audio_hubert_converter(audio_hubert)
        audio_vggish_std = self.audio_vggish_converter(audio_vggish, audio_vggish_lens)
        
        video_c3d_std = self.video_c3d_converter(video_c3d, video_c3d_lens)
        video_xclip_std = self.video_xclip_converter(video_xclip)

        internet_claim_bert_std = self.internet_claim_bert_converter(internet_claim_bert)
        internet_claim_xclip_std = self.internet_claim_xclip_converter(internet_claim_xclip)

        internet_bert_std = self.internet_bert_converter(internet_bert)
        internet_xclip_std = self.internet_xclip_converter(internet_xclip)

        non_internet_bert_std = self.non_internet_bert_converter(non_internet_bert)
        non_internet_xclip_std = self.non_internet_xclip_converter(non_internet_xclip)
        
        # 3. 融合同一模态内的不同特征 - 拼接[B, 32, 256]
        # text_emb = torch.cat([text_bert_std, text_xclip_std], dim=2)
        text_emb = torch.cat([text_bert_std, internet_claim_bert_std, text_xclip_std, internet_claim_xclip_std], dim=2)
        text_emb = F.relu(self.va_proj(text_emb)) 

        audio_emb = torch.cat([audio_hubert_std, audio_vggish_std], dim=2)
        video_emb = torch.cat([video_c3d_std, video_xclip_std], dim=2)

        # internet_claim = torch.cat([internet_claim_bert_std, internet_claim_xclip_std], dim=2)
        
        internet_emb = torch.cat([internet_bert_std, internet_xclip_std], dim=2)
        non_internet_emb = torch.cat([non_internet_bert_std, non_internet_xclip_std], dim=2)

 
        

        # 4. 单模态自注意力编码
        text_enc = self.self_attn(text_emb)
        audio_enc = self.self_attn(audio_emb)
        video_enc = self.self_attn(video_emb)

        internet_emb = self.self_attn(internet_emb)
        non_internet_emb = self.self_attn(non_internet_emb) #[B, 32, 256]
        
        video_va, audio_va = self.co_attn(video_enc, audio_enc)  # 各输出 [B, 32, 256]


        inter_noninter, noninter_inter = self.co_attn(internet_emb, non_internet_emb) # [B, 32, 256]

        t_i,i_t = self.co_attn(text_emb, inter_noninter) # [B, 32, 256]



        # 6. 视频音频-视频和文本的交互注意力
        # 将视频和音频交互后的特征拼接，然后压缩为原维度
        va_comb = torch.cat([video_va, audio_va], dim=2)  # [B, 32, 512]
        va_comb = F.relu(self.va_proj(va_comb))           # [B, 32, 256]
        
        
        # 7. 视频音频-音频和文本的交互注意力
        vaa_co, text_co = self.co_attn(va_comb, t_i) # [B, 32, 256]
        
        # 将文本和视听交互后的特征拼接，然后压缩为原维度
        vat_comb = torch.cat([vaa_co, text_co], dim=2)  # [B, 32, 512]
        vat_comb = F.relu(self.va_proj(vat_comb))   # [B, 32, 256]




        # 8. 自适应权重融合
        # 获取归一化的权重
        weights = self.softmax_weights(self.modality_weights)
        
        # 对每个模态应用权重并融合
        fused_feature = (
            weights[0] * vat_comb +
            weights[1] * text_emb + 
            weights[2] * internet_emb + 
            weights[3] * non_internet_emb + 
            weights[4] * audio_emb + 
            weights[5] * video_emb
        )


        # 8. 最终分类器输出2分类预测分数
        # fused_feature: [B, 32, 1536]
        x = fused_feature.transpose(1, 2)  # [B, 1536, 32]
        x = self.temporal_conv(x)        # [B, 512, 32]
        x = x.transpose(1, 2)            # [B, 32, 512]
        pooled = x.mean(dim=1)           # [B, 512]
        logits = self.classifier(pooled) # [B, 2]


        return logits

class FinalClassifier(nn.Module):
    def __init__(self, d_model=1024, num_classes=2, nhead=8, num_layers=2):
        super(FinalClassifier, self).__init__()
        # 可选：再加一层 Transformer 编码器，增强序列建模
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.LayerNorm(d_model),
            nn.Dropout(0.5),
            nn.Linear(d_model, num_classes)
        )
    
    def forward(self, x):
        # x: [B, 32, 256]
        x = self.transformer(x)  # [B, 32, 256]
        x = x.mean(dim=1)       # [B, 256]
        logits = self.classifier(x)  # [B, 2]
        return logits

class TransformerBlock(nn.Module):
    """
    自定义Transformer块：处理多头自注意力和前馈网络，用于单一模态特征建模
    输入格式：[B, seq_len, feat_dim]
    输出格式：与输入相同 [B, seq_len, feat_dim]
    """
    def __init__(self, d_model=256, nhead=8, dim_feedforward=None, dropout=0.1):
        super(TransformerBlock, self).__init__()
        if dim_feedforward is None:
            dim_feedforward = d_model * 4
        # 多头自注意力层
        self.self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=True)
        # 前馈全连接网络
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        # 层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        输入: x 张量，形状 [B, seq_len, d_model]
        输出: 与输入形状相同，经过自注意力和前馈网络更新的特征
        """
        # 自注意力子层
        attn_output, _ = self.self_attn(x, x, x)
        # 残差连接 + 层归一化
        x = self.norm1(x + self.dropout(attn_output))
        # 前馈网络子层
        ff_output = self.linear2(F.relu(self.linear1(x)))
        # 残差连接 + 层归一化
        x = self.norm2(x + self.dropout(ff_output))
        return x


class CoAttentionBlock(nn.Module):
    """
    双向Co-Attention模块：实现两个输入张量之间的双向交互注意力
    内部使用两个多头注意力层：
      - attn_xy：令X特征作为查询，从Y特征中获取注意力信息
      - attn_yx：令Y特征作为查询，从X特征中获取注意力信息
    """
    def __init__(self, d_model=256, nhead=8, dropout=0.3):
        super(CoAttentionBlock, self).__init__()
        # 用于X从Y中获取信息的多头注意力层
        self.attn_xy = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=True)
        # 用于Y从X中获取信息的多头注意力层
        self.attn_yx = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=True)
        # 层归一化
        self.norm_x = nn.LayerNorm(d_model)
        self.norm_y = nn.LayerNorm(d_model)
        # Dropout层
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, y):
        """
        输入: 
          - x张量 [B, Lx, d_model]
          - y张量 [B, Ly, d_model]
        输出: 
          - x_updated张量，形状与x相同，但融合了来自y的信息
          - y_updated张量，形状与y相同，但融合了来自x的信息
        """
        # X attend on Y：以x为查询，y为键和值，计算注意力
        attn_output_x, _ = self.attn_xy(x, y, y)
        # X融合来自Y的信息，残差连接并层归一化
        x_updated = self.norm_x(x + self.dropout(attn_output_x))
        # Y attend on X：以y为查询，x为键和值，计算注意力
        attn_output_y, _ = self.attn_yx(y, x, x)
        # Y融合来自X的信息，残差连接并层归一化
        y_updated = self.norm_y(y + self.dropout(attn_output_y))
        return x_updated, y_updated


def save_results_to_csv(results_dict, csv_file="model_results.csv"):
    """保存结果到CSV文件"""
    headers = [
        "Drop", "Dim", "Conv_Out", "Kernel_Size", "Cls_Hidden", "Cls_Drop",
        "Fixed_Seq_Len", "Nhead", "Num_Epochs", "Learning_Rate",
        "Best_Val_Acc",
        "Test_Acc", "W_Prec", "W_Rec", "W_F1",
        "Fake_P", "Fake_R", "Fake_F1",
        "Real_P", "Real_R", "Real_F1"
    ]
    
    # 检查文件是否存在，如果不存在则创建并写入表头
    file_exists = os.path.exists(csv_file)
    
    with open(csv_file, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(headers)
        
        # 写入数据行
        row = [
            results_dict.get('Drop', ''),
            results_dict.get('Dim', ''),
            results_dict.get('Conv_Out', ''),
            results_dict.get('Kernel_Size', ''),
            results_dict.get('Cls_Hidden', ''),
            results_dict.get('Cls_Drop', ''),
            results_dict.get('Fixed_Seq_Len', ''),
            results_dict.get('Nhead', ''),
            results_dict.get('Num_Epochs', ''),
            results_dict.get('Learning_Rate', ''),
            results_dict.get('Best_Val_Acc', ''),
            results_dict.get('Test_Acc', ''),
            results_dict.get('W_Prec', ''),
            results_dict.get('W_Rec', ''),
            results_dict.get('W_F1', ''),
            results_dict.get('Fake_P', ''),
            results_dict.get('Fake_R', ''),
            results_dict.get('Fake_F1', ''),
            results_dict.get('Real_P', ''),
            results_dict.get('Real_R', ''),
            results_dict.get('Real_F1', ''),
        ]
        writer.writerow(row)
    
    print(f"结果已保存到: {csv_file}")


def train_model(model, train_loader, val_loader, test_loader, 
                num_epochs=20, learning_rate=2e-4, device='cuda',
                model_params=None, csv_file="model_results.csv", config_id=None):
    """完整的模型训练和评估流程"""
    # 修改损失函数初始化
    class_weights = torch.tensor([1.0, 304/238], device=device)  # 调整类别权重
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=100, verbose=True
    )
    
    best_val_acc = 0.0
    model_dir = os.path.join(os.path.dirname(__file__), "weights")
    os.makedirs(model_dir, exist_ok=True)
    
    # 如果提供了config_id，使用不同的文件名，避免覆盖
    if config_id is not None:
        best_model_path = os.path.join(model_dir, f"best_model_config_{config_id}.pth")
    else:
        best_model_path = os.path.join(model_dir, "best_model.pth")
    print(f"模型将保存到: {best_model_path}")
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        for i, batch in enumerate(train_loader, start=1):
            # 获取标签
            labels = batch['label'].to(device)
            
            # 模型前向传播
            outputs = model(batch, device)
            
            # 计算损失
            loss = criterion(outputs, labels)
            
            # 反向传播和参数更新
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # 打印训练过程
            if i % 10 == 0 or i == len(train_loader):
                print(f"Epoch [{epoch+1}/{num_epochs}] Batch {i}/{len(train_loader)} - Loss: {loss.item():.4f}")
        
        train_loss_avg = running_loss / len(train_loader)
        
        # 验证阶段
        model.eval()
        val_correct = 0
        val_total = 0
        all_val_labels = []
        all_val_preds = []
        
        with torch.no_grad():
            for batch in val_loader:
                labels = batch['label'].to(device)
                outputs = model(batch, device)
                preds = torch.argmax(outputs, dim=1)
                
                all_val_labels.extend(labels.cpu().numpy().tolist())
                all_val_preds.extend(preds.cpu().numpy().tolist())
                
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        
        val_acc = val_correct / val_total
        target_names = ["假", "真"]
        val_report = classification_report(all_val_labels, all_val_preds, target_names=target_names, digits=4)
        
        # 输出验证结果
        print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss_avg:.4f}, Val Accuracy: {val_acc:.4f}")
        print("Validation Classification Report:\n" + val_report)
        
        # 学习率调整
        scheduler.step(val_acc)
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            try:
                os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
                torch.save(model.state_dict(), best_model_path)
                print(f"Best model improved at epoch {epoch+1}, saved to {best_model_path}")
            except Exception as e:
                print(f"保存模型时出错: {e}")
                alt_path = os.path.expanduser("~/best_model.pth")
                torch.save(model.state_dict(), alt_path)
                print(f"模型已保存到备选路径: {alt_path}")
                best_model_path = alt_path
    
    # 训练完成后，使用test_best.py进行测试（通过subprocess独立进程，确保完全隔离和可复现）
    print("\n训练完成，开始使用test_best.py进行测试（独立进程）...")
    try:
        # 准备模型参数（使用传入的model_params或从模型中获取默认值）
        if model_params is None:
            model_params = {
                'dropout': 0.6,
                'fixed_dim': 128,
                'conv_out': 128,
                'kernel_size': 3,
                'cls_hidden': 56,
                'cls_dropout': 0.5,
                'fixed_seq_len': 32,
                'nhead': 8
            }
        
        # 获取test_best.py的路径（与当前脚本在同一目录）
        script_dir = os.path.dirname(os.path.abspath(__file__))
        test_best_script = os.path.join(script_dir, "test_best.py")
        
        # 构建subprocess命令
        cmd = [
            sys.executable,  # 使用当前Python解释器
            test_best_script,
            "--model_path", best_model_path,
            "--fixed_seq_len", str(model_params.get('fixed_seq_len', 32)),
            "--fixed_dim", str(model_params.get('fixed_dim', 128)),
            "--nhead", str(model_params.get('nhead', 8)),
            "--dropout", str(model_params.get('dropout', 0.6)),
            "--conv_out", str(model_params.get('conv_out', 128)),
            "--kernel_size", str(model_params.get('kernel_size', 3)),
            "--cls_hidden", str(model_params.get('cls_hidden', 56)),
            "--cls_dropout", str(model_params.get('cls_dropout', 0.5)),
            "--batch_size", "64",
            "--seed", "123",
            "--json_output"  # 启用JSON输出模式
        ]
        
        print(f"执行命令: {' '.join(cmd)}")
        
        # 执行subprocess，捕获输出
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, cwd=script_dir)
        stdout = result.stdout
        stderr = result.stderr
        
        # 如果有错误输出，打印出来
        if stderr:
            print("测试脚本的stderr输出:")
            print(stderr)
        
        # 解析JSON输出
        test_results = None
        if "@@JSON_START@@" in stdout and "@@JSON_END@@" in stdout:
            json_str = stdout.split("@@JSON_START@@")[1].split("@@JSON_END@@")[0].strip()
            test_results = json.loads(json_str)
            print("成功解析测试结果")
        else:
            print("警告: 无法从输出中解析JSON结果")
            print("stdout前500字符:", stdout[:500])
            raise ValueError("无法解析test_best.py的JSON输出")
        
        # 准备保存到CSV的结果字典
        results_dict = {
            'Drop': model_params.get('dropout', 0.6),
            'Dim': model_params.get('fixed_dim', 128),
            'Conv_Out': model_params.get('conv_out', 128),
            'Kernel_Size': model_params.get('kernel_size', 3),
            'Cls_Hidden': model_params.get('cls_hidden', 56),
            'Cls_Drop': model_params.get('cls_dropout', 0.5),
            'Fixed_Seq_Len': model_params.get('fixed_seq_len', 32),
            'Nhead': model_params.get('nhead', 8),
            'Num_Epochs': num_epochs,
            'Learning_Rate': learning_rate,
            'Best_Val_Acc': f"{best_val_acc:.4f}",
            'Test_Acc': f"{test_results['Test_Acc']:.4f}",
            'W_Prec': f"{test_results['W_Prec']:.4f}",
            'W_Rec': f"{test_results['W_Rec']:.4f}",
            'W_F1': f"{test_results['W_F1']:.4f}",
            'Fake_P': f"{test_results['Fake_P']:.4f}",
            'Fake_R': f"{test_results['Fake_R']:.4f}",
            'Fake_F1': f"{test_results['Fake_F1']:.4f}",
            'Real_P': f"{test_results['Real_P']:.4f}",
            'Real_R': f"{test_results['Real_R']:.4f}",
            'Real_F1': f"{test_results['Real_F1']:.4f}",
        }
        
        # 保存到CSV
        save_results_to_csv(results_dict, csv_file)
        
        print("\n测试结果:")
        print(f"Test Accuracy: {test_results['Test_Acc']:.4f}")
        print(f"Weighted Precision: {test_results['W_Prec']:.4f}")
        print(f"Weighted Recall: {test_results['W_Rec']:.4f}")
        print(f"Weighted F1: {test_results['W_F1']:.4f}")
        print(f"Fake - Precision: {test_results['Fake_P']:.4f}, Recall: {test_results['Fake_R']:.4f}, F1: {test_results['Fake_F1']:.4f}")
        print(f"Real - Precision: {test_results['Real_P']:.4f}, Recall: {test_results['Real_R']:.4f}, F1: {test_results['Real_F1']:.4f}")
        
    except subprocess.CalledProcessError as e:
        print(f"测试脚本执行失败: {e}")
        print(f"返回码: {e.returncode}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
    except Exception as e:
        print(f"测试过程中出错: {e}")
        import traceback
        traceback.print_exc()
    
    return model


# 设置数据路径
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BASE = os.environ.get(
    "EGRT_FAKESV_DATA_DIR",
    os.path.join(REPO_ROOT, "reproduction_data", "FakeSV"),
)
SPLIT_DIR = os.path.join(BASE, "data_split")
FEATURE_DIR = os.environ.get(
    "EGRT_FAKESV_FEATURE_DIR",
    os.path.join(BASE, "features", "features_original"),
)
JSON_PATH = os.path.join(BASE, "data_complete.json")

splits = {
    'train': os.path.join(SPLIT_DIR, "train.txt"),
    'val': os.path.join(SPLIT_DIR, "val.txt"),
    'test': os.path.join(SPLIT_DIR, "test.txt"),
}

# 原始特征文件路径
feature_info = {
    'audio_hubert': {
        'type': 'pkl',
        'path': os.path.join(FEATURE_DIR, "audio_hubert_feats_30s.pkl")
    },
    'audio_vggish': {
        'type': 'pkl',
        'path': os.path.join(FEATURE_DIR, "audio_vggish.pkl")
    },
    'text_bert': {
        'type': 'pkl',
        'path': os.path.join(FEATURE_DIR, "title_ocr_concat_bert.pkl")
    },
    'text_xclip': {
        'type': 'pkl',
        'path': os.path.join(FEATURE_DIR, "title_ocr_concat_xclip.pkl")
    },
    'video_c3d': {
        'type': 'hdf5',
        'path': os.path.join(FEATURE_DIR, "video_c3d.hdf5"),
        'feat_name': "c3d_features"
    },
    'video_xclip': {
        'type': 'h5',
        'path': os.path.join(FEATURE_DIR, "video_xclip.h5"),
        'feat_name': "xclip_features",
        'id_name': "video_ids"
    },
    'internet_bert': {
        'type': 'pkl',
        'path': os.path.join(FEATURE_DIR, "internet_bert_qudiaoduoyuzifu.pkl")
    },
    'internet_xclip': {
        'type': 'pkl',
        'path': os.path.join(FEATURE_DIR, "internet_xclip_qudiaoduoyuzifu.pkl")
    },  
    'internet_claim_bert': {
        'type': 'pkl',
        'path': os.path.join(FEATURE_DIR, "internet_claim_bert_qudiaoduoyuzifu.pkl")
    },
    'internet_claim_xclip': {
        'type': 'pkl',
        'path': os.path.join(FEATURE_DIR, "internet_claim_xclip_qudiaoduoyuzifu.pkl")
    },  
    'non_internet_summary_bert': {
        'type': 'pkl',
        'path': os.path.join(FEATURE_DIR, "non_internet_summary_bert.pkl")
    },
    'non_internet_summary_xclip': {
        'type': 'pkl',
        'path': os.path.join(FEATURE_DIR, "non_internet_summary_xclip.pkl")
    },
}


if __name__ == "__main__":
    # 单组最佳参数配置
    param_grid = [{
        'id': 1,
        'fixed_seq_len': 32,
        'fixed_dim': 128,
        'nhead': 8,
        'dropout': 0.625,
        'conv_out': 104,
        'kernel_size': 3,
        'cls_hidden': 64,
        'cls_dropout': 0.5,
        'learning_rate': 0.00014
    }]
    
    print(f"已生成 {len(param_grid)} 组参数 (单组最佳参数)，准备开始训练...")
    
    print("=" * 80)
    
    # 设置基础随机种子（每个配置训练前会重新设置）
    seed = 123
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 加载数据集（只加载一次，所有配置共享）
    batch_size = 64
    print("\n加载数据集...")
    train_dataset = MultiModalNewsDataset(splits['train'], JSON_PATH, feature_info)
    val_dataset = MultiModalNewsDataset(splits['val'], JSON_PATH, feature_info)
    test_dataset = MultiModalNewsDataset(splits['test'], JSON_PATH, feature_info)
    
    # 使用自定义collate_fn
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=0, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                           num_workers=0, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=0, collate_fn=custom_collate_fn)
    
    print("测试数据加载情况...")
    print(train_dataset.__len__(), "训练样本")
    print(val_dataset.__len__(), "验证样本")
    print(test_dataset.__len__(), "测试样本")
    print("=" * 80)
    
    # CSV文件保存在当前脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file = os.path.join(script_dir, "model_results.csv")
    
    # 只测试一组参数
    config_idx, model_params = 0, param_grid[0]
    config_id = model_params['id']
    print(f"\n{'='*80}")
    print(f"开始测试配置 {config_id}/1")
    print(f"参数: Drop={model_params['dropout']}, Dim={model_params['fixed_dim']}, "
          f"Conv_Out={model_params['conv_out']}, Kernel_Size={model_params['kernel_size']}, "
          f"Cls_Hidden={model_params['cls_hidden']}, Cls_Drop={model_params['cls_dropout']}, "
          f"LR={model_params['learning_rate']}")
    print(f"{'='*80}\n")
    
    try:
        # 每次训练前重新设置随机种子，确保可复现
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # 初始化模型
        model = CoLaVID(
            fixed_seq_len=model_params['fixed_seq_len'],
            fixed_dim=model_params['fixed_dim'],
            nhead=model_params['nhead'],
            dropout=model_params['dropout'],
            conv_out=model_params['conv_out'],
            kernel_size=model_params['kernel_size'],
            cls_hidden=model_params['cls_hidden'],
            cls_dropout=model_params['cls_dropout']
        ).to(device)
        
        print(f"配置 {config_id} - 模型参数总数: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
        
        # 训练模型（训练完成后会自动调用test_best.py进行测试并保存结果）
        trained_model = train_model(
            model, 
            train_loader, 
            val_loader, 
            test_loader,
            num_epochs=20,
            learning_rate=model_params.get('learning_rate', 2e-4),
            device=device,
            model_params=model_params,
            csv_file=csv_file,
            config_id=config_id
        )
        
        print(f"\n配置 {config_id} 完成!")
        
        # 清理GPU内存
        del model
        del trained_model
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"\n配置 {config_id} 训练/测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n{'='*80}")
    print(f"单组参数测试完成! 结果已保存到: {csv_file}")
    print(f"{'='*80}")
