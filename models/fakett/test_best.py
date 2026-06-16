# test_only.py
import torch
import random
import numpy as np
import argparse
import sys
import json

from torch.utils.data import DataLoader
# 记得引入你的模型类和数据集类
from CoLaVID_10_claim_TT import CoLaVID, MultiModalNewsDataset, custom_collate_fn, feature_info, JSON_PATH, splits
import os
from sklearn.metrics import classification_report, accuracy_score

def set_deterministic(seed=123):
    # 设置所有随机种子
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 多GPU
    random.seed(seed)
    np.random.seed(seed)
    
    # 设置确定性算法
    torch.use_deterministic_algorithms(True, warn_only=True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    
    # 禁用cudnn的不确定性操作
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def clean_test(model_path, fixed_seq_len=32, fixed_dim=128, nhead=8, dropout=0.4, 
               conv_out=128, kernel_size=5, cls_hidden=56, cls_dropout=0.5, 
               batch_size=64, seed=123, return_dict=False):
    """
    测试模型并返回详细结果
    
    Args:
        model_path: 模型权重文件路径
        fixed_seq_len: 模型参数
        fixed_dim: 模型参数
        nhead: 模型参数
        dropout: 模型参数
        conv_out: 模型参数
        kernel_size: 模型参数
        cls_hidden: 模型参数
        cls_dropout: 模型参数
        batch_size: 批次大小
        seed: 随机种子
        return_dict: 是否返回结果字典（用于被其他脚本调用）
    
    Returns:
        如果return_dict=True，返回包含所有指标字典；否则返回None（仅打印）
    """
    set_deterministic(seed)
    # 1. 强行设置确定性环境（最严格模式）
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True, warn_only=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not return_dict:
        print(f"Testing on device: {device}")

    # 2. 加载数据 
    test_dataset = MultiModalNewsDataset(splits['test'], JSON_PATH, feature_info)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=0, collate_fn=custom_collate_fn)

    # 3. 初始化空模型（使用传入的参数）
    model = CoLaVID(
        fixed_seq_len=fixed_seq_len,
        fixed_dim=fixed_dim,
        nhead=nhead,
        dropout=dropout,
        conv_out=conv_out,
        kernel_size=kernel_size,
        cls_hidden=cls_hidden,
        cls_dropout=cls_dropout
    ).to(device)
    
    # 4. 加载权重文件
    if not return_dict:
        print(f"Loading weights from: {model_path}")
    
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    
    # 5. 开启评估模式 (至关重要)
    model.eval()
    
    # 6. 推理
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if not return_dict:
                print(f"\rTesting batch {i+1}/{len(test_loader)}...", end="")
            labels = batch['label'].to(device)
            outputs = model(batch, device)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
    
    if not return_dict:
        print("\nDone!")
    
    # 7. 计算详细指标
    target_names = ["假", "真"]
    report_dict = classification_report(all_labels, all_preds, target_names=target_names, 
                                       digits=4, output_dict=True)
    
    # 提取指标
    test_acc = accuracy_score(all_labels, all_preds)
    
    # Weighted avg metrics
    w_prec = report_dict['weighted avg']['precision']
    w_rec = report_dict['weighted avg']['recall']
    w_f1 = report_dict['weighted avg']['f1-score']
    
    # Fake metrics (假)
    fake_p = report_dict['假']['precision']
    fake_r = report_dict['假']['recall']
    fake_f1 = report_dict['假']['f1-score']
    
    # Real metrics (真)
    real_p = report_dict['真']['precision']
    real_r = report_dict['真']['recall']
    real_f1 = report_dict['真']['f1-score']
    
    # 输出报告
    if not return_dict:
        print(classification_report(all_labels, all_preds, target_names=target_names, digits=4))
    
    # 构建结果字典
    result_dict = {
        'Test_Acc': test_acc,
        'W_Prec': w_prec,
        'W_Rec': w_rec,
        'W_F1': w_f1,
        'Fake_P': fake_p,
        'Fake_R': fake_r,
        'Fake_F1': fake_f1,
        'Real_P': real_p,
        'Real_R': real_r,
        'Real_F1': real_f1,
        'classification_report': report_dict
    }
    
    if return_dict:
        return result_dict
    else:
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test model using test_best.py')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model weights')
    parser.add_argument('--fixed_seq_len', type=int, default=32, help='Model parameter: fixed_seq_len')
    parser.add_argument('--fixed_dim', type=int, default=128, help='Model parameter: fixed_dim')
    parser.add_argument('--nhead', type=int, default=8, help='Model parameter: nhead')
    parser.add_argument('--dropout', type=float, default=0.4, help='Model parameter: dropout')
    parser.add_argument('--conv_out', type=int, default=128, help='Model parameter: conv_out')
    parser.add_argument('--kernel_size', type=int, default=5, help='Model parameter: kernel_size')
    parser.add_argument('--cls_hidden', type=int, default=56, help='Model parameter: cls_hidden')
    parser.add_argument('--cls_dropout', type=float, default=0.5, help='Model parameter: cls_dropout')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for testing')
    parser.add_argument('--seed', type=int, default=123, help='Random seed')
    parser.add_argument('--json_output', action='store_true', help='Output results as JSON for subprocess calling')
    
    args = parser.parse_args()
    
    result = clean_test(
        model_path=args.model_path,
        fixed_seq_len=args.fixed_seq_len,
        fixed_dim=args.fixed_dim,
        nhead=args.nhead,
        dropout=args.dropout,
        conv_out=args.conv_out,
        kernel_size=args.kernel_size,
        cls_hidden=args.cls_hidden,
        cls_dropout=args.cls_dropout,
        batch_size=args.batch_size,
        seed=args.seed,
        return_dict=args.json_output  # 如果使用json_output，则返回字典
    )
    
    # 如果使用json_output，输出JSON格式的结果（供subprocess解析）
    if args.json_output and result:
        # 只输出必要的指标，不包括完整的classification_report（太长了）
        json_result = {
            'Test_Acc': result['Test_Acc'],
            'W_Prec': result['W_Prec'],
            'W_Rec': result['W_Rec'],
            'W_F1': result['W_F1'],
            'Fake_P': result['Fake_P'],
            'Fake_R': result['Fake_R'],
            'Fake_F1': result['Fake_F1'],
            'Real_P': result['Real_P'],
            'Real_R': result['Real_R'],
            'Real_F1': result['Real_F1'],
        }
        # 使用特殊标记，方便解析
        print("@@JSON_START@@")
        print(json.dumps(json_result, indent=2))
        print("@@JSON_END@@")