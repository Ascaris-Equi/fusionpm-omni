import os
import math
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import argparse

###############################################################################
# 配置
###############################################################################
INFERENCE_CONFIG = {
    "PEP_MAX_LEN": 25,
    "HLA_MAX_LEN": 34,
    "D_MODEL": 128,
    "D_FF": 256,
    "N_HEADS": 8,
    "N_LAYERS": 1,
    "BATCH_SIZE": 2048,
    "LABEL_THRESHOLD": 0.5,
}

###############################################################################
# 数据处理
###############################################################################
AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY-")
vocab_dict  = {aa: i+1 for i,aa in enumerate(AMINO_ACIDS)}
vocab_size  = len(vocab_dict)+1

def seq_to_idx(seq, max_len, vocab):
    """将氨基酸序列转换为索引序列"""
    s = str(seq)[:max_len]
    s = s.ljust(max_len, '-')
    idxs = [vocab.get(ch, 0) for ch in s]
    return idxs

class MixedDataset(torch.utils.data.Dataset):
    """混合标签数据集 - 推理版本"""
    def __init__(self, df, pep_len, hla_len, vocab):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.pep_len = pep_len
        self.hla_len = hla_len
        self.vocab = vocab

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pep = row["Peptide"]
        pse = row["Pseudo_Sequence"]
        
        # 转换序列为索引
        pep_idx = seq_to_idx(pep, self.pep_len, self.vocab)
        pse_idx = seq_to_idx(pse, self.hla_len, self.vocab)
        
        # 推理时不需要标签，但为了保持与训练时相同的接口，添加默认值
        return (
            torch.LongTensor(pep_idx),
            torch.LongTensor(pse_idx),
            torch.tensor(0.0, dtype=torch.float),  # 占位符
            torch.tensor(0, dtype=torch.long)  # 占位符
        )

###############################################################################
# 模型定义
###############################################################################
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(position*div_term)
        pe[:, 1::2] = torch.cos(position*div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)

def get_attn_pad_mask(seq_q, seq_k):
    b, len_q = seq_q.size()
    b, len_k = seq_k.size()
    mask = seq_k.eq(0).unsqueeze(1)
    return mask.expand(b, len_q, len_k)

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super().__init__()
        self.d_k = d_k
    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1,-2)) / math.sqrt(self.d_k)
        scores.masked_fill_(attn_mask, -1e9)
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v
        self.W_Q = nn.Linear(d_model, d_k*n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k*n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v*n_heads, bias=False)
        self.fc  = nn.Linear(n_heads*d_v, d_model, bias=False)
        self.layer_norm = nn.LayerNorm(d_model)
    def forward(self, Q_in, K_in, V_in, mask):
        residual = Q_in
        B = Q_in.size(0)
        Q = self.W_Q(Q_in).view(B, -1, self.n_heads, self.d_k).transpose(1,2)
        K = self.W_K(K_in).view(B, -1, self.n_heads, self.d_k).transpose(1,2)
        V = self.W_V(V_in).view(B, -1, self.n_heads, self.d_v).transpose(1,2)
        mask = mask.unsqueeze(1).repeat(1,self.n_heads,1,1)
        context, attn = ScaledDotProductAttention(self.d_k)(Q, K, V, mask)
        context = context.transpose(1,2).reshape(B, -1, self.n_heads*self.d_v)
        out = self.fc(context)
        return self.layer_norm(out + residual), attn

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False)
        )
        self.layer_norm = nn.LayerNorm(d_model)
    def forward(self, x):
        residual = x
        out = self.fc(x)
        return self.layer_norm(out + residual)

class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, d_k, d_v, n_heads):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, d_k, d_v, n_heads)
        self.ffn = PoswiseFeedForwardNet(d_model, d_ff)
    def forward(self, enc_in, attn_mask):
        enc_out, attn = self.self_attn(enc_in, enc_in, enc_in, attn_mask)
        enc_out = self.ffn(enc_out)
        return enc_out, attn

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, d_ff, d_k, d_v, n_heads, n_layers, max_len=5000):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, d_ff, d_k, d_v, n_heads)
            for _ in range(n_layers)
        ])
    def forward(self, x):
        out = self.embed(x)
        out = self.pos_emb(out)
        mask= get_attn_pad_mask(x, x)
        attns=[]
        for layer in self.layers:
            out, attn = layer(out, mask)
            attns.append(attn)
        return out, attns

class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, d_k, d_v, n_heads):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, d_k, d_v, n_heads)
        self.ffn = PoswiseFeedForwardNet(d_model, d_ff)
    def forward(self, dec_in, mask):
        out, attn = self.self_attn(dec_in, dec_in, dec_in, mask)
        out = self.ffn(out)
        return out, attn

class Decoder(nn.Module):
    def __init__(self, d_model, d_ff, d_k, d_v, n_heads, n_layers, max_len=5000):
        super().__init__()
        self.pos_emb = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, d_ff, d_k, d_v, n_heads)
            for _ in range(n_layers)
        ])
    def forward(self, x):
        x = self.pos_emb(x)
        B,S,_ = x.size()
        mask = torch.zeros((B,S,S), dtype=torch.bool, device=x.device)
        attns=[]
        for layer in self.layers:
            x, attn = layer(x, mask)
            attns.append(attn)
        return x, attns

class Transformer(nn.Module):
    """Transformer模型"""
    def __init__(self,
                 vocab_size,
                 d_model,
                 d_ff,
                 d_k,
                 d_v,
                 n_heads,
                 n_layers,
                 pep_len,
                 hla_len):
        super().__init__()
        self.pep_encoder = Encoder(vocab_size, d_model, d_ff, d_k, d_v, n_heads, n_layers)
        self.hla_encoder = Encoder(vocab_size, d_model, d_ff, d_k, d_v, n_heads, n_layers)
        self.decoder     = Decoder(d_model, d_ff, d_k, d_v, n_heads, n_layers)
        self.tgt_len     = pep_len + hla_len

        # 共享特征提取层
        self.shared_layers = nn.Sequential(
            nn.Linear(self.tgt_len*d_model, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 64),
            nn.ReLU(),
        )
        
        # 分类输出层 - 正类和负类概率
        self.cls_proj = nn.Linear(64, 2)
        
        # 回归输出层 - 亲和力
        self.aff_proj = nn.Linear(64, 1)

    def forward(self, pep_in, pse_in):
        pep_out, _ = self.pep_encoder(pep_in)
        pse_out, _ = self.hla_encoder(pse_in)
        enc_out = torch.cat([pep_out, pse_out], dim=1)
        dec_out, _ = self.decoder(enc_out)
        dec_out = dec_out.view(dec_out.size(0), -1)
        
        # 共享特征
        shared_features = self.shared_layers(dec_out)
        
        # 分类输出 (正负类概率)
        cls_logits = self.cls_proj(shared_features)
        
        # 亲和力输出
        affinity = self.aff_proj(shared_features)
        
        return cls_logits, affinity.squeeze(-1)

###############################################################################
# 推理函数
###############################################################################
def predict(input_csv, output_csv, model_path, config=None):
    """执行模型推理并保存结果"""
    if config is None:
        config = INFERENCE_CONFIG
    
    # 检测设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载输入数据
    print(f"加载数据: {input_csv}")
    input_df = pd.read_csv(input_csv)
    
    # 确保列名一致
    if 'MHC' in input_df.columns and 'Pseudo_Sequence' not in input_df.columns:
        input_df = input_df.rename(columns={'MHC': 'Pseudo_Sequence'})
    
    # 创建数据集和加载器
    dataset = MixedDataset(
        input_df,
        pep_len=config["PEP_MAX_LEN"],
        hla_len=config["HLA_MAX_LEN"],
        vocab=vocab_dict
    )
    
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config["BATCH_SIZE"],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # 初始化模型
    print(f"初始化模型并加载权重: {model_path}")
    model = Transformer(
        vocab_size=vocab_size,
        d_model=config["D_MODEL"],
        d_ff=config["D_FF"],
        d_k=config["D_MODEL"]//config["N_HEADS"],
        d_v=config["D_MODEL"]//config["N_HEADS"],
        n_heads=config["N_HEADS"],
        n_layers=config["N_LAYERS"],
        pep_len=config["PEP_MAX_LEN"],
        hla_len=config["HLA_MAX_LEN"]
    ).to(device)
    
    # 加载模型权重
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # 执行推理
    print("开始推理...")
    all_pred_values = []
    all_pred_probs = []
    
    with torch.no_grad():
        for pep, pse, _, _ in loader:
            pep, pse = pep.to(device), pse.to(device)
            
            # 模型推理
            cls_logits, affinity = model(pep, pse)
            
            # 获取预测结果
            probs = nn.Softmax(dim=1)(cls_logits)
            pos_probs = probs[:, 1].cpu().numpy()
            
            # 收集预测结果
            all_pred_values.extend(affinity.cpu().numpy())
            all_pred_probs.extend(pos_probs)
    
    # 将预测结果添加到原始数据
    input_df['Predicted_Value'] = all_pred_values
    input_df['Predicted_Probability'] = all_pred_probs
    
    # 根据阈值增加二分类标签
    input_df['Predicted_Binary'] = (input_df['Predicted_Value'] >= config["LABEL_THRESHOLD"]).astype(int)
    
    # 保存预测结果
    print(f"保存预测结果到: {output_csv}")
    input_df.to_csv(output_csv, index=False)
    
    print("推理完成!")
    return input_df

###############################################################################
# 主函数
###############################################################################
def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='HLA-peptide binding prediction inference')
    parser.add_argument('--input', type=str, default='input.csv', help='Input CSV file path')
    parser.add_argument('--output', type=str, default='output.csv', help='Output CSV file path')
    parser.add_argument('--model', type=str, default='best_model.pth', help='Model weights file path')
    
    args = parser.parse_args()
    
    # 执行推理
    predict(args.input, args.output, args.model)

if __name__ == "__main__":
    main()