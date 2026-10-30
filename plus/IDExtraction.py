import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DynamicIDExtraction(nn.Module):
    def __init__(self, n_items, n_categories, config):
        super(DynamicIDExtraction, self).__init__()
        self.hidden_size = config.hidden_size
        self.n_demand = config.n_demand
        self.topk = getattr(config, 'topk', 2)
        self.num_heads = 4
        self.embedding_i = nn.Embedding(n_items, self.hidden_size, padding_idx=0)
        self.embedding_c = nn.Embedding(n_categories, self.hidden_size, padding_idx=0)
        self.pos_embedding = nn.Embedding(config.hyper_maximum_session_length + 1, self.hidden_size)
        self.dropout = nn.Dropout(config.drop_item)
        drop_c = getattr(config, 'drop_catgy', 0.0)
        self.dropout_cat = nn.Dropout(drop_c)
        self.layer_norm = nn.LayerNorm(self.hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_size, nhead=self.num_heads,
                                                   dim_feedforward=self.hidden_size * 4, dropout=0.1,
                                                   batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.W_a = nn.Linear(self.hidden_size, self.n_demand, bias=False)
        self.W_alpha = nn.Linear(self.hidden_size, self.hidden_size)
        self.q_alpha = nn.Linear(self.hidden_size, 1, bias=False)

    def forward(self, item_seq, cat_seq, mask_node):
        batch_size, seq_len = item_seq.shape
        device = item_seq.device
        positions = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, -1)
        emb_i = self.embedding_i(item_seq)
        emb_c = self.embedding_c(cat_seq)
        emb_p = self.pos_embedding(positions)
        emb_c = self.dropout_cat(emb_c)
        x = emb_i + emb_c + emb_p
        x = self.dropout(self.layer_norm(x))
        key_padding_mask = (mask_node == 0)
        H = self.transformer_encoder(x, src_key_padding_mask=key_padding_mask)
        att_logits = self.W_a(H)
        att_logits = att_logits.masked_fill(mask_node.unsqueeze(-1) == 0, -1e9)
        A_c = F.softmax(att_logits, dim=1)
        d_c = torch.matmul(A_c.transpose(1, 2), H)
        dot_product = torch.matmul(d_c.unsqueeze(2), H.unsqueeze(1).transpose(-1, -2)).squeeze(2)
        scale = math.sqrt(self.hidden_size)
        dot_product = dot_product.masked_fill(mask_node.unsqueeze(1) == 0, -1e9)
        s_m = torch.sigmoid(dot_product / scale)
        s_m = s_m * mask_node.unsqueeze(1)
        d_f = torch.matmul(s_m, H)
        alpha_hidden = torch.tanh(self.W_alpha(d_f))
        alpha_scores = self.q_alpha(alpha_hidden).squeeze(-1)
        _, topk_indices = torch.topk(alpha_scores, self.topk, dim=1)
        batch_idx = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, self.topk)
        d_active = d_f[batch_idx, topk_indices]
        s_active_scores = s_m[batch_idx, topk_indices]
        return d_active, s_active_scores, topk_indices, H
