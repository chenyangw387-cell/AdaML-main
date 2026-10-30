import math
import torch
import torch.nn as nn
from plus.IDExtraction import DynamicIDExtraction
from plus.IDGNN import DynamicIDGNN
from plus.NoiseEstimator import NoiseEstimator
from layer.match import Match


class CoDeRPlus(nn.Module):
    def __init__(self, n_items, n_categories, config):
        super(CoDeRPlus, self).__init__()
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_items = n_items
        self.n_categories = n_categories
        self.n_demand = config.n_demand
        self.hidden_size = config.hidden_size
        self.ad = config.ad
        self.demand_extractor = DynamicIDExtraction(n_items, n_categories, config)
        self.gnn_layer = DynamicIDGNN(n_items, config)
        self.noise_estimator = None
        if self.ad:
            self.noise_estimator = NoiseEstimator(config, n_items)
        self.match = Match(2 * config.hidden_size, config.batch_norm, self.n_demand, config.rs)
        self.embedding_i = self.demand_extractor.embedding_i
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for name, param in self.named_parameters():
            if 'embedding' not in name and param.requires_grad:
                nn.init.uniform_(param, -stdv, stdv)

    def compute_score(self, session_rep, candidate_emb):
        if candidate_emb.dim() == 2 and candidate_emb.shape[0] != session_rep.shape[0]:
            return torch.matmul(session_rep, candidate_emb.t())
        return torch.sum(session_rep.unsqueeze(1) * candidate_emb, dim=-1)

    def forward(self, nodes, categories, mask_node, target_item=None, candidate_embeddings=None):
        batch_size = nodes.shape[0]
        d_active, s_scores, topk_idx, H = self.demand_extractor(nodes, categories, mask_node)
        seq_len = nodes.shape[1]
        adj_seq = torch.zeros(batch_size, seq_len, seq_len, device=self.device)
        indices = torch.arange(seq_len - 1, device=self.device)
        adj_seq[:, indices, indices + 1] = 1.0
        s_G_k, H_updated = self.gnn_layer(H, s_scores, adj_seq, nodes, mask_node)
        s_G = torch.mean(s_G_k, dim=1)
        if not self.ad or self.noise_estimator is None:
            if candidate_embeddings is None:
                candidate_embeddings = self.embedding_i.weight
            final_scores = self.compute_score(s_G, candidate_embeddings)
            eta = None
        else:
            if self.training:
                if target_item is None: raise ValueError("Training mode requires target_item!")
                all_items_emb = self.embedding_i.weight
                y_TE_all = self.compute_score(s_G, all_items_emb)
                v_t_true = self.embedding_i(target_item)
                eta = self.noise_estimator(s_G, v_t_true, d_active, s_scores, s_G_k, mask_node, target_item)
                final_scores = self.noise_estimator.estimate_causal_effect(y_TE_all, eta)
            else:
                if candidate_embeddings is None: candidate_embeddings = self.embedding_i.weight
                y_TE = self.compute_score(s_G, candidate_embeddings)
                proxy_item_id = torch.argmax(y_TE[:, 1:], dim=1) + 1
                v_t_proxy = self.embedding_i(proxy_item_id)
                eta = self.noise_estimator(s_G, v_t_proxy, d_active, s_scores, s_G_k, mask_node, proxy_item_id)
                final_scores = self.noise_estimator.estimate_causal_effect(y_TE, eta)
        aux_info = {
            'd_active': d_active, 'H_updated': H_updated, 's_G_k': s_G_k,
            's_scores': s_scores, 'eta': eta
        }
        return final_scores, aux_info
