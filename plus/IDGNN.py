import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicIDGNN(nn.Module):
    def __init__(self, n_items, config):
        super(DynamicIDGNN, self).__init__()
        self.hidden_size = config.hidden_size
        self.n_gnn_layers = config.n_gnn_layer
        self.n_items = n_items
        self.top_k_sem = 3
        self.lambda_seq = nn.Parameter(torch.tensor(0.33))
        self.lambda_sem = nn.Parameter(torch.tensor(0.33))
        self.lambda_co = nn.Parameter(torch.tensor(0.33))
        self.global_co_emb = nn.Embedding(n_items, self.hidden_size, padding_idx=0)
        self.W_in = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.gru = nn.GRUCell(self.hidden_size, self.hidden_size)
        self.W_beta = nn.Linear(self.hidden_size, self.hidden_size)
        self.U_beta = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_beta = nn.Linear(self.hidden_size, 1, bias=False)

    def get_semantic_adj(self, H, mask):
        H_norm = F.normalize(H, p=2, dim=-1)
        sim_matrix = torch.bmm(H_norm, H_norm.transpose(1, 2))
        mask_2d = mask.unsqueeze(1) * mask.unsqueeze(2)
        seq_len = H.shape[1]
        eye = torch.eye(seq_len, device=H.device).unsqueeze(0)
        sim_matrix = sim_matrix * mask_2d
        sim_matrix = sim_matrix.masked_fill(eye == 1, -1e9)
        k = min(self.top_k_sem, seq_len - 1)
        if k > 0:
            topk_val, topk_idx = torch.topk(sim_matrix, k, dim=-1)
            sparse_sim = torch.zeros_like(sim_matrix)
            sparse_sim.scatter_(2, topk_idx, topk_val)
            return F.relu(sparse_sim)
        else:
            return torch.zeros_like(sim_matrix)

    def get_global_co_adj(self, item_seq, mask):
        global_h = self.global_co_emb(item_seq)
        H_norm = F.normalize(global_h, p=2, dim=-1)
        sim_matrix = torch.bmm(H_norm, H_norm.transpose(1, 2))
        mask_2d = mask.unsqueeze(1) * mask.unsqueeze(2)
        seq_len = item_seq.shape[1]
        eye = torch.eye(seq_len, device=item_seq.device).unsqueeze(0)
        sim_matrix = sim_matrix * mask_2d
        sim_matrix = sim_matrix.masked_fill(eye == 1, -1e9)
        return F.relu(sim_matrix)

    def forward(self, H, s_active_scores, adj_seq, item_seq, mask_node):
        batch_size, k_demand, seq_len = s_active_scores.shape
        hidden_dim = self.hidden_size
        adj_seq_expanded = adj_seq.unsqueeze(1).expand(-1, k_demand, -1, -1).reshape(batch_size * k_demand, seq_len,
                                                                                     seq_len)
        adj_sem = self.get_semantic_adj(H, mask_node)
        adj_sem_expanded = adj_sem.unsqueeze(1).expand(-1, k_demand, -1, -1).reshape(batch_size * k_demand, seq_len,
                                                                                     seq_len)
        adj_co = self.get_global_co_adj(item_seq, mask_node)
        adj_co_expanded = adj_co.unsqueeze(1).expand(-1, k_demand, -1, -1).reshape(batch_size * k_demand, seq_len,
                                                                                   seq_len)
        adj_total = (self.lambda_seq * adj_seq_expanded +
                     self.lambda_sem * adj_sem_expanded +
                     self.lambda_co * adj_co_expanded)
        H_expanded = H.unsqueeze(1).expand(-1, k_demand, -1, -1).reshape(batch_size * k_demand, seq_len, hidden_dim)
        s_scores_flat = s_active_scores.reshape(batch_size * k_demand, seq_len)
        curr_H = H_expanded
        for _ in range(self.n_gnn_layers):
            Z = self.W_in(curr_H)
            Z_gated = Z * s_scores_flat.unsqueeze(-1)
            m = torch.bmm(adj_total, Z_gated)
            m_reshaped = m.reshape(-1, hidden_dim)
            h_reshaped = curr_H.reshape(-1, hidden_dim)
            new_h = self.gru(m_reshaped, h_reshaped)
            curr_H = new_h.reshape(batch_size * k_demand, seq_len, hidden_dim)
        H_updated = curr_H.reshape(batch_size, k_demand, seq_len, hidden_dim)
        lengths = mask_node.sum(dim=1).long() - 1
        lengths_expanded = lengths.unsqueeze(1).expand(-1, k_demand).reshape(-1)
        batch_idx = torch.arange(batch_size * k_demand, device=H.device)
        h_last = curr_H[batch_idx, lengths_expanded]
        energy = self.v_beta(torch.tanh(self.W_beta(curr_H) + self.U_beta(h_last).unsqueeze(1)))
        mask_flat = mask_node.unsqueeze(1).expand(-1, k_demand, -1).reshape(batch_size * k_demand, seq_len)
        energy = energy.masked_fill(mask_flat.unsqueeze(-1) == 0, -1e9)
        beta = F.softmax(energy, dim=1)
        s_G_flat = torch.sum(beta * curr_H, dim=1)
        s_G_k = s_G_flat.reshape(batch_size, k_demand, hidden_dim)
        return s_G_k, H_updated
