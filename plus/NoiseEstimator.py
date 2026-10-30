import torch
import torch.nn as nn


class NoiseEstimator(nn.Module):
    def __init__(self, config, n_items):
        super(NoiseEstimator, self).__init__()
        self.hidden_size = config.hidden_size
        self.popularity_embedding = nn.Embedding(n_items, 1)
        torch.nn.init.zeros_(self.popularity_embedding.weight)
        input_dim = 4 * self.hidden_size + 2
        self.W1 = nn.Linear(input_dim, self.hidden_size)
        self.w2 = nn.Linear(self.hidden_size, 1)
        self.dropout = nn.Dropout(0.1)

    def calculate_features(self, s_G, v_t, d_active, s_active_scores, s_G_k, mask_node, item_id):
        d_t = torch.einsum('bkl,bkd->bld', s_active_scores, d_active)
        delta_d = d_t[:, 1:, :] - d_t[:, :-1, :]
        mask_diff = mask_node[:, 1:].unsqueeze(-1)
        delta_d = delta_d * mask_diff
        valid_len = torch.clamp(mask_node.sum(dim=1, keepdim=True) - 1, min=1.0)
        mean_delta_d = torch.sum(delta_d, dim=1) / valid_len
        mean_expanded = mean_delta_d.unsqueeze(1)
        variance_vec = torch.sum((delta_d - mean_expanded) ** 2 * mask_diff, dim=1) / valid_len
        std_delta_d = torch.sqrt(variance_vec + 1e-8)
        mu = torch.mean(s_G_k, dim=1, keepdim=True)
        diff_norm_sq = torch.norm(s_G_k - mu, p=2, dim=-1) ** 2
        var_s_k = torch.mean(diff_norm_sq, dim=1, keepdim=True)
        item_popularity = self.popularity_embedding(item_id)
        f = torch.cat([
            s_G, v_t, mean_delta_d, std_delta_d, var_s_k,
            item_popularity
        ], dim=-1)
        return f

    def forward(self, s_G, v_t, d_active, s_active_scores, s_G_k, mask_node, item_id):
        f = self.calculate_features(s_G, v_t, d_active, s_active_scores, s_G_k, mask_node, item_id)
        hidden = torch.relu(self.W1(f))
        hidden = self.dropout(hidden)
        logit = self.w2(hidden)
        eta = torch.sigmoid(logit)
        return eta

    def estimate_causal_effect(self, y_TE, eta):
        y_NDE = eta * y_TE
        y_TIE = y_TE - y_NDE
        return y_TIE
