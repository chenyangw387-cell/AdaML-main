import torch
import torch.nn as nn
import torch.nn.functional as F


class CoDeRPlusLoss(nn.Module):
    def __init__(self, opt):
        super(CoDeRPlusLoss, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.lambda_seq = getattr(opt, 'lambda_seq', 0.1)
        self.lambda_item = getattr(opt, 'lambda_item', 0.1)
        self.tau = getattr(opt, 'tau', 0.5)
        self.criterion = nn.CrossEntropyLoss()

    def compute_seq_loss(self, z1, z2):
        if z1 is None or z2 is None:
            return torch.tensor(0.0, device=self.device)
        batch_size = z1.shape[0]
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        logits = torch.matmul(z1, z2.T) / self.tau
        labels = torch.arange(batch_size, device=self.device)
        loss_a = self.criterion(logits, labels)
        loss_b = self.criterion(logits.T, labels)
        return (loss_a + loss_b) / 2

    def compute_item_loss(self, s_G, s_active_scores):
        batch_size = s_G.shape[0]
        score_per_demand = s_active_scores.sum(dim=-1)
        score_per_demand = s_active_scores.sum(dim=-1)
        demand_labels = score_per_demand.argmax(dim=1)
        labels = demand_labels.contiguous().view(-1, 1)
        labels = demand_labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(self.device)
        mask_eye = torch.eye(batch_size, device=self.device)
        mask_no_eye = mask - mask_eye
        valid_anchor = (mask_no_eye.sum(1) > 0).float()
        if valid_anchor.sum() == 0:
            return torch.tensor(0.0, device=self.device)
        features = F.normalize(s_G, dim=1)
        anchor_dot_contrast = torch.div(torch.matmul(features, features.T), self.tau)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        exp_logits = torch.exp(logits) * (1 - mask_eye)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-9)
        mean_log_prob_pos = (mask_no_eye * log_prob).sum(1) / (mask_no_eye.sum(1) + 1e-9)
        loss = - (mean_log_prob_pos * valid_anchor).sum() / (valid_anchor.sum() + 1e-9)
        return loss

    def forward(self, l2_loss, result_scores, target_item, aux_info, z_seq1=None, z_seq2=None):
        click_loss = self.criterion(result_scores, target_item)
        if self.lambda_seq > 0 and z_seq1 is not None and z_seq2 is not None:
            loss_seq = self.compute_seq_loss(z_seq1, z_seq2)
        else:
            loss_seq = torch.tensor(0.0, device=self.device)
        if self.lambda_item > 0 and 's_scores' in aux_info and 's_G_k' in aux_info:
            s_G_anchor = aux_info['s_G_k'].mean(dim=1)
            loss_item = self.compute_item_loss(s_G_anchor, aux_info['s_scores'])
        else:
            loss_item = torch.tensor(0.0, device=self.device)
        total_loss = click_loss + l2_loss + \
                     self.lambda_seq * loss_seq + \
                     self.lambda_item * loss_item
        return total_loss, click_loss, loss_seq, loss_item
