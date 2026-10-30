import os
import time
import torch
import argparse
import datetime
import pickle
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from util.utils import seed_torch, file_write, cprint, print_result, print_auc_result
from data.dataset import load_data, RsData, sess_collate_fn
from util.metric import evaluate
from plus.model import CoDeRPlus
from plus.loss import CoDeRPlusLoss


def str2bool(v):
    if isinstance(v, bool): return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def train(log_path, model, criterion, optimizer, train_loader, device, epoch, opt):
    file_write(log_path, f'start training: {datetime.datetime.now()}')
    model.train()
    batch_loss = []
    batch_main_loss = []
    batch_seq_loss = []
    batch_item_loss = []
    start_batch = time.time()
    for i, (sess_nodes, sess_categories, adj_matrixes, nodes_categories, target,
            session_last_item, mask_node, session_last_catgy, mask_catgy, sess_full_items) in enumerate(train_loader):
        nodes = sess_full_items.to(device)
        categories = sess_categories.to(device)
        mask = mask_catgy.to(device)
        target_item = target[:, 0].to(device)
        optimizer.zero_grad()
        scores, aux_info = model(nodes, categories, mask, target_item=target_item)
        _, aux_info_aug = model(nodes, categories, mask, target_item=target_item)
        z1 = torch.mean(aux_info['s_G_k'], dim=1)
        z2 = torch.mean(aux_info_aug['s_G_k'], dim=1)
        l2_reg = torch.tensor(0., device=device)
        if opt.embed_l2 > 0:
            for name, param in model.named_parameters():
                if 'embedding' in name or 'weight' in name:
                    l2_reg += param.pow(2).sum()
        l2_loss = opt.embed_l2 * l2_reg
        loss, main_loss, l_seq, l_item = criterion(l2_loss, scores, target_item, aux_info, z1, z2)
        loss.backward()
        optimizer.step()
        batch_loss.append(loss.item())
        batch_main_loss.append(main_loss.item())
        batch_seq_loss.append(l_seq.item())
        batch_item_loss.append(l_item.item())
        if i % 100 == 0:
            log_str = (f'[Epoch {epoch} Batch {i}] Loss: {loss.item():.4f} '
                       f'(Main: {main_loss.item():.4f}, Seq: {l_seq.item():.4f}, Item: {l_item.item():.4f})')
            print(log_str)
            file_write(log_path, log_str, whether_print=False)
    mean_loss = sum(batch_loss) / len(batch_loss)
    file_write(log_path, f'\tEpoch {epoch} Loss mean: {mean_loss:.4f}')


def test(opt, model, test_loader, device, Ks, Ks_auc):
    log_file = opt.log_path
    print(f'\t start predicting... @ {datetime.datetime.now()}')
    model.eval()
    recalls, mrrs, ndcgs, aucs = [], [], [], []
    all_eta_values = []
    with torch.no_grad():
        for i, (sess_nodes, sess_categories, adj_matrixes, nodes_categories, target,
                session_last_item, mask_node, session_last_catgy, mask_catgy, sess_full_items) in enumerate(
            test_loader):
            nodes = sess_full_items.to(device)
            categories = sess_categories.to(device)
            mask = mask_catgy.to(device)
            target_item = target[:, 0].to(device)
            scores, aux_info = model(nodes, categories, mask)
            metrics = evaluate(scores, target_item, Ks, Ks_auc)
            recalls.append(metrics['recall'])
            mrrs.append(metrics['mrr'])
            ndcgs.append(metrics['ndcg'])
            aucs.append(metrics['auc'])
            if 'eta' in aux_info and aux_info['eta'] is not None:
                all_eta_values.append(aux_info['eta'].detach().cpu())
    if all_eta_values:
        final_eta_tensor = torch.cat(all_eta_values)
        eta_mean = final_eta_tensor.mean().item()
        eta_std = final_eta_tensor.std().item()
        eta_min = final_eta_tensor.min().item()
        eta_max = final_eta_tensor.max().item()
        eta_count = final_eta_tensor.numel()
        eta_summary_str = (f"\n--- Overall ETA Diagnose (ad={opt.ad}) ---\n"
                           f"    Mean: {eta_mean:.4f}\n"
                           f"    Std Dev: {eta_std:.4f}\n"
                           f"    Min: {eta_min:.4f}\n"
                           f"    Max: {eta_max:.4f}\n"
                           f"    Total Samples: {eta_count}\n"
                           f"--------------------------")
        cprint(log_file, eta_summary_str)
    recall_mean = np.mean(recalls, axis=0) * 100
    mrr_mean = np.mean(mrrs, axis=0) * 100
    ndcg_mean = np.mean(ndcgs, axis=0) * 100
    auc_mean = np.mean(aucs, axis=0) * 100
    return recall_mean, mrr_mean, ndcg_mean, auc_mean


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='tafeng', help='dataset name')
    parser.add_argument('--epoch', type=int, default=3000)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--embed_l2', type=float, default=1e-5)
    parser.add_argument('--drop_item', type=float, default=0.1)
    parser.add_argument('--drop_catgy', type=float, default=0.0, help='category dropout rate')
    parser.add_argument('--hyper_maximum_session_length', type=int, default=50)
    parser.add_argument('--n_demand', type=int, default=8)
    parser.add_argument('--topk', type=int, default=2)
    parser.add_argument('--n_gnn_layer', type=int, default=1)
    parser.add_argument('--ad', type=str2bool, default=True, help='whether to enable adjustment module')
    parser.add_argument('--lambda_seq', type=float, default=0.1)
    parser.add_argument('--lambda_item', type=float, default=0.1)
    parser.add_argument('--tau', type=float, default=0.5)
    parser.add_argument('--batch_norm', type=str, default='feature')
    parser.add_argument('--rs', type=str, default='dot')
    parser.add_argument('--alpha', type=float, default=0.85, help='critical value for modularity gain')
    opt = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu_id)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    time_path = time.strftime("%y%m%d-%H%M%S", time.localtime(time.time()))
    ad_str = "AD" if opt.ad else "noAD"
    log_folder_name = f"AdaML_{time_path}_M{opt.n_demand}_K{opt.topk}_{ad_str}_BS{opt.batch_size}"
    log_path = os.path.join('./log', opt.dataset, log_folder_name)
    if not os.path.exists(log_path): os.makedirs(log_path)
    log_file = os.path.join(log_path, "log.txt")
    opt.log_path = log_file
    seed_torch(log_file, opt.seed)
    file_write(log_file, str(opt))
    dataset_dir = os.path.join('datasets', opt.dataset)
    train_data, valid_data, test_data, _, session_info = load_data(dataset_dir)
    train_loader = DataLoader(RsData(log_file, train_data), batch_size=opt.batch_size, shuffle=True,
                              collate_fn=sess_collate_fn, num_workers=4)
    valid_loader = DataLoader(RsData(log_file, valid_data), batch_size=opt.batch_size, shuffle=False,
                              collate_fn=sess_collate_fn, num_workers=4)
    test_loader = DataLoader(RsData(log_file, test_data), batch_size=opt.batch_size, shuffle=False,
                             collate_fn=sess_collate_fn, num_workers=4)
    try:
        n_items = session_info["num_of_item[not include 0]"] + 1
    except:
        n_items = session_info["num_of_item"] + 1
    n_categories = session_info["num_of_category"] + 1
    file_write(log_file, f"CoDeR++ Config: M={opt.n_demand}, K={opt.topk}, Ad={opt.ad}")
    file_write(log_file, f"Dataset: Items={n_items}, Cats={n_categories}")
    model = CoDeRPlus(n_items, n_categories, opt).to(device)
    criterion = CoDeRPlusLoss(opt).to(device)
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    Ks = [10, 20, 40]
    Ks_auc = [50, 100]
    best_res = {'recall': 0.0, 'epoch': 0}
    bad_counter = 0
    for epoch in range(opt.epoch):
        train(log_file, model, criterion, optimizer, train_loader, device, epoch, opt)
        recall, mrr, ndcg, _ = test(opt, model, valid_loader, device, Ks, Ks_auc)
        cprint(log_file, f"Validation Result for Epoch {epoch}:")
        print_result(log_file, 'Valid Recall', recall, Ks)
        print_result(log_file, 'Valid NDCG', ndcg, Ks)
        curr_score = recall[1]
        if curr_score > best_res['recall']:
            best_res['recall'] = curr_score
            best_res['epoch'] = epoch
            bad_counter = 0
            torch.save(model.state_dict(), os.path.join(log_path, "model_best.pth"))
            file_write(log_file, f"[BEST] New Best Epoch: {epoch} (Recall@20: {curr_score:.4f}). Model saved.")
        else:
            bad_counter += 1
            file_write(log_file, f"Patience: {bad_counter}/{opt.patience}")
        if bad_counter >= opt.patience:
            file_write(log_file, "Early stopping triggered...")
            break
    cprint(log_file, "Loading best model for final testing...")
    model.load_state_dict(torch.load(os.path.join(log_path, "model_best.pth")))
    recall, mrr, ndcg, auc = test(opt, model, test_loader, device, Ks, Ks_auc)
    cprint(log_file, "Final Test Results (CoDeR++):")
    print_result(log_file, 'Recall', recall, Ks)
    print_result(log_file, 'MRR', mrr, Ks)
    print_result(log_file, 'NDCG', ndcg, Ks)
    print_auc_result(log_file, 'AUC', auc, Ks_auc)
    file_write(log_file, f"Best validation epoch: {best_res['epoch']}")


if __name__ == '__main__':
    main()
