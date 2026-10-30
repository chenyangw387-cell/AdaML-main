import os
import time
import torch
import psutil
import random
import numpy as np
from functools import wraps


def seed_torch(log_path_txt, seed=None):
    if seed is None:
        seed = int(time.time())
    file_write(log_path_txt, f'************ seed ***********: {seed}')
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def timefn(fn):
    @wraps(fn)
    def measure_time(*args, **kwargs):
        t1 = time.time()
        result = fn(*args, **kwargs)
        t2 = time.time()
        print(f"@timefn: {fn.__name__} took {t2 - t1: .5f} s")
        return result

    return measure_time


def show_memory_info(hint):
    pid = os.getpid()
    p = psutil.Process(pid)
    info = p.memory_full_info()
    memory = info.uss / 1024. / 1024
    print("{} memory used: {} MB".format(hint, memory))


def print_result(log_file, title, metric, Ks=[10, 20, 50]):
    assert len(metric) == len(Ks)
    title_list = [f"{title}@{k}" for k in Ks]
    title_str = "\t".join(title_list)
    metric_list = [f"{m:.4f}" for m in metric]
    result_str = "\t".join(metric_list)
    file_write(log_file, title_str)
    file_write(log_file, result_str)
    return


def print_auc_result(log_file, title, metric, Ks_auc=[50, 100, 200, 500]):
    assert len(metric) == len(Ks_auc)
    title_list = [f"{title}@{k}" for k in Ks_auc]
    title_str = "\t".join(title_list)
    metric_list = [f"{m:.4f}" for m in metric]
    result_str = "\t".join(metric_list)
    file_write(log_file, title_str)
    file_write(log_file, result_str)
    return


def cprint(log_file, words: str):
    print(f"\033[31;1m{words}\033[0m")
    file_write(log_file, words, whether_print=False)


import os


def file_write(log_file, s, whether_print=True):
    if whether_print:
        print(s)
    try:
        log_dir = os.path.dirname(log_file)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(s + '\n')
    except Exception as e:
        print(f"Error writing to log file {log_file}: {e}")


def get_items_byCategory(target, item_catgy, sample_strategy=None):
    n_items = item_catgy.shape[0]
    if sample_strategy != 'category':
        return torch.arange(1, n_items).tolist()
    target_category = item_catgy[target]
    item_catgy = item_catgy.cpu().numpy()
    items_index = np.argwhere(item_catgy == target_category.cpu().numpy())
    items_index = items_index.reshape(1, -1).tolist()[0]
    return items_index


def normalize_array(arr, type):
    arr = np.array(arr, dtype=np.float32)
    min_val = np.min(arr)
    max_val = np.max(arr)
    mean_val = np.mean(arr)
    std_val = np.std(arr)
    arr = arr.astype(float)
    exp_arr = np.exp(arr)
    sum_arr = np.sum(exp_arr)
    if type == 'max_min':
        norm_arr = (arr - min_val) / (max_val - min_val)
    if type == 'z_score':
        norm_arr = (arr - mean_val) / std_val
    if type == 'softmax':
        norm_arr = exp_arr / sum_arr
    if type == 'sigmoid':
        norm_arr = 1 / (1 + np.exp(-arr))
    return norm_arr


def normalize_array_2D(arr, type):
    min_val = np.min(arr)
    max_val = np.max(arr)
    mean_val = np.mean(arr)
    std_val = np.std(arr)
    if type == 'max_min':
        norm_arr = (arr - min_val) / (max_val - min_val)
    if type == 'z_score':
        norm_arr = (arr - mean_val) / std_val
    if type == 'softmax':
        norm_arr = np.apply_along_axis(lambda x: np.exp(x) / np.sum(np.exp(x)), 1, arr)
    return norm_arr
