# import pickle
# import random
# from data.dataloader import CustomDataset


# def load_challenge_2019(training_ratio=0.8):
#     x, y, static, mask, name = pickle.load(open('./data/Challenge2019/data_normalized.pkl', 'rb'))
#     patient_index = list(range(len(x)))
#     random.shuffle(patient_index)
#     x_len = [len(i) for i in x]

#     train_num = int(len(x) * training_ratio)
#     val_num = int(len(x) * ((1 - training_ratio) / 2))
#     test_num = len(x) - train_num - val_num

#     train_data = []
#     for idx in patient_index[: train_num]:
#         train_data.append({
#             "x": x[idx],
#             "labels": y[idx],
#             "lens": x_len[idx],
#             "mask": mask[idx],
#             "static": static[idx]
#         })
        
#     val_data = []
#     for idx in patient_index[train_num : train_num + val_num]:
#         val_data.append({
#             "x": x[idx],
#             "labels": y[idx],
#             "lens": x_len[idx],
#             "mask": mask[idx],
#             "static": static[idx]
#         })
        
#     test_data = []
#     for idx in patient_index[train_num + val_num :]:
#         test_data.append({
#             "x": x[idx],
#             "labels": y[idx],
#             "lens": x_len[idx],
#             "mask": mask[idx],
#             "static": static[idx]
#         })

#     return CustomDataset(train_data), CustomDataset(val_data), CustomDataset(test_data)


import pickle
import random
from data.dataloader import CustomDataset


def split_indices(indices, training_ratio=0.8):
    """8:1:1 split helper"""
    random.shuffle(indices)
    n = len(indices)

    train_num = int(n * training_ratio)
    val_num = int(n * ((1 - training_ratio) / 2))
    test_num = n - train_num - val_num

    train_idx = indices[:train_num]
    val_idx = indices[train_num:train_num + val_num]
    test_idx = indices[train_num + val_num:]

    return train_idx, val_idx, test_idx


def build_dataset(x, y, static, mask, indices):
    """Pack data into CustomDataset"""
    data = []
    for idx in indices:
        data.append({
            "x": x[idx],
            "labels_y": y[idx],      # ***你 evaluate_risk 用 labels_y，所以我統一這樣命名***
            "labels": y[idx],
            "lens": len(x[idx]),
            "mask": mask[idx],
            "static": static[idx],
        })
    return CustomDataset(data)


# ==========================================================
# ① Pretrain 版本（只用負樣本 → 8:1:1）
# ==========================================================
def load_challenge_2019_pretrain(training_ratio=0.8):
    """
    Pretrain 版本：只使用負樣本（y=0）
    train/valid/test = 8:1:1
    """
    x, y, static, mask, name = pickle.load(open('./data/Challenge2019/data_normalized.pkl', 'rb'))

    # 只取負樣本 indices
    neg_indices = [i for i, label in enumerate(y) if label == 0]

    # 8:1:1 split（全部都負樣本）
    train_idx, val_idx, test_idx = split_indices(neg_indices, training_ratio)

    # 建立 dataset
    return (
        build_dataset(x, y, static, mask, train_idx),
        build_dataset(x, y, static, mask, val_idx),
        build_dataset(x, y, static, mask, test_idx)
    )


# ==========================================================
# ② Fine-tune 版本（原始 positive + negative → 8:1:1）
# ==========================================================
def load_challenge_2019(training_ratio=0.8):
    """
    Fine-tune 版本：使用全部資料（y=0 + y=1）
    train/valid/test = 8:1:1
    """
    x, y, static, mask, name = pickle.load(open('./data/Challenge2019/data_normalized.pkl', 'rb'))

    all_indices = list(range(len(x)))

    # 8:1:1 split（正+負自然佔比）
    train_idx, val_idx, test_idx = split_indices(all_indices, training_ratio)

    return (
        build_dataset(x, y, static, mask, train_idx),
        build_dataset(x, y, static, mask, val_idx),
        build_dataset(x, y, static, mask, test_idx)
    )
