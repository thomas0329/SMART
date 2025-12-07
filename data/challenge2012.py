# import pickle
# import random
# from data.dataloader import CustomDataset


# def load_challenge_2012(training_ratio=0.8):
#     x, y, static, mask, name = pickle.load(open('./data/Challenge2012/data_normalized.pkl', 'rb'))
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


# ============================================================
#  A. PRETRAIN DATASET (NEGATIVE ONLY, 8:1:1)
# ============================================================
def load_c12_pretrain(ratio=0.8):
    """
    Pretraining dataset:
    - Only negative samples (label == 0)
    - Split into 8:1:1
    """

    x, y, static, mask, name = pickle.load(
        open('./data/Challenge2012/data_normalized.pkl', 'rb')
    )

    neg_idx = [i for i, label in enumerate(y) if label == 0]
    random.shuffle(neg_idx)

    n = len(neg_idx)

    train_idx = neg_idx[: int(0.8 * n)]
    valid_idx = neg_idx[int(0.8 * n) : int(0.9 * n)]
    test_idx  = neg_idx[int(0.9 * n) : ]

    def build(indices):
        data = []
        for idx in indices:
            data.append({
                "x": x[idx],
                "labels": y[idx],     # labels still kept for convenience
                "lens": len(x[idx]),
                "mask": mask[idx],
                "static": static[idx]
            })
        return CustomDataset(data)

    return build(train_idx), build(valid_idx), build(test_idx)



# ============================================================
#  B. FINETUNE DATASET (NEG + POS, NORMAL 8:1:1)
# ============================================================
def load_challenge_2012(ratio=0.8):
    """
    Original SMART-style split:
    - Use all data (positive + negative)
    - Split into 8:1:1
    """
    x, y, static, mask, name = pickle.load(
        open('./data/Challenge2012/data_normalized.pkl', 'rb')
    )

    total_idx = list(range(len(x)))
    random.shuffle(total_idx)

    n = len(total_idx)

    train_idx = total_idx[: int(0.8 * n)]
    valid_idx = total_idx[int(0.8 * n) : int(0.9 * n)]
    test_idx  = total_idx[int(0.9 * n) : ]

    def build(indices):
        data = []
        for idx in indices:
            data.append({
                "x": x[idx],
                "labels": y[idx],
                "lens": len(x[idx]),
                "mask": mask[idx],
                "static": static[idx]
            })
        return CustomDataset(data)

    return build(train_idx), build(valid_idx), build(test_idx)
