import argparse
import copy
import json
import os
import logging
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler, SequentialSampler
import numpy as np
from torch import nn

from data.challenge2012 import load_challenge_2012
from data.challenge2019 import load_challenge_2019
from data.mimiciii import load_mimic_iii_mortality, load_mimic_iii_phenotyping, load_mimic_iii_decompensation, load_mimic_iii_lengthofstay
from data.dataloader import collate_fn
from models.smart import Encoder, EmbeddingDecoder
from utils.utils import set_seed, distributed_init, init_logging


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        """
        device = features.device

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        # Normalize features
        features = F.normalize(features, dim=2)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)

        # compute mean of log-likelihood over positive
        # mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        # Avoid nan if no positives
        sum_mask = mask.sum(1)
        sum_mask[sum_mask == 0] = 1
        mean_log_prob_pos = (mask * log_prob).sum(1) / sum_mask

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


def random_masking(x, original_mask, min_mask_ratio, max_mask_ratio):
    """
    Perform per-sample random masking.
    """
    N, L, V = x.shape  # batch, length, var

    # Calculate mask ratios and lengths to keep for each sample in the batch
    mask_ratios = torch.rand(N, device=x.device) * \
        (max_mask_ratio - min_mask_ratio) + min_mask_ratio
    
    mask = torch.rand_like(x) < mask_ratios.view(-1, 1, 1)
    x = x * (~mask)  # True for reconstruction, False for original
    return x, original_mask * (~mask),  original_mask * mask


def test(args, checkpoint_path, test_dataloader, criterion):
    checkpoint = torch.load(os.path.join(args.save_dir, checkpoint_path))
    save_epoch = checkpoint['epoch']
    log(logger, "last saved model is in epoch {}".format(save_epoch))
    encoder.load_state_dict(checkpoint['encoder'])
    predictor.load_state_dict(checkpoint['predictor'])
    # target_encoder.load_state_dict(checkpoint['target_encoder']) # Not used
    encoder.eval()
    predictor.eval()
    # target_encoder.eval()
    test_loss = 0
    with torch.no_grad():
        for batch in test_dataloader:
            for key in batch:
                batch[key] = batch[key].cuda()
            # with torch.no_grad():
            #     h = target_encoder(**batch)
            # batch['labels'] = batch['x'] # Removed: We want actual labels
            
            # Keep random masking for augmentation/robustness, but don't use the mask for loss
            # Randomly augment roughly 50% of the batch
            do_augment = torch.rand(batch['x'].shape[0], device=batch['x'].device) > 0.5
            x_aug = batch['x'].clone()
            mask_aug = batch['mask'].clone()
            
            if do_augment.any():
                x_to_aug = x_aug[do_augment]
                mask_to_aug = mask_aug[do_augment]
                x_masked, mask_masked, _ = random_masking(x_to_aug, mask_to_aug, args.min_mask_ratio, args.max_mask_ratio)
                x_aug[do_augment] = x_masked
                mask_aug[do_augment] = mask_masked
            
            batch['x'] = x_aug
            batch['mask'] = mask_aug
            
            z = encoder(**batch)
            
            # Extract [CLS] token (Time index 0) and use it for SupCon
            z_cls = z[:, :, 0, :] # [B, Variables, D]
            z_cls = predictor(z_cls) # [B, Variables, D] - Using predictor as projection head
            z_flat = z_cls.reshape(z_cls.shape[0], -1) # [B, Variables*D]
            
            # SupCon expects [B, n_views, D]
            features = z_flat.unsqueeze(1)
            
            test_loss += criterion(features, batch['labels']).item() * batch['x'].shape[0]
            
    log(logger, 'Test Loss %.4f' % (test_loss / len(test_dataset)))


def smooth_l1_loss(pred, target, pad_mask, beta=1.0):
    diff = torch.abs(pred - target)
    cond = diff < beta
    loss = torch.where(cond, 0.5 * diff ** 2 / beta, diff - 0.5 * beta)
    combined_mask = pad_mask.bool()
    loss = (loss * combined_mask).sum() / (combined_mask.sum() + 1e-6)
    return loss


def log(logger, msg):
    if logger is not None:
        logger.info(msg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mimic_decompensation', choices=['c12', 'c19', 'mimic_mortality', 
                            'mimic_phenotyping', 'mimic_decompensation', 'mimic_lengthofstay'])
    parser.add_argument('--data_dropout', type=float, default=0.)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--d_model', type=int, default=32)
    parser.add_argument('--seed', type=int, default=3407) 
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--save_model', type=bool, default=True)
    parser.add_argument('--save_dir', type=str, default='./export/')
    parser.add_argument('--local-rank', type=int, default=1)
    parser.add_argument('--min_mask_ratio', type=float, default=0.)
    parser.add_argument('--max_mask_ratio', type=float, default=0.75)
    parser.add_argument('--e_layers', type=int, default=2)
    parser.add_argument('--n_heads', type=int, default=4)
    args = parser.parse_args()
    args.save_dir = os.path.join(args.save_dir, args.dataset, 'smart_supcon') # Changed dir name
    if args.local_rank == 0 and args.save_model and not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    if args.local_rank == 0:
        logger = logging.getLogger()
        init_logging(logger, args.save_dir if args.save_model else None)
    else:
        logger = None
    log(logger, json.dumps(vars(args), indent=4))
    set_seed(args.seed)
    distributed_init(args)

    if args.dataset == 'c12':
        args.input_dim = 37
        args.demo_dim = 4
        args.num_class = 2
        args.max_len = 48
        train_dataset, val_dataset, test_dataset = load_challenge_2012()
    elif args.dataset == 'c19':
        args.input_dim = 34
        args.demo_dim = 5
        args.num_class = 2
        args.max_len = 60
        train_dataset, val_dataset, test_dataset = load_challenge_2019()
    elif args.dataset == 'mimic_mortality':
        args.input_dim = 17
        args.demo_dim = 0
        args.num_class = 2
        args.max_len = 48
        train_dataset, val_dataset, test_dataset = load_mimic_iii_mortality()
    elif args.dataset == 'mimic_phenotyping':
        args.input_dim = 17
        args.demo_dim = 0
        args.num_class = 25
        args.max_len = 60
        train_dataset, val_dataset, test_dataset = load_mimic_iii_phenotyping()
    elif args.dataset == 'mimic_decompensation':
        args.input_dim = 17
        args.demo_dim = 0
        args.num_class = 2
        args.max_len = 24
        train_dataset, val_dataset, test_dataset = load_mimic_iii_decompensation()
    elif args.dataset == 'mimic_lengthofstay':
        args.input_dim = 17
        args.demo_dim = 0
        args.num_class = 10
        args.max_len = 24
        args.max_mask_ratio = 0.75
        train_dataset, val_dataset, test_dataset = load_mimic_iii_lengthofstay()
    else:
        raise Exception("Dataset not exist!")
    if args.data_dropout > 0:
        train_dataset.dropout_data(args.data_dropout)
        val_dataset.dropout_data(args.data_dropout)
        test_dataset.dropout_data(args.data_dropout)
    log(logger, 'Dataset Loaded.')
    if args.dataset != 'all':
        if args.distributed:
            train_sampler = DistributedSampler(train_dataset, num_replicas=args.world_size, rank=args.rank, shuffle=True, drop_last=True)
            val_sampler = SequentialSampler(val_dataset)
            test_sampler = SequentialSampler(test_dataset)
        else:
            train_sampler = RandomSampler(train_dataset)
            val_sampler = SequentialSampler(val_dataset)
            test_sampler = SequentialSampler(test_dataset)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, collate_fn=collate_fn)
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler, collate_fn=collate_fn)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, sampler=test_sampler, collate_fn=collate_fn)

    encoder = Encoder(args).cuda()
    predictor = EmbeddingDecoder(args).cuda() # Using as projection head
    # target_encoder = copy.deepcopy(encoder) # Not used for SupCon in this setup
    
    if args.distributed:
        encoder = torch.nn.parallel.DistributedDataParallel(encoder, static_graph=True, device_ids=[args.gpu], output_device=args.local_rank, find_unused_parameters=True)
        predictor = torch.nn.parallel.DistributedDataParallel(predictor, static_graph=True, device_ids=[args.gpu], output_device=args.local_rank, find_unused_parameters=True)
        # target_encoder = torch.nn.parallel.DistributedDataParallel(target_encoder, device_ids=[args.gpu], output_device=args.local_rank, find_unused_parameters=True)
    # for p in target_encoder.parameters():
    #     p.requires_grad = False
        
    ema = [0.996, 1]
    ipe = len(train_dataloader)
    ipe_scale = 1.0
    momentum_scheduler = (ema[0] + i*(ema[1]-ema[0])/(ipe*args.epochs*ipe_scale)
                          for i in range(int(ipe*args.epochs*ipe_scale)+1))
    
    param_groups = [
        {
            'params': encoder.parameters(),
        }, 
        {
            'params': predictor.parameters()
        }
    ]
    optimizer = torch.optim.Adam(param_groups, args.lr)
    criterion = SupConLoss() # Changed loss function

    best_auc = 0
    best_prc = 0
    best_mse = 1e9 # Initialize with large number for loss comparison
    for i in range(1, args.epochs + 1):
        train_loss = 0
        val_loss = 0
        encoder.train()
        predictor.train()
        # target_encoder.train()
        for step, batch in enumerate(train_dataloader, 1):
            for key in batch:
                batch[key] = batch[key].cuda()
            # with torch.no_grad():
            #     h = target_encoder(**batch)
            # batch['labels'] = batch['x'] # Removed
            
            # Randomly augment roughly 50% of the batch, keep others original
            # Generate a random mask for deciding which samples to augment
            do_augment = torch.rand(batch['x'].shape[0], device=batch['x'].device) > 0.5
            
            # Create copies to avoid modifying original batch in place for unaugmented ones
            x_aug = batch['x'].clone()
            mask_aug = batch['mask'].clone()
            
            # Apply masking only to the selected samples
            if do_augment.any():
                # Extract samples to augment
                x_to_aug = x_aug[do_augment]
                mask_to_aug = mask_aug[do_augment]
                
                # Apply random masking
                x_masked, mask_masked, _ = random_masking(x_to_aug, mask_to_aug, args.min_mask_ratio, args.max_mask_ratio)
                
                # Put back into the batch tensors
                x_aug[do_augment] = x_masked
                mask_aug[do_augment] = mask_masked
            
            batch['x'] = x_aug
            batch['mask'] = mask_aug
            
            z = encoder(**batch)
            
            # Extract [CLS] token (Time index 0)
            z_cls = z[:, :, 0, :] # [B, Variables, D]
            z_cls = predictor(z_cls) # [B, Variables, D]
            z_flat = z_cls.reshape(z_cls.shape[0], -1) # [B, Variables*D]
            
            # SupCon features: [B, n_views, D]
            features = z_flat.unsqueeze(1)
            
            # Loss using labels
            loss = criterion(features, batch['labels'])
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Momentum update skipped as target_encoder is not used
            # with torch.no_grad():
            #     m = next(momentum_scheduler)
            #     for param_q, param_k in zip(encoder.parameters(), target_encoder.parameters()):
            #         param_k.data.mul_(m).add_((1.-m) * param_q.detach().data)
            train_loss += loss.item() * batch['x'].shape[0]

        encoder.eval()
        predictor.eval()
        # target_encoder.eval()
        with torch.no_grad():
            for batch in val_dataloader:
                for key in batch:
                    batch[key] = batch[key].cuda()
                # with torch.no_grad():
                #     h = target_encoder(**batch)
                # batch['labels'] = batch['x']
                
                # Randomly augment roughly 50% of the batch for validation as well
                do_augment = torch.rand(batch['x'].shape[0], device=batch['x'].device) > 0.5
                x_aug = batch['x'].clone()
                mask_aug = batch['mask'].clone()
                
                if do_augment.any():
                    x_to_aug = x_aug[do_augment]
                    mask_to_aug = mask_aug[do_augment]
                    x_masked, mask_masked, _ = random_masking(x_to_aug, mask_to_aug, args.min_mask_ratio, args.max_mask_ratio)
                    x_aug[do_augment] = x_masked
                    mask_aug[do_augment] = mask_masked
                
                batch['x'] = x_aug
                batch['mask'] = mask_aug
                
                z = encoder(**batch)
                
                z_cls = z[:, :, 0, :] 
                z_cls = predictor(z_cls)
                z_flat = z_cls.reshape(z_cls.shape[0], -1)
                
                features = z_flat.unsqueeze(1)
                
                val_loss += criterion(features, batch['labels']).item() * batch['x'].shape[0]
                
        log(logger, 'Epoch %d: Train Loss %.4f, Valid Loss %.4f' % (i, train_loss / len(train_dataset) * args.world_size, val_loss / len(val_dataset)))
        cur_loss = val_loss / len(val_dataset)
        if cur_loss < best_mse: # Reusing best_mse variable name for best_loss
            best_mse = cur_loss
            if args.local_rank == 0:
                state = {
                    'encoder': encoder.state_dict(),
                    'predictor': predictor.state_dict(),
                    # 'target_encoder': target_encoder.state_dict(),
                    'epoch': i
                }
                log(logger, '----- Save best model - Loss: %.4f -----' % cur_loss)
                torch.save(state, os.path.join(args.save_dir, 'checkpoint-supcon.pth'))
        if args.distributed:
            dist.barrier()

    if args.distributed:
        dist.barrier()
    test(args, 'checkpoint-supcon.pth', test_dataloader, criterion)
