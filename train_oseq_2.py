import torch
import torch.nn as nn
from Qdmodel_oseq import QuantumDiffusionModel_oseq
from Qdmodel_oseq_rope import QuantumDiffusionModel_oseq_rope
from Qdmodel_oseq_rope_2d import QuantumDiffusionModel_oseq_rope_2d
from Qdmodel_oseq_rope_2d_relative_b import (
    QuantumDiffusionModel_oseq_rope_2d as QuantumDiffusionModel_oseq_rope_2d_relative_b,
)
from Qdmodel_oseq_rope_2d_g import (
    QuantumDiffusionModel_oseq_rope_2d as QuantumDiffusionModel_oseq_rope_2d_g,
)
from Qdmodel_oseq_rope_gnn import QuantumDiffusionModel_oseq_rope_gnn
from Qdmodel_oseq_rope_sharepos import QuantumDiffusionModel_oseq_rope_sharepos
import json
from torch.utils.data import Dataset, DataLoader, Sampler
from tqdm import tqdm
import argparse
import random
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Numerical Diffusion Model Training with LR Scheduler (oseq layout)")

    # 数据相关
    parser.add_argument(
        '--data_path',
        type=str,
        nargs='+',
        default=['/home/fyp26lyh/QuantumLLADA/data/TFI_train_data_nseq.json'],
        help=(
            '训练 JSON 路径；1D 多尺寸时与 --qubits 顺序一致；'
            '2D 模式直接从各文件的 lattice_shape 读取尺寸'
        ),
    )
    parser.add_argument('--eval_loss', action='store_true',
                        help='是否在训练时于验证集上评估 loss')
    parser.add_argument('--qubits', type=int, nargs='+', default=[10],
                        help='1D 量子比特数列表；2D 模式忽略该参数')

    parser.add_argument('--head_num', type=int, default=8, help='注意力头数（oseq 原版默认 8）')
    parser.add_argument('--hidden_dim', type=int, default=128, help='隐藏维度')
    parser.add_argument('--layer_num', type=int, default=4, help='Transformer 层数（oseq 原版默认 4）')

    # RoPE models, including the explicit 2D variant.
    parser.add_argument(
        '--max_seq_len',
        type=int,
        default=4096,
        help='RoPE max_position_embeddings',
    )
    parser.add_argument(
        '--max_N',
        type=int,
        default=None,
        help='预留；与 Qdmodel_nseq_rope_oseq 一致时一般取 max(qubits)',
    )
    parser.add_argument(
        '--rope_scaling_type',
        type=str,
        default='none',
        choices=['none', 'linear', 'dynamic', 'ntk'],
        help='RoPE 外推：none | linear | dynamic/ntk',
    )
    parser.add_argument('--rope_scaling_factor', type=float, default=1.0, help='linear/dynamic 的 scaling_factor')
    parser.add_argument('--rope_theta', type=float, default=10000.0, help='RoPE theta')

    # 混合局域/全局注意力（仅 model_type=oseq_rope_gnn）
    parser.add_argument(
        '--num_global_heads',
        type=int,
        default=1,
        help='全注意力头数；其余为局域窗口头（|i-j|<=local_window_radius）',
    )
    parser.add_argument(
        '--local_window_radius',
        type=int,
        default=2,
        help='局域头在序列下标上的半窗半径 r（|i-j|<=r）',
    )

    parser.add_argument('--lr', type=float, default=1e-5, help='初始学习率')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--eps', type=float, default=1e-3, help='掩码概率下界偏移（constant 或 linear 起点）')
    parser.add_argument(
        '--eps_schedule',
        type=str,
        default='constant',
        choices=['constant', 'linear'],
        help='eps 是否随 epoch 线性变化',
    )
    parser.add_argument('--eps_end', type=float, default=1e-2, help='eps_schedule=linear 时末 epoch 的 eps')
    parser.add_argument(
        '--pair_loss_weight',
        type=float,
        default=0.0,
        help='Weight for pair-aware correlation loss. 0 disables it.',
    )
    parser.add_argument(
        '--pair_loss_distances',
        type=int,
        nargs='+',
        default=[1, 2],
        help='PBC distances used by pair-aware loss, e.g. --pair_loss_distances 1 2.',
    )
    parser.add_argument('--batch_size', type=int, default=256, help='基准 batch size（多尺寸时按长度缩放）')
    parser.add_argument('--patience', type=int, default=120, help='早停耐心（epoch）')
    parser.add_argument(
        '--model_type',
        type=str,
        default='oseq',
        choices=[
            'oseq',
            'oseq_rope',
            'oseq_rope_2d',
            'oseq_rope_2d_relative_b',
            'oseq_rope_2d_g',
            'oseq_rope_sharepos',
            'oseq_rope_gnn',
        ],
        help=(
            'oseq=原版 MHA；oseq_rope=交错+RoPE；'
            'oseq_rope_2d=显式二维网格+2D RoPE；'
            'oseq_rope_2d_relative_b=2D RoPE+周期局部拓扑 bias；'
            'oseq_rope_2d_g=将 g embedding 广播到全部 P/b token；'
            'oseq_rope_sharepos=交错+RoPE+pair 共享 id；'
            'oseq_rope_gnn=混合局域/全局头'
        ),
    )

    parser.add_argument(
        '--lr_scheduler',
        type=str,
        default='constant',
        choices=['constant', 'cosine', 'step', 'linear', 'cosine_warmup'],
        help='学习率调度',
    )
    parser.add_argument('--step_size', type=int, default=5, help='StepLR 步长')
    parser.add_argument('--gamma', type=float, default=0.5, help='StepLR/Linear 衰减因子')

    parser.add_argument('--model_save_path', type=str,
                        default='/home/fyp26lyh/QuantumLLADA/DshadowGPT/model_train/Quantum_diffusion_model.pth',
                        help='模型保存路径')
    parser.add_argument(
        '--pre_final_checkpoint_count',
        type=int,
        default=0,
        help='在 final 之前额外保存的 checkpoint 数量 n；0 表示不保存',
    )
    parser.add_argument(
        '--pre_final_checkpoint_interval',
        type=int,
        default=1,
        help='final 前各 checkpoint 的 epoch 间隔 k',
    )
    parser.add_argument('--loss_save_path', type=str,
                        default='/home/fyp26lyh/QuantumLLADA/DshadowGPT/model_train/loss_fig/loss.json',
                        help='训练日志 JSON')
    parser.add_argument('--checkpoint_path', type=str, default='',
                        help='Path to a checkpoint or model state_dict to load before training')
    parser.add_argument('--model_load_path', type=str, default='',
                        help='Alias of --checkpoint_path, kept for compatibility with other train scripts')
    parser.add_argument('--resume_checkpoint', action='store_true',
                        help='Restore optimizer/scheduler/epoch metadata when present in the checkpoint')
    parser.add_argument('--checkpoint_non_strict', action='store_true',
                        help='Load checkpoint weights with strict=False')

    parser.add_argument('--print_grad_per_step', action='store_true', help='每步打印梯度范数')

    return parser.parse_args()


MASK_TOKEN_ID = -1.0
MODEL_TYPES_2D = (
    'oseq_rope_2d',
    'oseq_rope_2d_relative_b',
    'oseq_rope_2d_g',
)


def get_scheduler(optimizer, args, total_steps):
    if args.lr_scheduler == 'constant':
        return None
    elif args.lr_scheduler == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=5e-6)
    elif args.lr_scheduler == 'step':
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    elif args.lr_scheduler == 'linear':
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: max(0.0, 1.0 - step / total_steps)
        )
    elif args.lr_scheduler == 'cosine_warmup':
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=2, eta_min=0.000001)
    else:
        raise ValueError(f"Unknown scheduler: {args.lr_scheduler}")


def get_eps_for_epoch(args, epoch: int) -> float:
    if args.eps_schedule == 'constant':
        return float(args.eps)
    if args.eps_schedule == 'linear':
        e0, e1 = float(args.eps), float(args.eps_end)
        if args.epochs <= 1:
            return e0
        t = epoch / (args.epochs - 1)
        return e0 + (e1 - e0) * t
    raise ValueError(f"Unknown eps_schedule: {args.eps_schedule}")


def safe_torch_load(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _looks_like_state_dict(obj):
    return isinstance(obj, dict) and len(obj) > 0 and all(
        isinstance(v, torch.Tensor) for v in obj.values()
    )


def extract_model_state_dict(checkpoint):
    if isinstance(checkpoint, nn.Module):
        return checkpoint.state_dict()
    if not isinstance(checkpoint, dict):
        return checkpoint

    for key in ('model_state_dict', 'state_dict', 'model', 'net', 'network'):
        value = checkpoint.get(key)
        if isinstance(value, nn.Module):
            return value.state_dict()
        if isinstance(value, dict):
            return value

    if _looks_like_state_dict(checkpoint):
        return checkpoint

    expected = "model_state_dict/state_dict/model/net/network"
    raise ValueError(f"Checkpoint does not contain a model state_dict ({expected}).")


def strip_module_prefix(state_dict):
    if not isinstance(state_dict, dict) or len(state_dict) == 0:
        return state_dict
    if all(isinstance(k, str) and k.startswith('module.') for k in state_dict.keys()):
        return {k[len('module.'):]: v for k, v in state_dict.items()}
    return state_dict


def resolve_checkpoint_path(args):
    checkpoint_path = args.checkpoint_path.strip()
    model_load_path = args.model_load_path.strip()
    if checkpoint_path and model_load_path and checkpoint_path != model_load_path:
        raise ValueError("--checkpoint_path and --model_load_path point to different files.")
    return checkpoint_path or model_load_path


def load_training_checkpoint(args, model, optimizer, scheduler, map_location='cuda'):
    checkpoint_path = resolve_checkpoint_path(args)
    if not checkpoint_path:
        return {}
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = safe_torch_load(checkpoint_path, map_location=map_location)
    state_dict = strip_module_prefix(extract_model_state_dict(checkpoint))
    incompatible = model.load_state_dict(state_dict, strict=not args.checkpoint_non_strict)

    if args.checkpoint_non_strict:
        if incompatible.missing_keys:
            print(f"Missing keys when loading checkpoint: {incompatible.missing_keys}")
        if incompatible.unexpected_keys:
            print(f"Unexpected keys when loading checkpoint: {incompatible.unexpected_keys}")

    restored = {}
    if not args.resume_checkpoint or not isinstance(checkpoint, dict):
        print("Checkpoint weights loaded.")
        return restored

    optimizer_state = checkpoint.get('optimizer_state_dict', checkpoint.get('optimizer'))
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
        print("Optimizer state restored.")

    scheduler_state = checkpoint.get('scheduler_state_dict', checkpoint.get('scheduler'))
    if scheduler is not None and scheduler_state is not None:
        scheduler.load_state_dict(scheduler_state)
        print("Scheduler state restored.")

    if 'start_epoch' in checkpoint:
        restored['start_epoch'] = int(checkpoint['start_epoch'])
    elif 'epoch' in checkpoint:
        restored['start_epoch'] = int(checkpoint['epoch']) + 1

    for key in ('grad_norms', 'lrs', 'epoch_losses', 'test_losses', 'epoch_ce_losses', 'epoch_pair_losses'):
        value = checkpoint.get(key)
        if isinstance(value, list):
            restored[key] = value

    for key in ('best_eval_loss', 'best_test_loss'):
        if key in checkpoint and checkpoint[key] is not None:
            restored['best_eval_loss'] = float(checkpoint[key])
            break

    if 'patience_counter' in checkpoint:
        restored['patience_counter'] = int(checkpoint['patience_counter'])

    print(f"Checkpoint loaded. Resuming from epoch {restored.get('start_epoch', 0) + 1}.")
    return restored


def forward_process(batch, qubit, eps=1e-3):
    """
    oseq 布局：[g, P1,b1, P2,b2, ...]；仅对 b 位置（偶数索引 2,4,...）加掩码。
    batch: [B, 1+2*qubit]
    """
    x = batch.float()
    batch_size, L = x.shape
    device = x.device
    assert L == 1 + 2 * qubit, f"期望长度 {1 + 2 * qubit}, 得到 {L}"

    b = x[:, 2:L:2]

    t = torch.rand(batch_size, device=device)
    p_mask = (1 - eps) * t + eps

    p_mask_expanded = p_mask.unsqueeze(1).expand(-1, qubit)
    rand = torch.rand((batch_size, qubit), device=device)
    masked_indices = rand < p_mask_expanded
    b_masked = torch.where(masked_indices, b.new_tensor(-1.0), b)

    noisy_batch = x.clone()
    noisy_batch[:, 2:L:2] = b_masked

    return noisy_batch, masked_indices, p_mask


def forward_process_2d(outcome_grid, eps=1e-3):
    """Mask only outcome tokens in a ``[B, Ly, Lx]`` grid."""
    if outcome_grid.ndim != 3:
        raise ValueError("outcome_grid must have shape [B, Ly, Lx]")

    outcomes = outcome_grid.float()
    batch_size, ly, lx = outcomes.shape
    device = outcomes.device

    t = torch.rand(batch_size, device=device)
    p_mask = (1 - eps) * t + eps
    mask_probability = p_mask.reshape(batch_size, 1, 1).expand(-1, ly, lx)
    masked_indices = torch.rand(
        (batch_size, ly, lx), device=device
    ) < mask_probability
    noisy_outcomes = torch.where(
        masked_indices,
        outcomes.new_tensor(MASK_TOKEN_ID),
        outcomes,
    )
    return noisy_outcomes, masked_indices, p_mask


def pair_aware_loss(pred, target, pauli_tokens, masked_indices, distances):
    """
    Encourage predicted masked bits to reproduce same-basis pair correlations.

    pred: [B, N, 2] logits for b_i
    target: [B, N] true b_i in {0, 1}
    pauli_tokens: [B, N] P_i in {2, 3, 4}
    masked_indices: [B, N] True where b_i was masked
    """
    if not distances:
        return pred.new_tensor(0.0)

    prob_one = torch.softmax(pred, dim=-1)[..., 1]
    pred_spin = 2.0 * prob_one - 1.0
    true_spin = 2.0 * target.float() - 1.0

    sq_error_sum = pred.new_tensor(0.0)
    valid_count = pred.new_tensor(0.0)
    num_sites = pred.shape[1]

    for raw_d in distances:
        d = int(raw_d) % num_sites
        if d == 0:
            continue

        pred_pair = pred_spin * torch.roll(pred_spin, shifts=-d, dims=1)
        true_pair = true_spin * torch.roll(true_spin, shifts=-d, dims=1)

        same_basis = pauli_tokens == torch.roll(pauli_tokens, shifts=-d, dims=1)
        pair_mask = same_basis & masked_indices & torch.roll(masked_indices, shifts=-d, dims=1)
        if not torch.any(pair_mask):
            continue

        diff = pred_pair[pair_mask] - true_pair[pair_mask]
        sq_error_sum = sq_error_sum + torch.sum(diff * diff)
        valid_count = valid_count + pair_mask.sum().to(dtype=pred.dtype)

    if valid_count.item() == 0:
        return pred.new_tensor(0.0)
    return sq_error_sum / valid_count


def pair_aware_loss_2d(pred, target, pauli_grid, masked_indices, distances):
    """Pair-aware loss along both periodic lattice axes."""
    if not distances:
        return pred.new_tensor(0.0)

    prob_one = torch.softmax(pred, dim=-1)[..., 1]
    pred_spin = 2.0 * prob_one - 1.0
    true_spin = 2.0 * target.float() - 1.0
    sq_error_sum = pred.new_tensor(0.0)
    valid_count = pred.new_tensor(0.0)

    for axis in (1, 2):
        axis_size = pred_spin.shape[axis]
        for raw_d in distances:
            distance = int(raw_d) % axis_size
            if distance == 0:
                continue

            shifted_pred = torch.roll(pred_spin, shifts=-distance, dims=axis)
            shifted_true = torch.roll(true_spin, shifts=-distance, dims=axis)
            shifted_pauli = torch.roll(pauli_grid, shifts=-distance, dims=axis)
            shifted_mask = torch.roll(masked_indices, shifts=-distance, dims=axis)

            pair_mask = (
                (pauli_grid == shifted_pauli)
                & masked_indices
                & shifted_mask
            )
            if not torch.any(pair_mask):
                continue

            diff = (
                pred_spin * shifted_pred - true_spin * shifted_true
            )[pair_mask]
            sq_error_sum = sq_error_sum + torch.sum(diff * diff)
            valid_count = valid_count + pair_mask.sum().to(dtype=pred.dtype)

    if valid_count.item() == 0:
        return pred.new_tensor(0.0)
    return sq_error_sum / valid_count


def combined_oseq_loss(pred, data, masked_indices, loss_fn, pair_loss_weight, pair_loss_distances):
    B, L = data.shape
    target = data[:, 2:L:2].round().long()
    flat_logits = pred.reshape(-1, 2)
    flat_targets = target.reshape(-1)
    flat_mask = masked_indices.reshape(-1)

    masked_logits = flat_logits[flat_mask]
    masked_targets = flat_targets[flat_mask]
    ce_loss = loss_fn(masked_logits, masked_targets)

    if pair_loss_weight <= 0:
        return ce_loss, ce_loss.detach(), pred.new_tensor(0.0)

    pauli_tokens = data[:, 1:L:2].round().long()
    pair_loss = pair_aware_loss(
        pred=pred,
        target=target,
        pauli_tokens=pauli_tokens,
        masked_indices=masked_indices,
        distances=pair_loss_distances,
    )
    total_loss = ce_loss + float(pair_loss_weight) * pair_loss
    return total_loss, ce_loss.detach(), pair_loss.detach()


def combined_2d_loss(
    pred,
    pauli_grid,
    outcome_grid,
    masked_indices,
    loss_fn,
    pair_loss_weight,
    pair_loss_distances,
):
    """Loss for ``pred[B, Ly, Lx, 2]`` on masked lattice sites."""
    if pred.shape[:-1] != outcome_grid.shape or pred.shape[-1] != 2:
        raise ValueError(
            "2D prediction must have shape [B, Ly, Lx, 2] matching outcomes"
        )

    target = outcome_grid.round().long()
    ce_loss = loss_fn(pred[masked_indices], target[masked_indices])

    if pair_loss_weight <= 0:
        return ce_loss, ce_loss.detach(), pred.new_tensor(0.0)

    pair_loss = pair_aware_loss_2d(
        pred=pred,
        target=target,
        pauli_grid=pauli_grid.round().long(),
        masked_indices=masked_indices,
        distances=pair_loss_distances,
    )
    total_loss = ce_loss + float(pair_loss_weight) * pair_loss
    return total_loss, ce_loss.detach(), pair_loss.detach()


class SimpleQuantumDataset(Dataset):
    def __init__(self, json_file):
        print(f"Loading data from {json_file}...")
        with open(json_file, 'r') as f:
            self.data = json.load(f)
        print("the length of data:", len(self.data))
        print(f"Loaded {len(self.data)} samples. Shuffling...")
        random.shuffle(self.data)
        print("Data loading and shuffling completed.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32)


class MultiSizeQuantumDataset(Dataset):
    def __init__(self, json_files, sizes):
        self.samples = []
        for N, json_file in zip(sizes, json_files):
            with open(json_file, 'r') as f:
                data = json.load(f)
            for item in data:
                self.samples.append({'N': N, 'tensor': torch.tensor(item, dtype=torch.float32)})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]['tensor']


class TFI2DGridDataset(Dataset):
    """Dataset for ``tfi2d_grid_v1`` files produced by the 2D generator."""

    def __init__(self, json_files):
        self.samples = []
        self.lattice_shapes = set()

        for json_file in json_files:
            print(f"Loading 2D data from {json_file}...")
            with open(json_file, 'r', encoding='utf-8') as f:
                payload = json.load(f)

            if not isinstance(payload, dict):
                raise ValueError(
                    f"{json_file} is not a tfi2d_grid_v1 dataset object"
                )
            if payload.get('format') != 'tfi2d_grid_v1':
                raise ValueError(
                    f"{json_file} has unsupported format {payload.get('format')!r}"
                )
            if payload.get('channels') != ['pauli', 'outcome']:
                raise ValueError(
                    f"{json_file} must use channels ['pauli', 'outcome']"
                )

            raw_shape = payload.get('lattice_shape')
            if not isinstance(raw_shape, list) or len(raw_shape) != 2:
                raise ValueError(
                    f"{json_file} must contain lattice_shape=[Ly, Lx]"
                )
            lattice_shape = (int(raw_shape[0]), int(raw_shape[1]))
            if lattice_shape[0] <= 0 or lattice_shape[1] <= 0:
                raise ValueError(f"invalid lattice_shape in {json_file}")

            file_samples = payload.get('samples')
            if not isinstance(file_samples, list):
                raise ValueError(f"{json_file} must contain a samples list")

            expected_shape = lattice_shape + (2,)
            for sample_index, item in enumerate(file_samples):
                if not isinstance(item, dict) or 'g' not in item:
                    raise ValueError(
                        f"invalid sample {sample_index} in {json_file}"
                    )
                measurements = torch.as_tensor(
                    item.get('measurements'), dtype=torch.float32
                )
                if tuple(measurements.shape) != expected_shape:
                    raise ValueError(
                        f"sample {sample_index} in {json_file} has shape "
                        f"{tuple(measurements.shape)}, expected {expected_shape}"
                    )
                self.samples.append({
                    'g': float(item['g']),
                    'measurements': measurements,
                    'lattice_shape': lattice_shape,
                })
            self.lattice_shapes.add(lattice_shape)

        if not self.samples:
            raise ValueError("2D dataset contains no samples")
        random.shuffle(self.samples)
        shapes = ', '.join(f"{ly}x{lx}" for ly, lx in sorted(self.lattice_shapes))
        print(f"Loaded {len(self.samples)} 2D samples ({shapes}).")

    @property
    def max_num_sites(self):
        return max(ly * lx for ly, lx in self.lattice_shapes)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    return torch.stack(batch)


def collate_2d_fn(batch):
    lattice_shape = batch[0]['lattice_shape']
    if any(item['lattice_shape'] != lattice_shape for item in batch):
        raise ValueError("a 2D batch cannot mix different lattice shapes")

    measurements = torch.stack([item['measurements'] for item in batch])
    return {
        'condition': torch.tensor(
            [item['g'] for item in batch], dtype=torch.float32
        ),
        'pauli_grid': measurements[..., 0],
        'outcome_grid': measurements[..., 1],
        'lattice_shape': lattice_shape,
    }


class GroupedBatchSampler(Sampler):
    def __init__(self, dataset, batch_size_per_size):
        self.batch_size_per_size = batch_size_per_size
        self.groups = {}
        for idx, sample in enumerate(dataset.samples):
            N = sample['N']
            if N not in self.groups:
                self.groups[N] = []
            self.groups[N].append(idx)

    def __iter__(self):
        all_batches = []
        for N, indices in self.groups.items():
            batch_size = self.batch_size_per_size[N]
            shuffled = indices.copy()
            random.shuffle(shuffled)
            for i in range(0, len(shuffled), batch_size):
                batch = shuffled[i:i + batch_size]
                if len(batch) >= batch_size // 2:
                    all_batches.append(batch)
        random.shuffle(all_batches)
        return iter(all_batches)

    def __len__(self):
        total = 0
        for N, indices in self.groups.items():
            batch_size = self.batch_size_per_size[N]
            total += (len(indices) + batch_size - 1) // batch_size
        return total


class Grouped2DBatchSampler(Sampler):
    def __init__(self, dataset, batch_size_per_sites):
        self.batch_size_per_sites = batch_size_per_sites
        self.groups = {}
        for idx, sample in enumerate(dataset.samples):
            shape = sample['lattice_shape']
            self.groups.setdefault(shape, []).append(idx)

    def __iter__(self):
        all_batches = []
        for shape, indices in self.groups.items():
            num_sites = shape[0] * shape[1]
            batch_size = self.batch_size_per_sites[num_sites]
            shuffled = indices.copy()
            random.shuffle(shuffled)
            for start in range(0, len(shuffled), batch_size):
                batch = shuffled[start:start + batch_size]
                if batch:
                    all_batches.append(batch)
        random.shuffle(all_batches)
        return iter(all_batches)

    def __len__(self):
        total = 0
        for shape, indices in self.groups.items():
            num_sites = shape[0] * shape[1]
            batch_size = self.batch_size_per_sites[num_sites]
            total += (len(indices) + batch_size - 1) // batch_size
        return total


def get_batch_sizes(base_batch_size, sizes):
    batch_sizes = {}
    max_size = max(sizes)
    for N in sizes:
        seq_len = 1 + 2 * N
        base_seq_len = 1 + 2 * max_size
        scale = base_seq_len / seq_len
        batch_sizes[N] = int(base_batch_size * scale)
    return batch_sizes


@torch.no_grad()
def evaluate(model, dataloader_eval, loss_fn, args, eps=None):
    model.eval()
    total_loss = 0.0
    num_batches = 0
    eval_eps = float(args.eps) if eps is None else float(eps)

    for data in dataloader_eval:
        if args.model_type in MODEL_TYPES_2D:
            condition = data['condition'].to('cuda')
            pauli_grid = data['pauli_grid'].to('cuda')
            outcome_grid = data['outcome_grid'].to('cuda')
            noisy_outcomes, masked_indices, _ = forward_process_2d(
                outcome_grid=outcome_grid,
                eps=eval_eps,
            )
            pred = model.forward_2d(
                condition=condition,
                pauli_grid=pauli_grid,
                outcome_grid=noisy_outcomes,
                mask_indices=masked_indices,
            )
            loss, _, _ = combined_2d_loss(
                pred=pred,
                pauli_grid=pauli_grid,
                outcome_grid=outcome_grid,
                masked_indices=masked_indices,
                loss_fn=loss_fn,
                pair_loss_weight=args.pair_loss_weight,
                pair_loss_distances=args.pair_loss_distances,
            )
        else:
            data = data.to('cuda')
            _, length = data.shape
            qubit = int((length - 1) // 2)

            noisy_data, masked_indices, _ = forward_process(
                batch=data,
                qubit=qubit,
                eps=eval_eps,
            )
            pred = model(noisy_data.unsqueeze(-1), mask_indices=masked_indices)

            loss, _, _ = combined_oseq_loss(
                pred=pred,
                data=data,
                masked_indices=masked_indices,
                loss_fn=loss_fn,
                pair_loss_weight=args.pair_loss_weight,
                pair_loss_distances=args.pair_loss_distances,
            )

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def main():
    args = parse_args()
    if args.pre_final_checkpoint_count < 0:
        raise ValueError('--pre_final_checkpoint_count must be non-negative')
    if args.pre_final_checkpoint_interval <= 0:
        raise ValueError('--pre_final_checkpoint_interval must be positive')

    pre_final_checkpoint_epochs = {
        args.epochs - i * args.pre_final_checkpoint_interval
        for i in range(1, args.pre_final_checkpoint_count + 1)
        if args.epochs - i * args.pre_final_checkpoint_interval > 0
    }
    print("🚀 Starting training with arguments:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")

    if args.model_type in MODEL_TYPES_2D:
        print("Training with explicit 2D grids and 2D RoPE...")
        dataset_train = TFI2DGridDataset(args.data_path)
        site_counts = sorted({ly * lx for ly, lx in dataset_train.lattice_shapes})
        batch_sizes = get_batch_sizes(args.batch_size, sizes=site_counts)
        batch_sampler = Grouped2DBatchSampler(dataset_train, batch_sizes)
        dataloader_train = DataLoader(
            dataset_train,
            batch_sampler=batch_sampler,
            collate_fn=collate_2d_fn,
            num_workers=0,
        )

        dataloader_test = None
        if args.eval_loss:
            print("Loading 2D eval dataset...")
            eval_paths = [path.replace('train', 'test') for path in args.data_path]
            dataset_test = TFI2DGridDataset(eval_paths)
            eval_site_counts = sorted({
                ly * lx for ly, lx in dataset_test.lattice_shapes
            })
            all_site_counts = sorted(set(site_counts + eval_site_counts))
            eval_batch_sizes = get_batch_sizes(
                args.batch_size,
                sizes=all_site_counts,
            )
            test_batch_sampler = Grouped2DBatchSampler(
                dataset_test,
                eval_batch_sizes,
            )
            dataloader_test = DataLoader(
                dataset_test,
                batch_sampler=test_batch_sampler,
                collate_fn=collate_2d_fn,
                num_workers=0,
            )

        first_batch = next(iter(dataloader_train))
        print(
            f"First 2D batch: g={tuple(first_batch['condition'].shape)}, "
            f"P={tuple(first_batch['pauli_grid'].shape)}, "
            f"b={tuple(first_batch['outcome_grid'].shape)}"
        )
        print(f"\nTotal number of 2D training samples: {len(dataset_train):,}")

    elif args.model_type in ('oseq_rope', 'oseq_rope_sharepos', 'oseq_rope_gnn'):
        print("Training with multi-size (oseq layout + RoPE)...")
        if len(args.data_path) != len(args.qubits):
            raise ValueError("data_path 与 qubits 数量须一致。")
        sizes = args.qubits
        batch_sizes = get_batch_sizes(args.batch_size, sizes=sizes)
        dataset_train = MultiSizeQuantumDataset(args.data_path, sizes)
        batch_sampler = GroupedBatchSampler(dataset_train, batch_sizes)
        dataloader_train = DataLoader(
            dataset_train,
            batch_sampler=batch_sampler,
            collate_fn=collate_fn,
            num_workers=0,
            drop_last=False,
        )

        for i, data in enumerate(dataloader_train):
            if i == 0:
                print("Batch 0:", data[0].cpu().numpy())
            elif i == 1:
                print("Batch 1:", data[0].cpu().numpy())
                break

        dataloader_test = None
        if args.eval_loss:
            print("loading eval dataset...")
            eval_paths = [path.replace('train', 'test') for path in args.data_path]
            dataset_test = MultiSizeQuantumDataset(eval_paths, sizes)
            test_batch_sampler = GroupedBatchSampler(dataset_test, batch_sizes)
            dataloader_test = DataLoader(
                dataset_test,
                batch_sampler=test_batch_sampler,
                collate_fn=collate_fn,
                num_workers=0,
                drop_last=False,
            )
    else:
        print("Training with single-size dataset (oseq)...")
        dataset_train = SimpleQuantumDataset(args.data_path[0])
        dataloader_test = None
        if args.eval_loss:
            dataset_test = SimpleQuantumDataset(args.data_path[0].replace('train', 'test'))
            dataloader_test = DataLoader(
                dataset_test,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=0,
                drop_last=False,
            )

        print(f"\n✅ Total number of samples in training dataset: {len(dataset_train):,}")

        dataloader_train = DataLoader(
            dataset_train,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=False,
        )

        for i, data in enumerate(dataloader_train):
            if i == 0:
                print("Batch 0:", data[0].cpu().numpy())
            elif i == 1:
                print("Batch 1:", data[0].cpu().numpy())
                break

    total_steps = args.epochs * len(dataloader_train)

    if args.model_type == 'oseq':
        model = QuantumDiffusionModel_oseq(
            hidden_dim=args.hidden_dim,
            num_layers=args.layer_num,
            head_count=args.head_num,
        ).to('cuda')
    elif args.model_type in MODEL_TYPES_2D:
        max_n = (
            args.max_N
            if args.max_N is not None
            else dataset_train.max_num_sites
        )
        rope_st = None if args.rope_scaling_type == 'none' else args.rope_scaling_type
        model_classes = {
            'oseq_rope_2d': QuantumDiffusionModel_oseq_rope_2d,
            'oseq_rope_2d_relative_b': QuantumDiffusionModel_oseq_rope_2d_relative_b,
            'oseq_rope_2d_g': QuantumDiffusionModel_oseq_rope_2d_g,
        }
        model_class = model_classes[args.model_type]
        model = model_class(
            num_layers=args.layer_num,
            head_count=args.head_num,
            hidden_dim=args.hidden_dim,
            max_seq_len=args.max_seq_len,
            max_N=max_n,
            rope_scaling_type=rope_st,
            rope_scaling_factor=args.rope_scaling_factor,
            rope_theta=args.rope_theta,
        ).to('cuda')
    elif args.model_type == 'oseq_rope_gnn':
        rope_st = None if args.rope_scaling_type == 'none' else args.rope_scaling_type
        model = QuantumDiffusionModel_oseq_rope_gnn(
            num_layers=args.layer_num,
            head_count=args.head_num,
            num_global_heads=args.num_global_heads,
            local_window_radius=args.local_window_radius,
            hidden_dim=args.hidden_dim,
            max_seq_len=args.max_seq_len,
            rope_scaling_type=rope_st,
            rope_scaling_factor=args.rope_scaling_factor,
            rope_theta=args.rope_theta,
        ).to('cuda')
    elif args.model_type == 'oseq_rope_sharepos':
        max_n = args.max_N if args.max_N is not None else max(args.qubits)
        rope_st = None if args.rope_scaling_type == 'none' else args.rope_scaling_type
        model = QuantumDiffusionModel_oseq_rope_sharepos(
            num_layers=args.layer_num,
            head_count=args.head_num,
            hidden_dim=args.hidden_dim,
            max_seq_len=args.max_seq_len,
            max_N=max_n,
            rope_scaling_type=rope_st,
            rope_scaling_factor=args.rope_scaling_factor,
            rope_theta=args.rope_theta,
        ).to('cuda')
    else:
        max_n = args.max_N if args.max_N is not None else max(args.qubits)
        rope_st = None if args.rope_scaling_type == 'none' else args.rope_scaling_type
        model = QuantumDiffusionModel_oseq_rope(
            num_layers=args.layer_num,
            head_count=args.head_num,
            hidden_dim=args.hidden_dim,
            max_seq_len=args.max_seq_len,
            max_N=max_n,
            rope_scaling_type=rope_st,
            rope_scaling_factor=args.rope_scaling_factor,
            rope_theta=args.rope_theta,
        ).to('cuda')

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    scheduler = get_scheduler(optimizer, args, total_steps)

    grad_norms = []
    lrs = []
    epoch_losses = []
    epoch_ce_losses = []
    epoch_pair_losses = []
    test_losses = []

    best_eval_loss = float('inf')
    patience = args.patience
    patience_counter = 0

    restored_state = load_training_checkpoint(args, model, optimizer, scheduler, map_location='cuda')
    start_epoch = restored_state.get('start_epoch', 0)
    grad_norms = restored_state.get('grad_norms', grad_norms)
    lrs = restored_state.get('lrs', lrs)
    epoch_losses = restored_state.get('epoch_losses', epoch_losses)
    epoch_ce_losses = restored_state.get('epoch_ce_losses', epoch_ce_losses)
    epoch_pair_losses = restored_state.get('epoch_pair_losses', epoch_pair_losses)
    test_losses = restored_state.get('test_losses', test_losses)
    best_eval_loss = restored_state.get('best_eval_loss', best_eval_loss)
    patience_counter = restored_state.get('patience_counter', patience_counter)

    if start_epoch >= args.epochs:
        print(f"Checkpoint start epoch {start_epoch + 1} is beyond --epochs {args.epochs}; no extra epochs will run.")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss = 0.0
        total_ce_loss = 0.0
        total_pair_loss = 0.0
        eps_epoch = get_eps_for_epoch(args, epoch)
        for data in tqdm(dataloader_train, desc=f"Epoch {epoch+1}/{args.epochs}"):
            if args.model_type in MODEL_TYPES_2D:
                condition = data['condition'].to('cuda')
                pauli_grid = data['pauli_grid'].to('cuda')
                outcome_grid = data['outcome_grid'].to('cuda')
                noisy_outcomes, masked_indices, _ = forward_process_2d(
                    outcome_grid=outcome_grid,
                    eps=eps_epoch,
                )
                pred = model.forward_2d(
                    condition=condition,
                    pauli_grid=pauli_grid,
                    outcome_grid=noisy_outcomes,
                    mask_indices=masked_indices,
                )
                loss, ce_loss, pair_loss = combined_2d_loss(
                    pred=pred,
                    pauli_grid=pauli_grid,
                    outcome_grid=outcome_grid,
                    masked_indices=masked_indices,
                    loss_fn=loss_fn,
                    pair_loss_weight=args.pair_loss_weight,
                    pair_loss_distances=args.pair_loss_distances,
                )
            else:
                data = data.to('cuda')
                _, length = data.shape
                qubit = int((length - 1) // 2)

                noisy_data, masked_indices, _ = forward_process(
                    batch=data,
                    qubit=qubit,
                    eps=eps_epoch,
                )
                pred = model(noisy_data.unsqueeze(-1), mask_indices=masked_indices)

                loss, ce_loss, pair_loss = combined_oseq_loss(
                    pred=pred,
                    data=data,
                    masked_indices=masked_indices,
                    loss_fn=loss_fn,
                    pair_loss_weight=args.pair_loss_weight,
                    pair_loss_distances=args.pair_loss_distances,
                )

            optimizer.zero_grad()
            loss.backward()

            total_norm = 0.0
            for name, param in model.named_parameters():
                if param.grad is not None:
                    total_norm += param.grad.data.norm(2).item() ** 2
            total_norm = total_norm ** 0.5
            grad_norms.append(total_norm)

            if args.print_grad_per_step:
                print(
                    f"  Loss: {loss.item():.6f} | CE: {ce_loss.item():.6f} "
                    f"| Pair: {pair_loss.item():.6f} | Grad Norm: {total_norm:.6f}"
                )

            optimizer.step()

            if scheduler is not None:
                scheduler.step()
                lrs.append(optimizer.param_groups[0]['lr'])

            total_loss += loss.item()
            total_ce_loss += ce_loss.item()
            total_pair_loss += pair_loss.item()

        avg_train_loss = total_loss / len(dataloader_train)
        avg_ce_loss = total_ce_loss / len(dataloader_train)
        avg_pair_loss = total_pair_loss / len(dataloader_train)
        epoch_losses.append(avg_train_loss)
        epoch_ce_losses.append(avg_ce_loss)
        epoch_pair_losses.append(avg_pair_loss)

        if args.eval_loss and dataloader_test is not None:
            avg_eval_loss = evaluate(model, dataloader_test, loss_fn, args, eps=eps_epoch)
            test_losses.append(avg_eval_loss)
            model.train()
        else:
            avg_eval_loss = 0.0

        avg_grad_norm = sum(grad_norms[-len(dataloader_train):]) / len(dataloader_train)
        current_lr = optimizer.param_groups[0]['lr']
        eps_info = f" | eps: {eps_epoch:.4e}" if args.eps_schedule != 'constant' else ""
        print(
            f"Epoch {epoch+1}/{args.epochs} - Train Loss: {avg_train_loss:.8f} | Eval Loss: {avg_eval_loss:.8f} "
            f"| CE: {avg_ce_loss:.8f} | Pair: {avg_pair_loss:.8f} "
            f"| Grad: {avg_grad_norm:.6f} | LR: {current_lr:.2e}{eps_info}"
        )

        epoch_number = epoch + 1
        if epoch_number in pre_final_checkpoint_epochs:
            checkpoint_root, checkpoint_ext = os.path.splitext(
                args.model_save_path
            )
            checkpoint_ext = checkpoint_ext or '.pth'
            checkpoint_path = (
                f'{checkpoint_root}_epoch_{epoch_number}{checkpoint_ext}'
            )
            torch.save(model.state_dict(), checkpoint_path)
            print(f'Checkpoint saved: {checkpoint_path}')


        if args.eval_loss:
            if avg_eval_loss < best_eval_loss:
                best_eval_loss = avg_eval_loss
                torch.save(model.state_dict(), args.model_save_path)
                print(f"✅ Best model saved, eval loss: {best_eval_loss:.8f}")
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= patience:
                print(f"🛑 Early stopping after {epoch+1} epochs.")
                break

        if avg_grad_norm < 1e-5:
            print("Gradient norm is too small, stopping training.")
            break

    torch.save(model.state_dict(), args.model_save_path.replace('.pth', '_final.pth'))
    print("Model saved.")

    train_log = {
        "args": vars(args),
        "epoch_losses": epoch_losses,
        "epoch_ce_losses": epoch_ce_losses,
        "epoch_pair_losses": epoch_pair_losses,
        "test_losses": test_losses,
        "final_loss": epoch_losses[-1] if epoch_losses else None,
        "best_test_loss": best_eval_loss if args.eval_loss else None,
        "total_epochs": len(epoch_losses),
    }

    log_save_path = args.loss_save_path
    if os.path.exists(log_save_path):
        with open(log_save_path, 'r') as f:
            try:
                logs = json.load(f)
            except json.JSONDecodeError:
                logs = []
    else:
        logs = []

    logs.append(train_log)
    with open(log_save_path, 'w') as f:
        json.dump(logs, f, indent=4)

    print(f"✅ Training log appended to: {log_save_path}")


if __name__ == "__main__":
    main()
