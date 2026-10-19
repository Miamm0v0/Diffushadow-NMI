import torch
import torch.nn as nn
from numericalmodel import NumericalDiffusionModel
import json
import torch.nn.functional as F
from Qdmodel_oseq import QuantumDiffusionModel_oseq
from Qdmodel_nseq import QuantumDiffusionModel_nseq
from Qdmodel_nseq_rope import QuantumDiffusionModel_nseq_rope
from Qdmodel_oseq_rope import QuantumDiffusionModel_oseq_rope
from Qdmodel_oseq_rope_sharepos import QuantumDiffusionModel_oseq_rope_sharepos
from Qdmodel_oseq_rope_gnn import QuantumDiffusionModel_oseq_rope_gnn
from Qmultimodel import QuantumMultimodalDModel_nseq
from Qdmodel_nseq_likeshadow import QuantumDModel_likeshadow
from test_1 import generate_nseq, generate_oseq, generate_nseq_both, generate_nseq_batch, generate_oseq_batch
from eval_utils import median_of_means, median_of_means_two, median_of_means_X, find_g_key, median_of_means_two_from_c
from eval_utils import ground_cal_averge_two_point_pbc, ground_cal_averge_Xstring_pbc, obtain_eigenrho, Hamiltonian_sym_circle, SparseTFI, calculate_exact_values, cal_ZZ_nearest_from_c
from eval_utils import median_of_means_correlation, median_of_means_heisenberg, median_of_means_xxz_energy_and_derivative, median_of_means_xxz_magnetization
from eval_utils import median_of_means_j1j2_energy, median_of_means_j1j2_dimer_proxy, median_of_means_annni_energy, median_of_means_single_pauli, median_of_means_annni_structure_factor_z
# from eval_data_heisenberg import compute_exact_heisenberg_values
# from eval_data_xxz import compute_exact_xxz_values
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import argparse
import os
import random
import subprocess
import sys

MASK_TOKEN_ID = -1.0

_EXACT_CAL_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_cal_exact.py")
_DEFAULT_EXACT_CACHE_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), ".", "eval_exact_cache")
)


def _parse_d_values(raw_values):
    values = []
    for raw in raw_values:
        for part in str(raw).replace(",", " ").split():
            try:
                value = int(part)
            except ValueError as exc:
                raise argparse.ArgumentTypeError(
                    f"Invalid distance value {part!r}; --d_values expects positive integers."
                ) from exc
            if value < 1:
                raise argparse.ArgumentTypeError("--d_values expects positive integers.")
            if value not in values:
                values.append(value)
    if not values:
        raise argparse.ArgumentTypeError("--d_values cannot be empty.")
    return values


def _select_distance_rows(array, d_values, name):
    if array is None:
        return None
    max_d = array.shape[0]
    missing = [d for d in d_values if d > max_d]
    if missing:
        raise ValueError(
            f"{name} only has cached rows for d=1..{max_d}, "
            f"but --d_values requested {missing}."
        )
    return array[np.array(d_values, dtype=int) - 1]


def _npz_has_keys(path, required_keys):
    if not os.path.isfile(path):
        return False
    try:
        with np.load(path) as data:
            return all(key in data.files for key in required_keys)
    except Exception:
        return False


def _ensure_exact_cache_and_load(args):
    """若缓存不存在则调用 eval_cal_exact.py，再从磁盘加载精确值（与 eval.py 中计算结果一致）。"""
    cache_dir = _DEFAULT_EXACT_CACHE_DIR
    pm = args.predict_model
    nq = args.num_qubits
    script = _EXACT_CAL_SCRIPT

    def _run_cal_exact():
        cmd = [
            sys.executable,
            script,
            "--predict_model",
            pm,
            "--num_qubits",
            str(nq),
            "--cache_dir",
            cache_dir,
        ]
        if pm == "xxz":
            cmd.extend(["--J_xy", "1.0"])
        if pm == "J1J2":
            cmd.extend(["--J1", str(args.J1)])
        if pm == "ANNNI":
            cmd.extend(["--J1", str(args.J1), "--annni_kappa", str(args.annni_kappa)])
        r = subprocess.run(cmd, cwd=os.path.dirname(script) or ".")
        if r.returncode != 0:
            raise RuntimeError(
                f"eval_cal_exact.py 退出码 {r.returncode}，命令: {' '.join(cmd)}"
            )

    if pm == "TFI":
        npz_p = os.path.join(cache_dir, f"exact_TFI_N{int(nq)}.npz")
        if not os.path.isfile(npz_p):
            _run_cal_exact()
        if not os.path.isfile(npz_p):
            raise FileNotFoundError(f"精确值缓存未生成: {npz_p}")
        z = np.load(npz_p)
        return {
            "h_values": z["h_values"],
            "ZZ_curves": z["ZZ_curves"],
            "Xs_curves": z["Xs_curves"],
            "Energy": z["Energy"],
        }
    if pm == "Heisenberg":
        npz_p = os.path.join(cache_dir, f"exact_Heisenberg_N{int(nq)}.npz")
        if not os.path.isfile(npz_p):
            _run_cal_exact()
        if not os.path.isfile(npz_p):
            raise FileNotFoundError(f"精确值缓存未生成: {npz_p}")
        z = np.load(npz_p)
        exact_value = {
            "energy": z["energy"],
            "correlations_XX": z["correlations_XX"],
            "correlations_YY": z["correlations_YY"],
            "correlations_ZZ": z["correlations_ZZ"],
            "correlations_spin_dot": z["correlations_spin_dot"],
        }
        return {"h_values": z["h_values"], "exact_value": exact_value}
    if pm == "xxz":
        J_xy = 1.0
        npz_p = os.path.join(cache_dir, f"exact_xxz_N{int(nq)}_J{J_xy}.npz")
        meta_p = npz_p.replace(".npz", "_meta.json")
        required_xxz_keys = [
            "single_energy",
            "single_energy_derivative",
            "single_magnetization_z",
            "single_magnetization_staggered",
            "single_corr_XX",
            "single_corr_YY",
            "single_corr_ZZ",
            "c_energy",
            "c_energy_derivative",
            "c_corr_XX",
            "c_corr_YY",
            "c_corr_ZZ",
            "energy_per_bond",
            "energy_derivative_per_bond",
            "energy_total",
            "energy_derivative_total",
            "single_energy_per_bond",
            "single_energy_derivative_per_bond",
            "single_energy_total",
            "single_energy_derivative_total",
            "single_local_Sz",
            "c_energy_per_bond",
            "c_energy_derivative_per_bond",
            "c_total_energy",
            "c_total_energy_derivative",
        ]
        if not (
            os.path.isfile(npz_p)
            and os.path.isfile(meta_p)
            and _npz_has_keys(npz_p, required_xxz_keys)
        ):
            _run_cal_exact()
        if not (os.path.isfile(npz_p) and os.path.isfile(meta_p)):
            raise FileNotFoundError(f"精确值缓存未生成: {npz_p} / {meta_p}")
        if not _npz_has_keys(npz_p, required_xxz_keys):
            raise KeyError(f"XXZ exact cache misses multimodal true-value fields: {npz_p}")
        z = np.load(npz_p)
        with open(meta_p, "r", encoding="utf-8") as f:
            special_points = json.load(f)
        exact_value = {
            "N": int(z["N"].item()),
            "J_xy": float(z["J_xy"].item()),
            "Delta_values": z["delta_values"].tolist(),
            "energy": z["energy"],
            "magnetization_z": z["magnetization_z"],
            "magnetization_staggered": z["magnetization_staggered"],
            "local_Sz": z["local_Sz"],
            "zz_correlation": z["zz_correlation"],
            "energy_derivative_numeric": z["energy_derivative_numeric"],
            "energy_derivative_analytic": z["energy_derivative_analytic"],
            "energy_second_derivative": z["energy_second_derivative"],
            "energy_total": z["energy_total"],
            "energy_per_bond": z["energy_per_bond"],
            "energy_derivative_per_bond": z["energy_derivative_per_bond"],
            "energy_derivative_total": z["energy_derivative_total"],
            "single_energy": z["single_energy"],
            "single_energy_derivative": z["single_energy_derivative"],
            "single_energy_per_bond": z["single_energy_per_bond"],
            "single_energy_derivative_per_bond": z["single_energy_derivative_per_bond"],
            "single_energy_total": z["single_energy_total"],
            "single_energy_derivative_total": z["single_energy_derivative_total"],
            "single_corr_XX": z["single_corr_XX"],
            "single_corr_YY": z["single_corr_YY"],
            "single_corr_ZZ": z["single_corr_ZZ"],
            "single_local_Sz": z["single_local_Sz"],
            "single_magnetization_z": z["single_magnetization_z"],
            "single_magnetization_staggered": z["single_magnetization_staggered"],
            "c_energy": z["c_energy"],
            "c_energy_derivative": z["c_energy_derivative"],
            "c_energy_per_bond": z["c_energy_per_bond"],
            "c_energy_derivative_per_bond": z["c_energy_derivative_per_bond"],
            "c_total_energy": z["c_total_energy"],
            "c_total_energy_derivative": z["c_total_energy_derivative"],
            "c_corr_XX": z["c_corr_XX"],
            "c_corr_YY": z["c_corr_YY"],
            "c_corr_ZZ": z["c_corr_ZZ"],
            "special_points": special_points,
        }
        return {"delta_values": z["delta_values"], "exact_value": exact_value}
    if pm == "J1J2":
        npz_p = os.path.join(cache_dir, f"exact_J1J2_N{int(nq)}_J1{args.J1}.npz")
        if not os.path.isfile(npz_p):
            _run_cal_exact()
        if not os.path.isfile(npz_p):
            raise FileNotFoundError(f"精确值缓存未生成: {npz_p}")
        z = np.load(npz_p)
        exact_value = {
            "energy": z["energy"],
            "correlations_XX": z["correlations_XX"],
            "correlations_YY": z["correlations_YY"],
            "correlations_ZZ": z["correlations_ZZ"],
            "correlations_spin_dot": z["correlations_spin_dot"],
            "dimer_proxy": z["dimer_proxy"],
        }
        return {"J2_values": z["J2_values"], "exact_value": exact_value}
    if pm == "ANNNI":
        npz_p = os.path.join(
            cache_dir,
            f"exact_ANNNI_N{int(nq)}_kappa{args.annni_kappa}_J1{args.J1}.npz",
        )
        if not os.path.isfile(npz_p):
            _run_cal_exact()
        if not os.path.isfile(npz_p):
            raise FileNotFoundError(f"精确值缓存未生成: {npz_p}")
        z = np.load(npz_p)
        exact_value = {
            "energy": z["energy"],
            "ZZ_curves": z["ZZ_curves"],
            "X_magnetization": z["X_magnetization"],
            "structure_factor_pi": z["structure_factor_pi"],
            "structure_factor_pi_over_2": z["structure_factor_pi_over_2"],
        }
        return {"h_values": z["h_values"], "exact_value": exact_value}
    return {}




def parse_args():
    parser = argparse.ArgumentParser(description="Numerical Diffusion Model Training with LR Scheduler")

    # 数据相关
    parser.add_argument('--input_sequence', type=str, default='oseq', choices=['nseq', 'oseq'], help='Type of input sequence: nseq (interleaved) or oseq (ordered)')

    parser.add_argument('--multimodal', action="store_true", help='Whether to use multimodal model (with modal C)')

    parser.add_argument(
        '--model_type',
        type=str,
        default='lyh',
        choices=['lyh', 'likeshadow', 'nseq_rope', 'oseq_rope', 'oseq_rope_sharepos', 'oseq_rope_gnn'],
        help='lyh / likeshadow / nseq_rope / oseq_rope / oseq_rope_sharepos（共享 pair RoPE id）/ oseq_rope_gnn（+局域混合头）',
    )

    parser.add_argument('--exact_value', action="store_true", help="whether to use exact values")

    parser.add_argument('--diffusion_steps', type=int, default=2, help="Number of diffusion steps")

    parser.add_argument(
        '--temperature',
        type=float,
        default=1.0,
        help="Sampling temperature (>0): lower is sharper, higher is flatter",
    )

    parser.add_argument('--repeat_times', type=int, default=6, help="Number of experiment repeats (for averaging measurements)")

    parser.add_argument('--h_length', type=int, default=41, help="Number of h points to generate. you can decrease to accelerate evaluation")

    parser.add_argument('--num_qubits', type=int, default=10, help="Number of qubits")

    parser.add_argument('--model_path', type=str, default='', help="path to the model to be evaluated")

    parser.add_argument('--save_data_path', type=str, default='', help="path to save the generated data")

    parser.add_argument('--generate_target', type=str, default='b', choices=['b','c','both'], help='the training target of model')

    parser.add_argument('--predict_model', type=str, default='TFI', choices=['TFI', 'Heisenberg', 'xxz', 'J1J2', 'ANNNI'], help='the physical model to evaluate')
    parser.add_argument('--J1', type=float, default=1.0, help='J1 coupling for J1J2/ANNNI')
    parser.add_argument('--annni_kappa', type=float, default=0.5, help='ANNNI next-nearest-neighbor coupling ratio')

    parser.add_argument('--sample_size_per_h', type=int, default=10000, help="number of samples to generate per h value")
    parser.add_argument(
        '--d_values',
        nargs='+',
        default=['1', '2', '3', '4', '5'],
        help='distance d values to evaluate, e.g. "--d_values 1 3 5" or "--d_values 1,3,5"',
    )
    parser.add_argument(
        "--gen_batch_size",
        type=int,
        default=256,
        help="generation batch size for GPU utilization (chunk sample_size_per_h)",
    )
    parser.add_argument(
        '--pair_condition_data_path',
        type=str,
        default='/home/fyp26lyh/QuantumLLADA/data/tfi_dataset_41_100_eval.json',
        help='pair/C condition JSON used when --multimodal --generate_target b',
    )
    parser.add_argument(
        '--single_condition_data_path',
        type=str,
        default='',
        help='single/nseq condition JSON used when --multimodal --generate_target c',
    )
    parser.add_argument(
        '--condition_group_size',
        type=int,
        default=100,
        help='number of condition rows per h value in condition JSON files',
    )
    parser.add_argument(
        '--pair_basis_mode',
        type=str,
        default='auto',
        choices=['auto', 'full', 'diagonal'],
        help='pair/C generation bases: auto uses diagonal XX/YY/ZZ for XXZ and full 9 Pauli products otherwise',
    )

    parser.add_argument('--confidence_decoding', action="store_true", help="whether to use confidence decoding")

    parser.add_argument('--hidden_dim', type=int, default=128, help="number of hidden dimensions")
    parser.add_argument('--layer_num', type=int, default=3, help="number of layers")
    parser.add_argument('--head_num', type=int, default=4, help="number of heads")

    parser.add_argument('--max_seq_len', type=int, default=4096, help='RoPE max_position_embeddings (nseq_rope)')
    parser.add_argument(
        '--max_N',
        type=int,
        default=None,
        help='nseq_rope 等；oseq_rope / oseq_rope_sharepos 构造参数（当前未用）。oseq_rope_gnn 不需要 max_N',
    )
    parser.add_argument(
        '--rope_scaling_type',
        type=str,
        default='none',
        choices=['none', 'linear', 'dynamic', 'ntk'],
        help='RoPE scaling: none | linear | dynamic/ntk (须与训练时一致以加载权重)',
    )
    parser.add_argument('--rope_scaling_factor', type=float, default=1.0, help='RoPE scaling factor')
    parser.add_argument('--rope_theta', type=float, default=10000.0, help='RoPE theta')

    parser.add_argument(
        '--num_global_heads',
        type=int,
        default=1,
        help='oseq_rope_gnn：全注意力头数（须与训练一致）',
    )
    parser.add_argument(
        '--local_window_radius',
        type=int,
        default=2,
        help='oseq_rope_gnn：局域头窗口半径 r（须与训练一致）',
    )

    args = parser.parse_args()
    try:
        args.d_values = _parse_d_values(args.d_values)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))
    return args


def _extract_state_dict(loaded):
    """从 torch.load 的返回值中取出 state_dict。"""
    if not isinstance(loaded, dict):
        return loaded
    for k in ("state_dict", "model_state_dict", "model"):
        if k in loaded and isinstance(loaded[k], dict):
            return loaded[k]
    return loaded


def _group_condition_data(path, h_length, group_size, name):
    if not path:
        return None
    with open(path, "r") as f:
        rows = json.load(f)
    if group_size <= 0:
        raise ValueError("--condition_group_size must be positive")
    grouped = [rows[i:i + group_size] for i in range(0, len(rows), group_size)]
    if len(grouped) < h_length:
        raise ValueError(
            f"{name} condition file has only {len(grouped)} h groups, "
            f"but h_length={h_length}. path={path}"
        )
    return grouped


def _single_rows_to_nseq(rows, num_qubits, device):
    rows = torch.tensor(rows, dtype=torch.float32, device=device)
    if rows.dim() == 1:
        rows = rows.unsqueeze(0)
    expected_len = 1 + 2 * num_qubits
    if rows.shape[1] != expected_len:
        raise ValueError(f"single condition rows must have length {expected_len}, got {rows.shape[1]}")

    body = rows[:, 1:]
    first_half = body[:, :num_qubits]
    second_half = body[:, num_qubits:]
    odd_basis = rows[:, 1::2]
    even_bits = rows[:, 2::2]

    looks_nseq = bool(((first_half >= 2) & (first_half <= 4)).all() and ((second_half == 0) | (second_half == 1)).all())
    looks_oseq = bool(((odd_basis >= 2) & (odd_basis <= 4)).all() and ((even_bits == 0) | (even_bits == 1)).all())

    if looks_nseq and not looks_oseq:
        return rows
    if looks_oseq and not looks_nseq:
        nseq = torch.zeros_like(rows)
        nseq[:, 0] = rows[:, 0]
        nseq[:, 1:1 + num_qubits] = odd_basis
        nseq[:, 1 + num_qubits:] = even_bits
        return nseq

    return rows


def _nseq_to_oseq_rows(single_nseq, num_qubits):
    rows = torch.zeros_like(single_nseq)
    rows[:, 0] = single_nseq[:, 0]
    rows[:, 1::2] = single_nseq[:, 1:1 + num_qubits]
    rows[:, 2::2] = single_nseq[:, 1 + num_qubits:]
    return rows


def _sample_single_condition(single_group, batch_size, h, num_qubits, device):
    if single_group is None:
        single = torch.zeros((batch_size, 1 + 2 * num_qubits), dtype=torch.float32, device=device)
        single[:, 0] = float(h)
        single[:, 1:1 + num_qubits] = torch.randint(2, 5, (batch_size, num_qubits), device=device).float()
        single[:, 1 + num_qubits:] = torch.randint(0, 2, (batch_size, num_qubits), device=device).float()
        return single

    picked = random.choices(single_group, k=batch_size)
    return _single_rows_to_nseq(picked, num_qubits, device)


def _sample_pair_condition(pair_group, device, h=None):
    if pair_group is None:
        return None

    if isinstance(pair_group, torch.Tensor):
        tensor = pair_group
        if tensor.dim() >= 2 and tensor.shape[0] > 1:
            idx = random.randrange(tensor.shape[0])
            tensor = tensor[idx:idx + 1]
        elif tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        if h is not None and tensor.dim() >= 2 and tensor.shape[1] > 0:
            tensor[:, 0] = float(h)
        return tensor.to(device)

    if isinstance(pair_group, (list, tuple)):
        if len(pair_group) == 0:
            raise ValueError("pair condition group is empty")
        is_single_row = all(isinstance(value, (int, float)) for value in pair_group)
        row = pair_group if is_single_row else random.choice(pair_group)
    else:
        row = pair_group

    tensor = torch.as_tensor(row, dtype=torch.float32, device=device)
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)
    if h is not None and tensor.dim() >= 2 and tensor.shape[1] > 0:
        tensor[:, 0] = float(h)
    return tensor


def _cal_pair_product_from_c(sqe_c, pauli_code):
    num_qubits = int(sqe_c.shape[1] // 3)
    total = 0.0
    count = 0
    for i in range(num_qubits):
        r1 = sqe_c[:, 3 * i]
        r2 = sqe_c[:, 3 * i + 1]
        p_val = sqe_c[:, 3 * i + 2]
        mask = (r1 == pauli_code) & (r2 == pauli_code)
        if torch.any(mask):
            total += torch.mean((2 * p_val[mask] - 1).float())
            count += 1
    result = total / count if count > 0 else torch.tensor(0.0, device=sqe_c.device)
    return result.to(dtype=torch.float32)


def _cal_x4_from_c(sqe_c):
    num_qubits = int(sqe_c.shape[1] // 3)
    total = 0.0
    count = 0
    for i in range(num_qubits):
        j = (i + 2) % num_qubits
        r1_i = sqe_c[:, 3 * i]
        r2_i = sqe_c[:, 3 * i + 1]
        p_i = sqe_c[:, 3 * i + 2]
        r1_j = sqe_c[:, 3 * j]
        r2_j = sqe_c[:, 3 * j + 1]
        p_j = sqe_c[:, 3 * j + 2]
        mask = (r1_i == 2) & (r2_i == 2) & (r1_j == 2) & (r2_j == 2)
        if torch.any(mask):
            total += torch.mean(((2 * p_i[mask] - 1) * (2 * p_j[mask] - 1)).float())
            count += 1
    result = total / count if count > 0 else torch.tensor(0.0, device=sqe_c.device)
    return result.to(dtype=torch.float32)


def median_of_means_X_from_c(sqe_c, num_parts):
    T, _ = sqe_c.shape
    part_size = max(1, T // num_parts)
    means = []
    for i in range(num_parts):
        part_sqe_c = sqe_c[i * part_size:(i + 1) * part_size]
        if part_sqe_c.numel() == 0:
            continue
        means.append(_cal_pair_product_from_c(part_sqe_c, pauli_code=2).item())
    return torch.median(torch.tensor(means)) if means else torch.tensor(0.0)


def median_of_means_X4_from_c(sqe_c, num_parts):
    T, _ = sqe_c.shape
    part_size = max(1, T // num_parts)
    means = []
    for i in range(num_parts):
        part_sqe_c = sqe_c[i * part_size:(i + 1) * part_size]
        if part_sqe_c.numel() == 0:
            continue
        means.append(_cal_x4_from_c(part_sqe_c).item())
    return torch.median(torch.tensor(means)) if means else torch.tensor(0.0)


def _sample_pair_bases(batch_size, num_qubits, device, pair_basis_mode="full"):
    if pair_basis_mode == "diagonal":
        bases = torch.randint(2, 5, (batch_size, num_qubits, 1), device=device).float()
        return bases.expand(-1, -1, 2).clone()
    if pair_basis_mode == "full":
        return torch.randint(2, 5, (batch_size, num_qubits, 2), device=device).float()
    raise ValueError(f"Unknown pair_basis_mode: {pair_basis_mode}")


def _cal_pair_product_mean_from_c(sqe_c, pauli_code):
    if sqe_c is None or sqe_c.numel() == 0:
        return float("nan")
    num_qubits = int(sqe_c.shape[1] // 3)
    pair_data = sqe_c.reshape(-1, num_qubits, 3)
    r1 = pair_data[:, :, 0]
    r2 = pair_data[:, :, 1]
    product_value = pair_data[:, :, 2]
    mask = (r1 == pauli_code) & (r2 == pauli_code)
    if not torch.any(mask):
        return float("nan")
    return float((2 * product_value[mask].float() - 1).mean().item())


def median_of_means_xxz_from_c(J_xy, Delta, sqe_c, num_parts=10):
    """
    Estimate XXZ nearest-neighbor energy terms directly from C modality rows.
    Token convention: 2=X, 3=Y, 4=Z.
    """
    T, _ = sqe_c.shape
    part_size = max(1, T // num_parts)
    energy_means = []
    derivative_means = []
    xx_means = []
    yy_means = []
    zz_means = []

    for i in range(num_parts):
        part_sqe_c = sqe_c[i * part_size:(i + 1) * part_size]
        if part_sqe_c.numel() == 0:
            continue

        xx = _cal_pair_product_mean_from_c(part_sqe_c, pauli_code=2)
        yy = _cal_pair_product_mean_from_c(part_sqe_c, pauli_code=3)
        zz = _cal_pair_product_mean_from_c(part_sqe_c, pauli_code=4)

        if np.isfinite(xx):
            xx_means.append(xx)
        if np.isfinite(yy):
            yy_means.append(yy)
        if np.isfinite(zz):
            zz_means.append(zz)
            derivative_means.append(float(J_xy) * zz)
        if np.isfinite(xx) and np.isfinite(yy) and np.isfinite(zz):
            energy_means.append(float(J_xy) * (xx + yy) + float(J_xy) * float(Delta) * zz)

    def median_or_nan(values):
        if not values:
            return torch.tensor(float("nan"), dtype=torch.float32)
        return torch.median(torch.tensor(values, dtype=torch.float32))

    return (
        median_or_nan(energy_means),
        median_or_nan(derivative_means),
        median_or_nan(xx_means),
        median_or_nan(yy_means),
        median_or_nan(zz_means),
    )


def mean_std_ignore_nan(tensor):
    array = tensor.numpy()
    if not np.isfinite(array).any():
        shape = array.shape[1:]
        return np.full(shape, np.nan), np.full(shape, np.nan)
    return np.nanmean(array, axis=0), np.nanstd(array, axis=0)


def generate_GPTmeasureoutput_c_nseq(
    h,
    model,
    num_qubits,
    sample_size,
    diff_steps,
    temp,
    device,
    single_group=None,
    conf_decode=True,
    gen_batch_size=256,
    pair_basis_mode="full",
):
    L = 1 + 2 * num_qubits
    Lc = 1 + 3 * num_qubits
    steps = diff_steps
    temperature = temp
    use_sampling = True
    outputs = []
    outputs_c = []
    remaining = int(sample_size)

    p_positions = 3 + 3 * torch.arange(num_qubits, device=device)
    r1_positions = 1 + 3 * torch.arange(num_qubits, device=device)
    r2_positions = 2 + 3 * torch.arange(num_qubits, device=device)

    pbar = tqdm.tqdm(
        total=remaining,
        desc=f"Generating c (h={float(h):.2f})",
        unit="sample",
        ncols=100,
        colour='green',
    )

    while remaining > 0:
        B = min(int(gen_batch_size), remaining)
        single_nseq = _sample_single_condition(single_group, B, h, num_qubits, device)
        single_nseq[:, 0] = float(h)
        x_single = single_nseq.unsqueeze(-1)
        no_b_mask = torch.zeros((B, num_qubits), dtype=torch.bool, device=device)

        prompt_c = torch.full((B, Lc, 1), MASK_TOKEN_ID, device=device)
        prompt_c[:, 0, 0] = float(h)
        pair_bases = _sample_pair_bases(B, num_qubits, device, pair_basis_mode=pair_basis_mode)
        prompt_c[:, r1_positions, 0] = pair_bases[:, :, 0]
        prompt_c[:, r2_positions, 0] = pair_bases[:, :, 1]

        step_count = max(1, steps)
        num_unmask_per_step = max(1, (num_qubits + step_count - 1) // step_count)
        for _ in range(steps):
            prompt_flat_c = prompt_c.squeeze(-1)
            current_masked_c = prompt_flat_c[:, p_positions] == MASK_TOKEN_ID
            if not current_masked_c.any():
                break

            with torch.no_grad():
                _logits_b, logits_c = model(
                    x_single=x_single,
                    x_pair=prompt_c,
                    mask_indices=no_b_mask,
                    mask_indices_pair=current_masked_c,
                )

            probs_c = torch.softmax(logits_c / temperature, dim=-1)
            p_c1 = probs_c[..., 1]
            k = min(num_unmask_per_step, num_qubits)

            if conf_decode:
                eps = 1e-12
                entropy_c = - (probs_c * torch.log(probs_c + eps)).sum(dim=-1)
                scores = entropy_c.masked_fill(~current_masked_c, float("inf"))
            else:
                scores = torch.rand((B, num_qubits), device=device).masked_fill(~current_masked_c, float("inf"))

            idx = torch.topk(scores, k=k, largest=False, dim=1).indices
            valid = torch.isfinite(scores.gather(1, idx))
            if not valid.any():
                break

            p_selected = p_c1.gather(1, idx)
            if use_sampling:
                pred_values = torch.bernoulli(p_selected).float()
            else:
                pred_values = (p_selected > 0.5).float()

            global_pos = p_positions[idx]
            batch_idx = torch.arange(B, device=device).unsqueeze(1).expand_as(global_pos)
            prompt_c[batch_idx[valid], global_pos[valid], 0] = pred_values[valid]

        outputs.append(_nseq_to_oseq_rows(single_nseq, num_qubits))
        outputs_c.append(prompt_c.squeeze(-1))
        remaining -= B
        pbar.update(B)

    pbar.close()
    outputs = torch.cat(outputs, dim=0) if outputs else torch.empty((0, L), device=device)
    outputs_c = torch.cat(outputs_c, dim=0) if outputs_c else torch.empty((0, Lc), device=device)
    return outputs, outputs_c


def generate_GPTmeasureoutput_nseq(
    h,
    model,
    num_qubits,
    sample_size,
    diff_steps,
    temp,
    device,
    generate_target,
    c=None,
    conf_decode=True,
    gen_batch_size=256,
    pair_basis_mode="full",
):
    if generate_target == 'c':
        return generate_GPTmeasureoutput_c_nseq(
            h,
            model,
            num_qubits,
            sample_size,
            diff_steps,
            temp,
            device,
            single_group=c,
            conf_decode=conf_decode,
            gen_batch_size=gen_batch_size,
            pair_basis_mode=pair_basis_mode,
        )

    """
    使用 diffusion 的 generate() 函数生成量子测量输出
    格式: [h, r1, b1, r2, b2, ..., rN, bN]，共 1 + 2*num_qubits 列
    
    Args:
        h: 外磁场值（标量或列表）
        model: 训练好的 diffusion model (QuantumDiffusionModel)
        num_qubits: 量子比特数（如 10）
        sample_size: 生成样本数量
        device: 'cuda' 或 'cpu'
    
    Returns:
        outputs: (sample_size, 1 + 2*num_qubits) 的 tensor
    """
    L = 1 + 2 * num_qubits  # 总长度：h + r1,b1, ..., rN,bN
    seq_len = num_qubits     # 状态序列长度（b1~bN）
    steps = diff_steps                # diffusion 去噪步数
    temperature = temp        # 温度
    use_sampling = True      # 是否用随机采样

    outputs = []
    outputs_c = []

    pbar = tqdm.tqdm(
        range(sample_size),
        desc=f"Generating (h={h:.2f})", 
        unit="sample",
        ncols=100,
        colour='green'
    )

    for _ in pbar:
        # --- 构造 prompt: [h, r1, r2, ..., rN, -1, -1, ..., -1] ---
        # 注意：你的模型输入是 [g, P1~P10, b1~b10] → 对应 [h, r1~r10, b1~b10]
        prompt = torch.full((1, L, 1), MASK_TOKEN_ID, device=device)  # [1, L, 1]


        # 设置 h (g)
        prompt[0, 0, 0] = float(h)

        # 随机生成 r1~rN (P1~PN)，作为条件输入
        random_inputs = torch.randint(2, 5, (num_qubits,), device=device)  # [N]
        prompt[0, 1:1+num_qubits, 0] = random_inputs  # r1~rN 填入 P 位置
        # print("the shape of prompt:", prompt.shape)
        # print("the prompt:", prompt)
        if generate_target == 'both':

            prompt_c = torch.full((1, L+num_qubits, 1), MASK_TOKEN_ID, device=device)
            prompt_c[0, 0, 0] = float(h)
            pair_bases = _sample_pair_bases(
                1,
                num_qubits,
                device,
                pair_basis_mode=pair_basis_mode,
            )[0]

            # 交错填充：r1,r2,MASK, r2,r3,MASK, r3,r4,MASK, ...
            for i in range(num_qubits):
                # 每3个位置为一组：两个r和一个MASK
                start_idx = 1 + i * 3
                prompt_c[0, start_idx, 0] = pair_bases[i, 0]
                prompt_c[0, start_idx + 1, 0] = pair_bases[i, 1]

            with torch.no_grad():
                completed_both_b, completed_both_c = generate_nseq_both(
                    model=model,
                    prompt=prompt,
                    prompt_c=prompt_c,
                    steps=steps,
                    mask_id=MASK_TOKEN_ID,
                    num_qubits=num_qubits,
                    temperature=temperature,
                    use_sampling=use_sampling
                )

            b_values = completed_both_b[0, 1+num_qubits:1+2*num_qubits, 0]  # [N]
            b_values = (b_values > 0.5).float()  # 确保是 0/1

            # --- 组合成 [h, r1, b1, r2, b2, ..., rN, bN] ---
            row = torch.zeros(L, device=device)
            row[0] = h
            for i in range(num_qubits):
                row[1 + 2*i]   = random_inputs[i]    # r_i
                row[1 + 2*i+1] = b_values[i]         # b_i

            outputs.append(row)
            outputs_c.append(completed_both_c.squeeze())


            # 清理缓存
            torch.cuda.empty_cache()

    


        # b1~bN 保持 mask (-1)，等待 diffusion 生成

        # --- 调用 diffusion generate() ---
        else:
            with torch.no_grad():
                completed = generate_nseq(
                    model=model,
                    prompt=prompt,
                    steps=steps,
                    C=_sample_pair_condition(c, device, h=h),
                    num_qubits=num_qubits,
                    temperature=temperature,
                    use_sampling=use_sampling,
                    conf_decode=conf_decode
                )  # 输出 [1, L, 1]

            # --- 提取 b1~bN ---
            b_values = completed[0, 1+num_qubits:1+2*num_qubits, 0]  # [N]
            b_values = (b_values > 0.5).float()  # 确保是 0/1

            # --- 组合成 [h, r1, b1, r2, b2, ..., rN, bN] ---
            row = torch.zeros(L, device=device)
            row[0] = h
            for i in range(num_qubits):
                row[1 + 2*i]   = random_inputs[i]    # r_i
                row[1 + 2*i+1] = b_values[i]         # b_i

            outputs.append(row)

            # 清理缓存
            torch.cuda.empty_cache()

    # 堆叠成 batch
    outputs = torch.stack(outputs)  # [sample_size, L]
    outputs_c = torch.stack(outputs_c) if outputs_c else None
    return outputs, outputs_c

def generate_GPTmeasureoutput_oseq(
    h,
    model,
    num_qubits,
    sample_size,
    diff_steps,
    temp,
    device,
    generate_target,
    C,
    conf_decode=True,
    gen_batch_size=256,
):
    if generate_target != 'b':
        raise ValueError("generate_target='c' is only implemented for the multimodal nseq model.")

    """
    使用 diffusion 的 generate() 函数生成量子测量输出
    格式: [h, r1, b1, r2, b2, ..., rN, bN]，共 1 + 2*num_qubits 列
    
    Args:
        h: 外磁场值（标量或列表）
        model: 训练好的 diffusion model (QuantumDiffusionModel)
        num_qubits: 量子比特数（如 10）
        sample_size: 生成样本数量
        device: 'cuda' 或 'cpu'
    
    Returns:
        outputs: (sample_size, 1 + 2*num_qubits) 的 tensor
    """
    L = 1 + 2 * num_qubits  # 总长度：h + r1,b1, ..., rN,bN
    seq_len = num_qubits     # 状态序列长度（b1~bN）
    steps = diff_steps                # diffusion 去噪步数
    temperature = temp       # sampling temperature passed from --temperature
    use_sampling = True      # 是否用随机采样

    outputs = []
    outputs_c = None

    remaining = int(sample_size)
    pbar = tqdm.tqdm(
        total=remaining,
        desc=f"Generating (h={float(h):.2f})",
        unit="sample",
        ncols=100,
        colour="green",
    )

    r_positions = torch.arange(1, L, 2, device=device)  # [N]
    while remaining > 0:
        B = min(int(gen_batch_size), remaining)
        prompt = torch.full((B, L, 1), MASK_TOKEN_ID, device=device)
        prompt[:, 0, 0] = float(h)
        random_inputs = torch.randint(2, 5, (B, num_qubits), device=device).float()
        prompt[:, r_positions, 0] = random_inputs

        with torch.no_grad():
            completed = generate_oseq_batch(
                model=model,
                prompt=prompt,
                steps=steps,
                temperature=temperature,
                use_sampling=use_sampling,
            )  # [B, L, 1]

        b_positions = torch.arange(2, L, 2, device=device)
        b_values = completed[:, b_positions, 0]
        b_values = (b_values > 0.5).float()

        row = torch.zeros((B, L), device=device)
        row[:, 0] = float(h)
        for i in range(num_qubits):
            row[:, 1 + 2 * i] = random_inputs[:, i]
            row[:, 1 + 2 * i + 1] = b_values[:, i]
        outputs.append(row)

        remaining -= B
        pbar.update(B)
    pbar.close()

    outputs = torch.cat(outputs, dim=0) if outputs else torch.empty((0, L), device=device)
    return outputs, outputs_c


def calculate_errors(pred_h, pred_values, true_h, true_values):
    """计算预测值与真实值的误差"""
    errors = []      # 保持为 list，用于收集
    mse_values = []  # 保持为 list，用于收集
    
    for i, h_val in enumerate(pred_h):
        # 找到对应的真实值（在真实h值中找到最接近的点）
        true_idx = np.argmin(np.abs(true_h - h_val))
        pred_val = pred_values[i]
        true_val = true_values[true_idx]
        
        # 绝对误差
        abs_error = np.abs(pred_val - true_val)
        errors.append(abs_error)
        
        # 平方误差（用于MSE）
        sq_error = (pred_val - true_val) ** 2
        mse_values.append(sq_error)

    # ✅ 循环结束后，再转换为 numpy 数组
    errors = np.array(errors)
    mse_values = np.array(mse_values)
    
    # 转换为实数（如果包含复数，取实部）
    # 注意：np.real_if_close 返回的是 ndarray，所以这一步可以在转换后做
    errors = np.real_if_close(errors).astype(float)
    mse_values = np.real_if_close(mse_values).astype(float)
    
    return errors, mse_values


def convert_to_serializable(obj):
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, (complex, np.complex128, np.complex64)):
        return float(obj.real)  # 复数只取实部
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj
    



def main():
    args = parse_args()
    if not np.isfinite(args.temperature) or args.temperature <= 0.0:
        raise ValueError("--temperature must be a finite positive number.")
    MASK_TOKEN_ID = -1.0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # model_path = r'/home/fyp26lyh/QuantumLLADA/DshadowGPT/model_train/Quantum_diffusion_model.pth'
    model_path = args.model_path
    generate_target = args.generate_target
    pair_basis_mode = args.pair_basis_mode
    if pair_basis_mode == "auto":
        pair_basis_mode = "diagonal" if args.predict_model == "xxz" else "full"
    print(f"Using pair_basis_mode={pair_basis_mode}")
    if generate_target == 'c' and not args.multimodal:
        raise ValueError("--generate_target c requires --multimodal, because only QuantumMultimodalDModel_nseq has the c head.")
    if generate_target == 'c' and args.input_sequence != 'nseq':
        raise ValueError("--generate_target c is implemented for input_sequence=nseq multimodal models.")
    generate_length = 1 + 2 * args.num_qubits

    # model = NumericalDiffusionModel()
    if args.input_sequence == "nseq":
        model = QuantumDiffusionModel_nseq(num_layers=3, head_count=4, qubit=args.num_qubits, length=generate_length)
        generate_func = generate_GPTmeasureoutput_nseq
    else:
        model = QuantumDiffusionModel_oseq()
        generate_func = generate_GPTmeasureoutput_oseq
    if args.multimodal:
        model = QuantumMultimodalDModel_nseq()
    if args.model_type == 'likeshadow':
        model = QuantumDModel_likeshadow()
    if args.model_type == 'nseq_rope':
        rope_st = None if args.rope_scaling_type == 'none' else args.rope_scaling_type
        max_n = args.max_N if args.max_N is not None else args.num_qubits
        model = QuantumDiffusionModel_nseq_rope(
            hidden_dim=args.hidden_dim,
            num_layers=args.layer_num,
            head_count=args.head_num,
            max_seq_len=args.max_seq_len,
            max_N=max_n,
            rope_scaling_type=rope_st,
            rope_scaling_factor=args.rope_scaling_factor,
            rope_theta=args.rope_theta,
        )
    if args.model_type == 'oseq_rope':
        rope_st = None if args.rope_scaling_type == 'none' else args.rope_scaling_type
        max_n = args.max_N if args.max_N is not None else args.num_qubits
        model = QuantumDiffusionModel_oseq_rope(
            hidden_dim=args.hidden_dim,
            num_layers=args.layer_num,
            head_count=args.head_num,
            max_seq_len=args.max_seq_len,
            max_N=max_n,
            rope_scaling_type=rope_st,
            rope_scaling_factor=args.rope_scaling_factor,
            rope_theta=args.rope_theta,
        )
        generate_func = generate_GPTmeasureoutput_oseq
    if args.model_type == 'oseq_rope_sharepos':
        rope_st = None if args.rope_scaling_type == 'none' else args.rope_scaling_type
        max_n = args.max_N if args.max_N is not None else args.num_qubits
        model = QuantumDiffusionModel_oseq_rope_sharepos(
            hidden_dim=args.hidden_dim,
            num_layers=args.layer_num,
            head_count=args.head_num,
            max_seq_len=args.max_seq_len,
            max_N=max_n,
            rope_scaling_type=rope_st,
            rope_scaling_factor=args.rope_scaling_factor,
            rope_theta=args.rope_theta,
        )
        generate_func = generate_GPTmeasureoutput_oseq
    if args.model_type == 'oseq_rope_gnn':
        rope_st = None if args.rope_scaling_type == 'none' else args.rope_scaling_type
        model = QuantumDiffusionModel_oseq_rope_gnn(
            hidden_dim=args.hidden_dim,
            num_layers=args.layer_num,
            head_count=args.head_num,
            num_global_heads=args.num_global_heads,
            local_window_radius=args.local_window_radius,
            max_seq_len=args.max_seq_len,
            rope_scaling_type=rope_st,
            rope_scaling_factor=args.rope_scaling_factor,
            rope_theta=args.rope_theta,
        )
        generate_func = generate_GPTmeasureoutput_oseq
    _raw = torch.load(model_path, map_location=device)
    _sd = _extract_state_dict(_raw)
    if args.model_type == "oseq_rope_gnn" and isinstance(_sd, dict):
        _sd.pop("position_embedding.weight", None)
    model.load_state_dict(_sd, strict=True)
    diff_steps = args.diffusion_steps
    temperature = args.temperature
    d_values = args.d_values
    num_d = len(d_values)
    model.to(device)
    model.eval()

    gpt_eval_points = torch.zeros((args.repeat_times, num_d, args.h_length))
    gpt_eval_pointsX = torch.zeros((args.repeat_times, num_d, args.h_length))
    gpt_eval_points_energy = torch.zeros((args.repeat_times, args.h_length))

    gpt_eval_points_from_c = torch.zeros((args.repeat_times, args.h_length))
    gpt_eval_pointsX_from_c = torch.zeros((args.repeat_times, args.h_length))
    gpt_eval_pointsX4_from_c = torch.zeros((args.repeat_times, args.h_length))


    if args.predict_model == "Heisenberg":
        gpt_eval_energy = torch.zeros((args.repeat_times, args.h_length))
        gpt_eval_corr_XX = torch.zeros((args.repeat_times, num_d, args.h_length))
        gpt_eval_corr_YY = torch.zeros((args.repeat_times, num_d, args.h_length))
        gpt_eval_corr_ZZ = torch.zeros((args.repeat_times, num_d, args.h_length))
        gpt_eval_corr_spin_dot = torch.zeros((args.repeat_times, num_d, args.h_length))
    if args.predict_model == "J1J2":
        gpt_eval_energy = torch.zeros((args.repeat_times, args.h_length))
        gpt_eval_corr_XX = torch.zeros((args.repeat_times, num_d, args.h_length))
        gpt_eval_corr_YY = torch.zeros((args.repeat_times, num_d, args.h_length))
        gpt_eval_corr_ZZ = torch.zeros((args.repeat_times, num_d, args.h_length))
        gpt_eval_corr_spin_dot = torch.zeros((args.repeat_times, num_d, args.h_length))
        gpt_eval_dimer_proxy = torch.zeros((args.repeat_times, args.h_length))
    if args.predict_model == "ANNNI":
        gpt_eval_energy = torch.zeros((args.repeat_times, args.h_length))
        gpt_eval_ZZ = torch.zeros((args.repeat_times, num_d, args.h_length))
        gpt_eval_X = torch.zeros((args.repeat_times, args.h_length))
        gpt_eval_sf_pi = torch.zeros((args.repeat_times, args.h_length))
        gpt_eval_sf_pi_over_2 = torch.zeros((args.repeat_times, args.h_length))
    # true_ZZ = torch.zeros(5, 101)
    # true_stringX = torch.zeros(5, 101)

    if args.predict_model == "TFI":
        print("Generating TFI h values...")
        hs = torch.linspace(0, 1, args.h_length)
    elif args.predict_model == "xxz":
        hs = torch.linspace(-2, 2, args.h_length)
        gpt_eval_energy = torch.zeros((args.repeat_times, args.h_length))
        gpt_eval_energy_derivative = torch.zeros((args.repeat_times, args.h_length))
        gpt_magnetization_z = torch.zeros((args.repeat_times, args.h_length))
        gpt_magnetization_s = torch.zeros((args.repeat_times, args.h_length))
        gpt_eval_corr_XX_single = torch.zeros((args.repeat_times, args.h_length))
        gpt_eval_corr_YY_single = torch.zeros((args.repeat_times, args.h_length))
        gpt_eval_corr_ZZ_single = torch.zeros((args.repeat_times, args.h_length))
        gpt_eval_energy_from_c = torch.full((args.repeat_times, args.h_length), float("nan"))
        gpt_eval_energy_derivative_from_c = torch.full((args.repeat_times, args.h_length), float("nan"))
        gpt_eval_corr_XX_from_c = torch.full((args.repeat_times, args.h_length), float("nan"))
        gpt_eval_corr_YY_from_c = torch.full((args.repeat_times, args.h_length), float("nan"))
        gpt_eval_corr_ZZ_from_c = torch.full((args.repeat_times, args.h_length), float("nan"))
        J = 1.0
    elif args.predict_model == "J1J2":
        hs = torch.linspace(0, 1, args.h_length)
    elif args.predict_model == "ANNNI":
        hs = torch.linspace(0, 2, args.h_length)
    else:
        hs = torch.linspace(-2,2, args.h_length)
    hs_numpy = hs.numpy()
    grouped_C = None
    grouped_single = None
    if args.multimodal:
        if generate_target == 'b':
            grouped_C = _group_condition_data(
                args.pair_condition_data_path,
                args.h_length,
                args.condition_group_size,
                "pair/C",
            )
        elif generate_target == 'c':
            grouped_single = _group_condition_data(
                args.single_condition_data_path,
                args.h_length,
                args.condition_group_size,
                "single",
            )
            if grouped_single is None:
                print("No --single_condition_data_path provided; using random single nseq conditions for c generation.")
        # C = C['data']
        # g_to_phys = {
        #     round(item['g'], 6): torch.tensor(item['product_vector'], dtype=torch.float32)
        #     for item in C
        # }
        # print("g_to_phys keys:", list(g_to_phys.keys()))

    
    # print(f"len(hs)={len(hs)}")


    for m in range(args.repeat_times):
        print(f"Experiment {m+1}/{args.repeat_times}")
        for j in tqdm.tqdm(range(args.h_length), desc=f"Experiment {m+1}/{args.repeat_times}", colour='blue'):
            # print(f"j={j}, len(hs)={len(hs)}")
            # print(f"lens of C: {len(grouped_C)}")
            h = hs[j]
            if generate_target == 'b' and args.multimodal:
                c_input = grouped_C[j]

                # print(f"the shape of c: {c.shape}")
            elif generate_target == 'c' and args.multimodal:
                c_input = grouped_single[j] if grouped_single is not None else None
            else:
                c_input = None
            generate_kwargs = {"gen_batch_size": args.gen_batch_size}
            if generate_func is generate_GPTmeasureoutput_nseq:
                generate_kwargs["pair_basis_mode"] = pair_basis_mode
            sqe, sqe_c = generate_func(
                h,
                model,
                args.num_qubits,
                args.sample_size_per_h,
                diff_steps,
                temperature,
                device,
                generate_target,
                c_input,
                args.confidence_decoding,
                **generate_kwargs,
            )
            print(f"Generated raw output shape: {sqe.shape}")
            print(f"Generated shape of sqe_c: {sqe_c.shape if sqe_c is not None else 'N/A'}")
            sqe = sqe[:, 1:].to(torch.long)
            sqe_c = sqe_c[:, 1:].to(torch.long) if sqe_c is not None else None
            # 避免打印超大 tensor（会拖慢并导致 GPU 同步）
            # print("b output:", sqe)
            # print("c output:", sqe_c)

            # Calculate different metrics
            if args.predict_model == 'TFI':
                for d_idx, d in enumerate(d_values):
                    gpt_eval_points[m, d_idx, j] = median_of_means_two(d, sqe, 10) # two point correlation function
                    gpt_eval_pointsX[m, d_idx, j] = median_of_means_X(d - 1, sqe, 10) # Xstring

                gpt_eval_points_energy[m, j] = median_of_means(h, sqe, 10) # ground energy

                if sqe_c is not None:
                    gpt_eval_points_from_c[m,j] = median_of_means_two_from_c(sqe_c, 10)
                    gpt_eval_pointsX_from_c[m,j] = median_of_means_X_from_c(sqe_c, 10)
                    gpt_eval_pointsX4_from_c[m,j] = median_of_means_X4_from_c(sqe_c, 10)

                torch.cuda.empty_cache()
            elif args.predict_model == "xxz":
                gpt_eval_energy[m, j], gpt_eval_energy_derivative[m, j] = median_of_means_xxz_energy_and_derivative(J, h, sqe, 10)
                gpt_magnetization_z[m, j], gpt_magnetization_s[m, j] = median_of_means_xxz_magnetization(sqe, 10)
                gpt_eval_corr_XX_single[m, j] = median_of_means_correlation(sqe, 1, pauli_type='X', num_parts=10)
                gpt_eval_corr_YY_single[m, j] = median_of_means_correlation(sqe, 1, pauli_type='Y', num_parts=10)
                gpt_eval_corr_ZZ_single[m, j] = median_of_means_correlation(sqe, 1, pauli_type='Z', num_parts=10)
                if sqe_c is not None:
                    (
                        gpt_eval_energy_from_c[m, j],
                        gpt_eval_energy_derivative_from_c[m, j],
                        gpt_eval_corr_XX_from_c[m, j],
                        gpt_eval_corr_YY_from_c[m, j],
                        gpt_eval_corr_ZZ_from_c[m, j],
                    ) = median_of_means_xxz_from_c(J, h, sqe_c, 10)

                torch.cuda.empty_cache()

            elif args.predict_model == "Heisenberg":
                gpt_eval_energy[m, j] = median_of_means_heisenberg(h, sqe, 10)

                for d_idx, d in enumerate(d_values):
                    gpt_eval_corr_XX[m, d_idx, j] = median_of_means_correlation(
                        sqe, d, pauli_type='X', num_parts=10
                    )
                    gpt_eval_corr_YY[m, d_idx, j] = median_of_means_correlation(
                        sqe, d, pauli_type='Y', num_parts=10
                    )
                    gpt_eval_corr_ZZ[m, d_idx, j] = median_of_means_correlation(
                        sqe, d, pauli_type='Z', num_parts=10
                    )       
                    # S_i·S_{i+d} = 1/4 (σ·σ)
                    gpt_eval_corr_spin_dot[m, d_idx, j] = 0.25 * (
                        gpt_eval_corr_XX[m, d_idx, j] + 
                        gpt_eval_corr_YY[m, d_idx, j] + 
                        gpt_eval_corr_ZZ[m, d_idx, j]
                    )

                torch.cuda.empty_cache()             
            elif args.predict_model == "J1J2":
                J2 = float(h)
                gpt_eval_energy[m, j] = median_of_means_j1j2_energy(args.J1, J2, sqe, 10)
                gpt_eval_dimer_proxy[m, j] = median_of_means_j1j2_dimer_proxy(sqe, 10)

                for d_idx, d in enumerate(d_values):
                    gpt_eval_corr_XX[m, d_idx, j] = median_of_means_correlation(
                        sqe, d, pauli_type='X', num_parts=10
                    )
                    gpt_eval_corr_YY[m, d_idx, j] = median_of_means_correlation(
                        sqe, d, pauli_type='Y', num_parts=10
                    )
                    gpt_eval_corr_ZZ[m, d_idx, j] = median_of_means_correlation(
                        sqe, d, pauli_type='Z', num_parts=10
                    )
                    gpt_eval_corr_spin_dot[m, d_idx, j] = (
                        gpt_eval_corr_XX[m, d_idx, j]
                        + gpt_eval_corr_YY[m, d_idx, j]
                        + gpt_eval_corr_ZZ[m, d_idx, j]
                    )

                torch.cuda.empty_cache()
            elif args.predict_model == "ANNNI":
                h_float = float(h)
                gpt_eval_energy[m, j] = median_of_means_annni_energy(
                    h_float, sqe, kappa=args.annni_kappa, J1=args.J1, num_parts=10
                )
                for d_idx, d in enumerate(d_values):
                    gpt_eval_ZZ[m, d_idx, j] = median_of_means_correlation(
                        sqe, d, pauli_type='Z', num_parts=10
                    )
                gpt_eval_X[m, j] = median_of_means_single_pauli(sqe, 'X', 10)
                gpt_eval_sf_pi[m, j] = median_of_means_annni_structure_factor_z(sqe, np.pi, 10)
                gpt_eval_sf_pi_over_2[m, j] = median_of_means_annni_structure_factor_z(sqe, np.pi / 2.0, 10)

                torch.cuda.empty_cache()

    print("Diffushadow generation complete...")

    # print("type of gpt_eval_points_energy:", type(gpt_eval_points_energy))
    # print("type of gpt_eval_points_from_c:", type(gpt_eval_points_from_c))

    # torch.save(gpt_eval_points, r'/home/fyp26lyh/QuantumLLADA/DshadowGPT/shadowGPTmy/Newarc_sym_6times_TFI_two_ZZ_d_gpt_es128_30w_MoM10_6_10000.pt')
    # torch.save(gpt_eval_pointsX, r'/home/fyp26lyh/QuantumLLADA/DshadowGPT/shadowGPTmy/Newarc_sym_6times_TFI_Xs_d_gpt_es128_30w_MoM10_6_10000.pt')
    # torch.save(gpt_eval_points_energy, r'/home/fyp26lyh/QuantumLLADA/DshadowGPT/shadowGPTmy/Newarc_sym_TFI_Energy_gpt_es128_30w_MoM10_6_10000.pt')
    # print("generated data saved.")

    if args.predict_model == "TFI":
        hs_gpt = torch.linspace(0, 1, args.h_length).numpy()  # (args.h_length,)
    elif args.predict_model == "Heisenberg":
        hs_gpt = torch.linspace(-1, 1, args.h_length).numpy()  # (args.h_length,)
    elif args.predict_model == "xxz":
        hs_gpt = torch.linspace(-2, 2, args.h_length).numpy()  # (args.h_length,)
    elif args.predict_model == "J1J2":
        hs_gpt = torch.linspace(0, 1, args.h_length).numpy()
    elif args.predict_model == "ANNNI":
        hs_gpt = torch.linspace(0, 2, args.h_length).numpy()

    # 取均值和标准差（6次实验）
    if args.predict_model == "TFI":
        zz_mean = gpt_eval_points.mean(dim=0).numpy()        # (5, args.h_length)
        zz_std = gpt_eval_points.std(dim=0).numpy()          # (5, args.h_length)

        xs_mean = gpt_eval_pointsX.mean(dim=0).numpy()       # (5, args.h_length)
        xs_std = gpt_eval_pointsX.std(dim=0).numpy()         # (5, args.h_length)

        energy_mean = gpt_eval_points_energy.mean(dim=0).numpy()  # (args.h_length,)
        energy_std = gpt_eval_points_energy.std(dim=0).numpy()    # (args.h_length,)

        zz_mean_from_c = gpt_eval_points_from_c.mean(dim=0).numpy()
        XX_mean_from_c = gpt_eval_pointsX_from_c.mean(dim=0).numpy()
        X4_mean_from_c = gpt_eval_pointsX4_from_c.mean(dim=0).numpy()

        print(f"shape of zz_mean_from_c: {zz_mean_from_c.shape}")
    
    elif args.predict_model == "Heisenberg":
        energy_mean = gpt_eval_energy.mean(dim=0).numpy()
        energy_std = gpt_eval_energy.std(dim=0).numpy()
        
        # 关联函数
        corr_XX_mean = gpt_eval_corr_XX.mean(dim=0).numpy()
        corr_XX_std = gpt_eval_corr_XX.std(dim=0).numpy()
        
        corr_YY_mean = gpt_eval_corr_YY.mean(dim=0).numpy()
        corr_YY_std = gpt_eval_corr_YY.std(dim=0).numpy()
        
        corr_ZZ_mean = gpt_eval_corr_ZZ.mean(dim=0).numpy()
        corr_ZZ_std = gpt_eval_corr_ZZ.std(dim=0).numpy()
        
        corr_spin_dot_mean = gpt_eval_corr_spin_dot.mean(dim=0).numpy()
        corr_spin_dot_std = gpt_eval_corr_spin_dot.std(dim=0).numpy()   

    elif args.predict_model == "xxz":
        energy_mean = gpt_eval_energy.mean(dim=0).numpy()
        energy_std = gpt_eval_energy.std(dim=0).numpy()

        energy_derivative_mean = gpt_eval_energy_derivative.mean(dim=0).numpy()
        energy_derivative_std = gpt_eval_energy_derivative.std(dim=0).numpy()

        magnetization_z_mean = gpt_magnetization_z.mean(dim=0).numpy()
        magnetization_z_std = gpt_magnetization_z.std(dim=0).numpy()

        magnetization_s_mean = gpt_magnetization_s.mean(dim=0).numpy()
        magnetization_s_std = gpt_magnetization_s.std(dim=0).numpy()

        corr_XX_mean_single = gpt_eval_corr_XX_single.mean(dim=0).numpy()
        corr_XX_std_single = gpt_eval_corr_XX_single.std(dim=0).numpy()
        corr_YY_mean_single = gpt_eval_corr_YY_single.mean(dim=0).numpy()
        corr_YY_std_single = gpt_eval_corr_YY_single.std(dim=0).numpy()
        corr_ZZ_mean_single = gpt_eval_corr_ZZ_single.mean(dim=0).numpy()
        corr_ZZ_std_single = gpt_eval_corr_ZZ_single.std(dim=0).numpy()

        energy_mean_from_c, energy_std_from_c = mean_std_ignore_nan(gpt_eval_energy_from_c)
        energy_derivative_mean_from_c, energy_derivative_std_from_c = mean_std_ignore_nan(
            gpt_eval_energy_derivative_from_c
        )
        corr_XX_mean_from_c, corr_XX_std_from_c = mean_std_ignore_nan(gpt_eval_corr_XX_from_c)
        corr_YY_mean_from_c, corr_YY_std_from_c = mean_std_ignore_nan(gpt_eval_corr_YY_from_c)
        corr_ZZ_mean_from_c, corr_ZZ_std_from_c = mean_std_ignore_nan(gpt_eval_corr_ZZ_from_c)
    elif args.predict_model == "J1J2":
        energy_mean = gpt_eval_energy.mean(dim=0).numpy()
        energy_std = gpt_eval_energy.std(dim=0).numpy()
        corr_XX_mean = gpt_eval_corr_XX.mean(dim=0).numpy()
        corr_XX_std = gpt_eval_corr_XX.std(dim=0).numpy()
        corr_YY_mean = gpt_eval_corr_YY.mean(dim=0).numpy()
        corr_YY_std = gpt_eval_corr_YY.std(dim=0).numpy()
        corr_ZZ_mean = gpt_eval_corr_ZZ.mean(dim=0).numpy()
        corr_ZZ_std = gpt_eval_corr_ZZ.std(dim=0).numpy()
        corr_spin_dot_mean = gpt_eval_corr_spin_dot.mean(dim=0).numpy()
        corr_spin_dot_std = gpt_eval_corr_spin_dot.std(dim=0).numpy()
        dimer_proxy_mean = gpt_eval_dimer_proxy.mean(dim=0).numpy()
        dimer_proxy_std = gpt_eval_dimer_proxy.std(dim=0).numpy()
    elif args.predict_model == "ANNNI":
        energy_mean = gpt_eval_energy.mean(dim=0).numpy()
        energy_std = gpt_eval_energy.std(dim=0).numpy()
        zz_mean = gpt_eval_ZZ.mean(dim=0).numpy()
        zz_std = gpt_eval_ZZ.std(dim=0).numpy()
        x_mean = gpt_eval_X.mean(dim=0).numpy()
        x_std = gpt_eval_X.std(dim=0).numpy()
        sf_pi_mean = gpt_eval_sf_pi.mean(dim=0).numpy()
        sf_pi_std = gpt_eval_sf_pi.std(dim=0).numpy()
        sf_pi_over_2_mean = gpt_eval_sf_pi_over_2.mean(dim=0).numpy()
        sf_pi_over_2_std = gpt_eval_sf_pi_over_2.std(dim=0).numpy()




    num_qubits = args.num_qubits

    if args.exact_value:
        exact_ctx = _ensure_exact_cache_and_load(args)
    else:
        exact_ctx = None

    if args.predict_model == "TFI":
        h_values = exact_ctx["h_values"] if exact_ctx is not None else None
        ZZ_curves_all = exact_ctx["ZZ_curves"] if exact_ctx is not None else None
        Xs_curves_all = exact_ctx["Xs_curves"] if exact_ctx is not None else None
        ZZ_curves = _select_distance_rows(ZZ_curves_all, d_values, "ZZ_curves")
        Xs_curves = _select_distance_rows(Xs_curves_all, d_values, "Xs_curves")
        Energy = exact_ctx["Energy"] if exact_ctx is not None else None
    elif args.predict_model == "Heisenberg":
        h_values = exact_ctx["h_values"] if exact_ctx is not None else None
        exact_value = exact_ctx["exact_value"] if exact_ctx is not None else None
        if exact_value is not None:
            exact_value = exact_value.copy()
            exact_value["correlations_XX"] = _select_distance_rows(
                exact_value["correlations_XX"], d_values, "correlations_XX"
            )
            exact_value["correlations_YY"] = _select_distance_rows(
                exact_value["correlations_YY"], d_values, "correlations_YY"
            )
            exact_value["correlations_ZZ"] = _select_distance_rows(
                exact_value["correlations_ZZ"], d_values, "correlations_ZZ"
            )
            exact_value["correlations_spin_dot"] = _select_distance_rows(
                exact_value["correlations_spin_dot"], d_values, "correlations_spin_dot"
            )
    elif args.predict_model == "xxz":
        delta_values = exact_ctx["delta_values"] if exact_ctx is not None else None
        exact_value = exact_ctx["exact_value"] if exact_ctx is not None else None
    elif args.predict_model == "J1J2":
        J2_values = exact_ctx["J2_values"] if exact_ctx is not None else None
        exact_value = exact_ctx["exact_value"] if exact_ctx is not None else None
        if exact_value is not None:
            exact_value = exact_value.copy()
            exact_value["correlations_XX"] = _select_distance_rows(
                exact_value["correlations_XX"], d_values, "correlations_XX"
            )
            exact_value["correlations_YY"] = _select_distance_rows(
                exact_value["correlations_YY"], d_values, "correlations_YY"
            )
            exact_value["correlations_ZZ"] = _select_distance_rows(
                exact_value["correlations_ZZ"], d_values, "correlations_ZZ"
            )
            exact_value["correlations_spin_dot"] = _select_distance_rows(
                exact_value["correlations_spin_dot"], d_values, "correlations_spin_dot"
            )
    elif args.predict_model == "ANNNI":
        h_values = exact_ctx["h_values"] if exact_ctx is not None else None
        exact_value = exact_ctx["exact_value"] if exact_ctx is not None else None
        if exact_value is not None:
            exact_value = exact_value.copy()
            exact_value["ZZ_curves"] = _select_distance_rows(
                exact_value["ZZ_curves"], d_values, "ZZ_curves"
            )

    print("True values calculation complete...")







    # ====================================================
    # 6️⃣ 保存所有绘图数据到文件（不绘图）
    # ====================================================

    # 准备要保存的数据字典
    if args.predict_model == "TFI":
        data_to_save = {
            'd_values': np.array(d_values, dtype=int),
            # GPT 生成结果（6次实验的原始数据）
            'gpt_eval_points': gpt_eval_points.numpy(),           # (repeat_times, len(d_values), args.h_length)
            'gpt_eval_pointsX': gpt_eval_pointsX.numpy(),         # (repeat_times, len(d_values), args.h_length)
            'gpt_eval_points_energy': gpt_eval_points_energy.numpy(),  # (6, args.h_length)  能量
            'gpt_eval_points_from_c': gpt_eval_points_from_c.numpy(),

            # GPT 统计值（均值和标准差）
            'zz_mean': zz_mean,           # (len(d_values), args.h_length)
            'zz_std': zz_std,             # (len(d_values), args.h_length)
            'xs_mean': xs_mean,           # (len(d_values), args.h_length)
            'xs_std': xs_std,             # (len(d_values), args.h_length)
            'energy_mean': energy_mean,   # (args.h_length,)
            'energy_std': energy_std,     # (args.h_length,)
            'zz_mean_from_c': zz_mean_from_c, #(args.h_length,)
            'XX_mean_from_c': XX_mean_from_c, #(args.h_length,)
            'X4_mean_from_c': X4_mean_from_c, #(args.h_length,

            # 真实值（101个h点）
            'h_values_true': h_values,    # (101,) 0.0 ~ 1.0
            'ZZ_curves': ZZ_curves,       # (len(d_values), 101)
            'Xs_curves': Xs_curves,       # (len(d_values), 101)
            'Energy_true': Energy,        # (101,)  真实能量

            # GPT 采样用的 h 点
            'hs_gpt': hs_gpt,             # (args.h_length,) 0.0 ~ 1.0
        }
    elif args.predict_model == "Heisenberg":
        data_to_save = {
            # 实验参数
            'Js': hs_gpt,
            'N': num_qubits,
            'd_values': np.array(d_values, dtype=int),
            
            # GPT估计结果（原始数据）
            'gpt_eval_energy_raw': gpt_eval_energy.numpy(),
            'gpt_eval_corr_XX_raw': gpt_eval_corr_XX.numpy(),
            'gpt_eval_corr_YY_raw': gpt_eval_corr_YY.numpy(),
            'gpt_eval_corr_ZZ_raw': gpt_eval_corr_ZZ.numpy(),
            'gpt_eval_corr_spin_dot_raw': gpt_eval_corr_spin_dot.numpy(),
            
            # GPT统计量
            'energy_mean': energy_mean,
            'energy_mean': energy_mean,
            'energy_std': energy_std,
            'corr_XX_mean': corr_XX_mean,
            'corr_XX_std': corr_XX_std,
            'corr_YY_mean': corr_YY_mean,
            'corr_YY_std': corr_YY_std,
            'corr_ZZ_mean': corr_ZZ_mean,
            'corr_ZZ_std': corr_ZZ_std,
            'corr_spin_dot_mean': corr_spin_dot_mean,
            'corr_spin_dot_std': corr_spin_dot_std,
            
            # 精确解
            'exact_energy': exact_value['energy'] if exact_value is not None else None,
            'exact_corr_XX': exact_value['correlations_XX'] if exact_value is not None else None,
            'exact_corr_YY': exact_value['correlations_YY'] if exact_value is not None else None,
            'exact_corr_ZZ': exact_value['correlations_ZZ'] if exact_value is not None else None,
            'exact_corr_spin_dot': exact_value['correlations_spin_dot'] if exact_value is not None else None,
            
            # 用于画图的密集J值
            'J_values_dense': h_values
        }        
    elif args.predict_model == "xxz":
        data_to_save = {
            # 实验参数
            'Deltas': hs_gpt,
            'J_xy': J,
            'N': num_qubits,
            
            # 阴影估计结果（原始数据）
            'shadow_eval_energy_raw': gpt_eval_energy.numpy(),
            'shadow_eval_energy_derivative_raw': gpt_eval_energy_derivative.numpy(),
            'shadow_eval_magnetization_z_raw': gpt_magnetization_z.numpy(),
            'shadow_eval_magnetization_s_raw': gpt_magnetization_s.numpy(),
            'shadow_eval_corr_XX_single_raw': gpt_eval_corr_XX_single.numpy(),
            'shadow_eval_corr_YY_single_raw': gpt_eval_corr_YY_single.numpy(),
            'shadow_eval_corr_ZZ_single_raw': gpt_eval_corr_ZZ_single.numpy(),
            'shadow_eval_energy_from_c_raw': gpt_eval_energy_from_c.numpy(),
            'shadow_eval_energy_derivative_from_c_raw': gpt_eval_energy_derivative_from_c.numpy(),
            'shadow_eval_corr_XX_from_c_raw': gpt_eval_corr_XX_from_c.numpy(),
            'shadow_eval_corr_YY_from_c_raw': gpt_eval_corr_YY_from_c.numpy(),
            'shadow_eval_corr_ZZ_from_c_raw': gpt_eval_corr_ZZ_from_c.numpy(),
            'energy_mean': energy_mean,
            
            # 阴影统计量
            'energy_mean': energy_mean,
            'energy_std': energy_std,
            'energy_derivative_mean': energy_derivative_mean,
            'energy_derivative_std': energy_derivative_std,
            'magnetization_z_mean': magnetization_z_mean,
            'magnetization_z_std': magnetization_z_std,
            'magnetization_s_mean': magnetization_s_mean,
            'magnetization_s_std': magnetization_s_std,
            'corr_XX_mean_single': corr_XX_mean_single,
            'corr_XX_std_single': corr_XX_std_single,
            'corr_YY_mean_single': corr_YY_mean_single,
            'corr_YY_std_single': corr_YY_std_single,
            'corr_ZZ_mean_single': corr_ZZ_mean_single,
            'corr_ZZ_std_single': corr_ZZ_std_single,
            'energy_mean_from_c': energy_mean_from_c,
            'energy_std_from_c': energy_std_from_c,
            'energy_derivative_mean_from_c': energy_derivative_mean_from_c,
            'energy_derivative_std_from_c': energy_derivative_std_from_c,
            'corr_XX_mean_from_c': corr_XX_mean_from_c,
            'corr_XX_std_from_c': corr_XX_std_from_c,
            'corr_YY_mean_from_c': corr_YY_mean_from_c,
            'corr_YY_std_from_c': corr_YY_std_from_c,
            'corr_ZZ_mean_from_c': corr_ZZ_mean_from_c,
            'corr_ZZ_std_from_c': corr_ZZ_std_from_c,
            'pair_basis_mode': np.array(pair_basis_mode),
            
            # 精确解
            'exact_energy': exact_value['energy'] if exact_value is not None else None,
            'exact_energy': exact_value['energy'] if exact_value is not None else None,
            'exact_magnetization_z': exact_value['magnetization_z'] if exact_value is not None else None,
            'exact_magnetization_s': exact_value['magnetization_staggered'] if exact_value is not None else None,
            'exact_energy_derivative_numeric': exact_value['energy_derivative_numeric'] if exact_value is not None else None,
            'exact_energy_derivative_analytic': exact_value['energy_derivative_analytic'] if exact_value is not None else None,
            'exact_energy_second_derivative': exact_value['energy_second_derivative'] if exact_value is not None else None,
            'exact_zz_correlation': exact_value['zz_correlation'] if exact_value is not None else None,
            'exact_energy_total': exact_value['energy_total'] if exact_value is not None else None,
            'exact_energy_per_bond': exact_value['energy_per_bond'] if exact_value is not None else None,
            'exact_energy_derivative_per_bond': exact_value['energy_derivative_per_bond'] if exact_value is not None else None,
            'exact_energy_derivative_total': exact_value['energy_derivative_total'] if exact_value is not None else None,
            'exact_single_energy': exact_value['single_energy'] if exact_value is not None else None,
            'exact_single_energy_derivative': exact_value['single_energy_derivative'] if exact_value is not None else None,
            'exact_single_energy_per_bond': exact_value['single_energy_per_bond'] if exact_value is not None else None,
            'exact_single_energy_derivative_per_bond': exact_value['single_energy_derivative_per_bond'] if exact_value is not None else None,
            'exact_single_energy_total': exact_value['single_energy_total'] if exact_value is not None else None,
            'exact_single_energy_derivative_total': exact_value['single_energy_derivative_total'] if exact_value is not None else None,
            'exact_single_corr_XX': exact_value['single_corr_XX'] if exact_value is not None else None,
            'exact_single_corr_YY': exact_value['single_corr_YY'] if exact_value is not None else None,
            'exact_single_corr_ZZ': exact_value['single_corr_ZZ'] if exact_value is not None else None,
            'exact_single_local_Sz': exact_value['single_local_Sz'] if exact_value is not None else None,
            'exact_single_magnetization_z': exact_value['single_magnetization_z'] if exact_value is not None else None,
            'exact_single_magnetization_s': exact_value['single_magnetization_staggered'] if exact_value is not None else None,
            'exact_c_energy': exact_value['c_energy'] if exact_value is not None else None,
            'exact_c_energy_derivative': exact_value['c_energy_derivative'] if exact_value is not None else None,
            'exact_c_energy_per_bond': exact_value['c_energy_per_bond'] if exact_value is not None else None,
            'exact_c_energy_derivative_per_bond': exact_value['c_energy_derivative_per_bond'] if exact_value is not None else None,
            'exact_c_total_energy': exact_value['c_total_energy'] if exact_value is not None else None,
            'exact_c_total_energy_derivative': exact_value['c_total_energy_derivative'] if exact_value is not None else None,
            'exact_c_corr_XX': exact_value['c_corr_XX'] if exact_value is not None else None,
            'exact_c_corr_YY': exact_value['c_corr_YY'] if exact_value is not None else None,
            'exact_c_corr_ZZ': exact_value['c_corr_ZZ'] if exact_value is not None else None,
            'Delta_values_dense': delta_values,
            'special_points': exact_value['special_points'] if exact_value is not None else None,
            
            # 用于画图的密集Δ值
            'Delta_values_dense': delta_values,
            
            # 特殊点信息
            'special_points': exact_value['special_points'] if exact_value is not None else None,
        }
    elif args.predict_model == "J1J2":
        data_to_save = {
            'J2s': hs_gpt,
            'J1': args.J1,
            'N': num_qubits,
            'd_values': np.array(d_values, dtype=int),
            'shadow_eval_energy_raw': gpt_eval_energy.numpy(),
            'shadow_eval_corr_XX_raw': gpt_eval_corr_XX.numpy(),
            'shadow_eval_corr_YY_raw': gpt_eval_corr_YY.numpy(),
            'shadow_eval_corr_ZZ_raw': gpt_eval_corr_ZZ.numpy(),
            'shadow_eval_corr_spin_dot_raw': gpt_eval_corr_spin_dot.numpy(),
            'shadow_eval_dimer_proxy_raw': gpt_eval_dimer_proxy.numpy(),
            'energy_mean': energy_mean,
            'energy_std': energy_std,
            'corr_XX_mean': corr_XX_mean,
            'corr_XX_std': corr_XX_std,
            'corr_YY_mean': corr_YY_mean,
            'corr_YY_std': corr_YY_std,
            'corr_ZZ_mean': corr_ZZ_mean,
            'corr_ZZ_std': corr_ZZ_std,
            'corr_spin_dot_mean': corr_spin_dot_mean,
            'corr_spin_dot_std': corr_spin_dot_std,
            'dimer_proxy_mean': dimer_proxy_mean,
            'dimer_proxy_std': dimer_proxy_std,
            'exact_energy': exact_value['energy'] if exact_value is not None else None,
            'exact_corr_XX': exact_value['correlations_XX'] if exact_value is not None else None,
            'exact_corr_YY': exact_value['correlations_YY'] if exact_value is not None else None,
            'exact_corr_ZZ': exact_value['correlations_ZZ'] if exact_value is not None else None,
            'exact_corr_spin_dot': exact_value['correlations_spin_dot'] if exact_value is not None else None,
            'exact_dimer_proxy': exact_value['dimer_proxy'] if exact_value is not None else None,
            'J2_values_dense': J2_values,
        }
    elif args.predict_model == "ANNNI":
        data_to_save = {
            'hs': hs_gpt,
            'J1': args.J1,
            'kappa': args.annni_kappa,
            'N': num_qubits,
            'd_values': np.array(d_values, dtype=int),
            'shadow_eval_energy_raw': gpt_eval_energy.numpy(),
            'shadow_eval_ZZ_raw': gpt_eval_ZZ.numpy(),
            'shadow_eval_X_raw': gpt_eval_X.numpy(),
            'shadow_eval_structure_factor_pi_raw': gpt_eval_sf_pi.numpy(),
            'shadow_eval_structure_factor_pi_over_2_raw': gpt_eval_sf_pi_over_2.numpy(),
            'energy_mean': energy_mean,
            'energy_std': energy_std,
            'zz_mean': zz_mean,
            'zz_std': zz_std,
            'x_mean': x_mean,
            'x_std': x_std,
            'structure_factor_pi_mean': sf_pi_mean,
            'structure_factor_pi_std': sf_pi_std,
            'structure_factor_pi_over_2_mean': sf_pi_over_2_mean,
            'structure_factor_pi_over_2_std': sf_pi_over_2_std,
            'exact_energy': exact_value['energy'] if exact_value is not None else None,
            'exact_ZZ_curves': exact_value['ZZ_curves'] if exact_value is not None else None,
            'exact_X_magnetization': exact_value['X_magnetization'] if exact_value is not None else None,
            'exact_structure_factor_pi': exact_value['structure_factor_pi'] if exact_value is not None else None,
            'exact_structure_factor_pi_over_2': exact_value['structure_factor_pi_over_2'] if exact_value is not None else None,
            'h_values_dense': h_values,
        }


    save_data_path = args.save_data_path
    # 保存为 .npz 文件（压缩可选：compress=True）
    np.savez(save_data_path, **data_to_save)

    print("✅ 所有绘图数据已保存")
    print(f"💾 保存路径: {save_data_path}")
    print(f"📊 数据包含: gpt_eval_points, gpt_eval_pointsX, energy, true ZZ/Xs/Energy 等")


    # 初始化误差字典
    error_metrics = {
        'model': args.model_path,
        'save_data_path': save_data_path,
        'd_values': d_values,
        'zz_errors': {},      # Two-point correlation errors
        'xs_errors': {},      # X-string errors  
        'energy_errors': {},  # Energy errors
        'zz_from_c_errors':{},
        'XX_from_c_errors':{},
        'X4_from_c_errors':{},

    }
    
    if args.predict_model == "TFI" and args.exact_value:
        # 计算每个d的误差
        for d_idx, d in enumerate(d_values):
            zz_errors, zz_mse = calculate_errors(hs_gpt, zz_mean[d_idx], h_values, ZZ_curves[d_idx])
            xs_errors, xs_mse = calculate_errors(hs_gpt, xs_mean[d_idx], h_values, Xs_curves[d_idx])
            
            error_metrics['zz_errors'][f'd{d}'] = {
                'absolute_errors': zz_errors.tolist(),
                'mse_values': zz_mse.tolist(),
                'mean_absolute_error': float(np.mean(zz_errors)),
                'mean_squared_error': float(np.mean(zz_mse)),
                'max_absolute_error': float(np.max(zz_errors))
            }
            
            error_metrics['xs_errors'][f'l{d}'] = {
                'absolute_errors': xs_errors.tolist(),
                'mse_values': xs_mse.tolist(), 
                'mean_absolute_error': float(np.mean(xs_errors)),
                'mean_squared_error': float(np.mean(xs_mse)),
                'max_absolute_error': float(np.max(xs_errors))
            }
        
        # 计算能量误差
        energy_errors, energy_mse = calculate_errors(hs_gpt, energy_mean, h_values, Energy)
        error_metrics['energy_errors'] = {
            'absolute_errors': energy_errors.tolist(),
            'mse_values': energy_mse.tolist(),
            'mean_absolute_error': float(np.mean(energy_errors)),
            'mean_squared_error': float(np.mean(energy_mse)),
            'max_absolute_error': float(np.max(energy_errors))
        }
        if args.multimodal:
            zz_from_c_errors, zz_from_c_mse = calculate_errors(hs_gpt, zz_mean_from_c, h_values, ZZ_curves_all[0]) # 这里以d=1的ZZ作为参考
            error_metrics['zz_from_c_errors'] = {
                'absolute_errors': zz_from_c_errors.tolist(),
                'mse_values': zz_from_c_mse.tolist(),
                'mean_absolute_error': float(np.mean(zz_from_c_errors)),
                'mean_squared_error': float(np.mean(zz_from_c_mse)),
                'max_absolute_error': float(np.max(zz_from_c_errors))
            }
            XX_from_c_errors, XX_from_c_mse = calculate_errors(hs_gpt, XX_mean_from_c, h_values, Xs_curves_all[1]) # 这里以l=2的Xs作为参考
            X4_from_c_errors, X4_from_c_mse = calculate_errors(hs_gpt, X4_mean_from_c, h_values, Xs_curves_all[3]) # 这里以l=4的Xs作为参考
            error_metrics['XX_from_c_errors'] = {
                'absolute_errors': XX_from_c_errors.tolist(),
                'mse_values': XX_from_c_mse.tolist(),
                'mean_absolute_error': float(np.mean(XX_from_c_errors)),
                'mean_squared_error': float(np.mean(XX_from_c_mse)),
                'max_absolute_error': float(np.max(XX_from_c_errors))
            }
            error_metrics['X4_from_c_errors'] = {
                'absolute_errors': X4_from_c_errors.tolist(),
                'mse_values': X4_from_c_mse.tolist(),
                'mean_absolute_error': float(np.mean(X4_from_c_errors)),
                'mean_squared_error': float(np.mean(X4_from_c_mse)),
                'max_absolute_error': float(np.max(X4_from_c_errors))
            }




    # 保存路径
    save_error_path = r"/home/fyp26lyh/QuantumLLADA/data/eval_results/error_metrics_TFI_model_eval.json"





    #save error metrics to json file
    if os.path.exists(save_error_path):
        with open(save_error_path, 'r') as f:
            try:
                logs = json.load(f)  # 已有的训练日志列表
            except json.JSONDecodeError:
                logs = []  # 如果文件为空或损坏，从空列表开始
    else:
        logs = []  # 文件不存在，初始化为空列表    
    logs.append(error_metrics)
    with open(save_error_path, 'w') as f:
        json.dump(logs, f, indent=4)

        print(f"✅ Error appended to: {save_error_path}")


if __name__ == "__main__":
    main()


# # ====================================================
# # 3️⃣ 设置颜色（按你的要求）
# # ====================================================

# # 绿色系：ZZ（d=1~5）
# green_colors = ['#e6ffe6', '#b3f0b3', '#66cc66', '#339933', '#006600']
# # 蓝色系：Xs（l=1~5）
# blue_colors = ['#e6f2ff', '#b3d9ff', '#66b2ff', '#3385ff', '#0059e6']
# # 红色系：能量（统一用中间红）
# red_color = '#d73027'  # 强烈红，适合能量

# # ====================================================
# # 4️⃣ 绘图：三联图（竖版）
# # ====================================================

# plt.rcParams['font.size'] = 12
# plt.rcParams['axes.labelsize'] = 12
# plt.rcParams['axes.titlesize'] = 14
# plt.rcParams['lines.linewidth'] = 2.5
# plt.rcParams['axes.xmargin'] = 0
# plt.rcParams['axes.ymargin'] = 0

# fig, axes = plt.subplots(3, 1, figsize=(8, 12))

# # -------------------------------
# # (a) Two-point Correlations <Z_i Z_{i+d}>
# # -------------------------------
# for d in range(5):
#     axes[0].plot(h_values, ZZ_curves[d], color=green_colors[d], linewidth=2.5, alpha=0.9, label=f'd={d+1} True')
#     axes[0].scatter(hs_gpt, zz_mean[d], color=green_colors[d], alpha=0.8, label=f'd={d+1} GPT')

# axes[0].set_xlabel(r'External field $h$')
# axes[0].set_ylabel(r'$\langle Z_i Z_{i+d} \rangle$')
# axes[0].set_title('(a) Two-point correlations', pad=10)
# axes[0].legend(title=r'$d$', loc='upper right', frameon=True, ncol=2, fontsize=10, title_fontsize=11)
# axes[0].grid(True, alpha=0.3)
# axes[0].set_xlim(0, 1)
# axes[0].set_ylim(-1.0, 1.0)

# # -------------------------------
# # (b) X-string Order <X_i \cdots X_{i+l-1}>
# # -------------------------------
# for l in range(5):
#     axes[1].plot(h_values, Xs_curves[l], color=blue_colors[l], linewidth=2.5, alpha=0.9, label=f'l={l+1} True')
#     axes[1].scatter(hs_gpt, xs_mean[l], color=blue_colors[l], alpha=0.8, label=f'l={l+1} GPT')

# axes[1].set_xlabel(r'External field $h$')
# axes[1].set_ylabel(r'$\langle \prod_{k=i}^{i+l-1} X_k \rangle$')
# axes[1].set_title('(b) String order parameters', pad=10)
# axes[1].legend(title=r'$l$', loc='upper right', frameon=True, ncol=2, fontsize=10, title_fontsize=11)
# axes[1].grid(True, alpha=0.3)
# axes[1].set_xlim(0, 1)
# axes[1].set_ylim(-1.0, 1.0)

# # -------------------------------
# # (c) Ground State Energy
# # -------------------------------
# axes[2].plot(h_values, Energy, color=red_color, linewidth=2.5, alpha=0.9, label='Exact')
# axes[2].scatter(hs_gpt, energy_mean, color=red_color, alpha=0.8, label='GPT Prediction')

# axes[2].set_xlabel(r'External field $h$')
# axes[2].set_ylabel(r'$E_0/N$')
# axes[2].set_title('(c) Ground state energy', pad=10)
# axes[2].legend(loc='lower right', frameon=True)
# axes[2].grid(True, alpha=0.3)
# axes[2].set_xlim(0, 1)

# # ====================================================
# # 5️⃣ 保存图像
# # ====================================================

# plt.tight_layout()
# save_path = r'/home/fyp26lyh/QuantumLLADA/data/figure2_exact_vs_gpt_Qmodel_mini_wo_amask61_6_10000.png'
# plt.savefig(save_path, dpi=300, bbox_inches='tight')

# print("✅ 已生成对比图：真实值 vs GPT预测")
# print(f"💾 保存为: {save_path}")
