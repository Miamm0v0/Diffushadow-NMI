import argparse
import json
import math
import os
import random

import torch
import tqdm

from Qdmodel_oseq import QuantumDiffusionModel_oseq
from Qdmodel_oseq_rope import QuantumDiffusionModel_oseq_rope
from Qdmodel_oseq_rope_gnn import QuantumDiffusionModel_oseq_rope_gnn
from Qdmodel_oseq_rope_sharepos import QuantumDiffusionModel_oseq_rope_sharepos
from test_1 import generate_oseq_batch


MASK_TOKEN_ID = -1.0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate oseq measurement JSON data for train_oseq.py"
    )

    parser.add_argument(
        "--model_type",
        type=str,
        default="oseq_rope_gnn",
        choices=["oseq", "oseq_rope", "oseq_rope_sharepos", "oseq_rope_gnn"],
        help="Model architecture. Must match the checkpoint.",
    )
    parser.add_argument("--model_path", type=str, required=True, help="Checkpoint path.")
    parser.add_argument("--save_data_path", type=str, required=True, help="Output JSON path.")
    parser.add_argument("--device", type=str, default=None, help="cuda/cpu. Defaults to cuda if available.")

    parser.add_argument("--num_qubits", "--num-qubits", dest="num_qubits", type=int, default=10, help="Number of qubits to generate.")
    parser.add_argument("--sample_size_per_h", type=int, default=10000, help="Samples per h/g value.")
    parser.add_argument("--gen_batch_size", type=int, default=256, help="Generation batch size.")
    parser.add_argument("--diffusion_steps", type=int, default=2, help="Diffusion decoding steps.")
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (>0): lower is sharper, higher is flatter.",
    )
    parser.add_argument(
        "--greedy",
        action="store_true",
        help="Use argmax decoding instead of Bernoulli sampling; temperature then has no effect.",
    )

    parser.add_argument(
        "--predict_model",
        type=str,
        default="TFI",
        choices=["TFI", "Heisenberg", "xxz", "J1J2", "ANNNI"],
        help="Only used to choose the default scalar-condition range.",
    )
    parser.add_argument("--j1", "--J1", dest="j1", type=float, default=1.0, help="J1 coupling for J1J2/ANNNI metadata consistency.")
    parser.add_argument(
        "--scan",
        choices=["kappa", "h"],
        default="kappa",
        help="ANNNI only: which scalar parameter is written in the first data column, matching generate_j1j2_annni_dataset.py.",
    )
    parser.add_argument("--h", type=float, default=0.6, help="ANNNI only: fixed transverse field when --scan kappa.")
    parser.add_argument("--kappa", type=float, default=0.4, help="ANNNI only: fixed frustration when --scan h.")
    parser.add_argument("--h_length", type=int, default=41, help="Number of scalar condition values.")
    parser.add_argument("--h_min", type=float, default=None, help="Override scalar condition range minimum.")
    parser.add_argument("--h_max", type=float, default=None, help="Override scalar condition range maximum.")
    parser.add_argument(
        "--h_values",
        "--params",
        dest="h_values",
        type=float,
        nargs="*",
        default=None,
        help="Explicit scalar condition values, same role as --params in generate_j1j2_annni_dataset.py. If set, h_min/h_max/h_length are ignored.",
    )
    parser.add_argument(
        "--sample_size_per_param",
        "--samples-per-param",
        dest="sample_size_per_param",
        type=int,
        default=None,
        help="Alias for --sample_size_per_h, useful for J1J2/ANNNI naming.",
    )

    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden dimension.")
    parser.add_argument("--layer_num", type=int, default=4, help="Number of transformer layers.")
    parser.add_argument("--head_num", type=int, default=8, help="Number of attention heads.")
    parser.add_argument("--max_seq_len", type=int, default=4096, help="RoPE max_position_embeddings.")
    parser.add_argument(
        "--max_N",
        type=int,
        default=None,
        help="Max qubit count for RoPE models. Defaults to num_qubits.",
    )
    parser.add_argument(
        "--rope_scaling_type",
        type=str,
        default="none",
        choices=["none", "linear", "dynamic", "ntk"],
    )
    parser.add_argument("--rope_scaling_factor", type=float, default=1.0)
    parser.add_argument("--rope_theta", type=float, default=10000.0)
    parser.add_argument("--num_global_heads", type=int, default=1)
    parser.add_argument("--local_window_radius", type=int, default=2)

    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle samples before saving.")
    parser.add_argument(
        "--checkpoint_non_strict",
        action="store_true",
        help="Load checkpoint with strict=False.",
    )

    return parser.parse_args()


def safe_torch_load(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def looks_like_state_dict(obj):
    return isinstance(obj, dict) and len(obj) > 0 and all(
        isinstance(value, torch.Tensor) for value in obj.values()
    )


def extract_state_dict(loaded):
    if not isinstance(loaded, dict):
        return loaded
    for key in ("model_state_dict", "merged_state_dict", "state_dict", "model", "net", "network"):
        value = loaded.get(key)
        if isinstance(value, dict):
            return value
    if looks_like_state_dict(loaded):
        return loaded
    raise ValueError("Checkpoint does not contain a model state_dict.")


def strip_module_prefix(state_dict):
    if not isinstance(state_dict, dict) or not state_dict:
        return state_dict
    if all(isinstance(key, str) and key.startswith("module.") for key in state_dict.keys()):
        return {key[len("module.") :]: value for key, value in state_dict.items()}
    return state_dict


def build_model(args):
    if args.model_type == "oseq":
        if args.num_qubits != 10:
            raise ValueError("model_type=oseq is fixed to 10 qubits. Use a RoPE model for other sizes.")
        return QuantumDiffusionModel_oseq(
            hidden_dim=args.hidden_dim,
            num_layers=args.layer_num,
            head_count=args.head_num,
        )

    rope_st = None if args.rope_scaling_type == "none" else args.rope_scaling_type
    max_n = args.max_N if args.max_N is not None else args.num_qubits

    if args.model_type == "oseq_rope":
        return QuantumDiffusionModel_oseq_rope(
            hidden_dim=args.hidden_dim,
            num_layers=args.layer_num,
            head_count=args.head_num,
            max_seq_len=args.max_seq_len,
            max_N=max_n,
            rope_scaling_type=rope_st,
            rope_scaling_factor=args.rope_scaling_factor,
            rope_theta=args.rope_theta,
        )
    if args.model_type == "oseq_rope_sharepos":
        return QuantumDiffusionModel_oseq_rope_sharepos(
            hidden_dim=args.hidden_dim,
            num_layers=args.layer_num,
            head_count=args.head_num,
            max_seq_len=args.max_seq_len,
            max_N=max_n,
            rope_scaling_type=rope_st,
            rope_scaling_factor=args.rope_scaling_factor,
            rope_theta=args.rope_theta,
        )
    if args.model_type == "oseq_rope_gnn":
        return QuantumDiffusionModel_oseq_rope_gnn(
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

    raise ValueError(f"Unknown model_type: {args.model_type}")


def load_model(args, device):
    model = build_model(args).to(device)
    loaded = safe_torch_load(args.model_path, map_location=device)
    state_dict = strip_module_prefix(extract_state_dict(loaded))
    if args.model_type == "oseq_rope_gnn" and isinstance(state_dict, dict):
        state_dict.pop("position_embedding.weight", None)
    incompatible = model.load_state_dict(state_dict, strict=not args.checkpoint_non_strict)
    if args.checkpoint_non_strict:
        if incompatible.missing_keys:
            print(f"Missing keys: {incompatible.missing_keys}")
        if incompatible.unexpected_keys:
            print(f"Unexpected keys: {incompatible.unexpected_keys}")
    model.eval()
    return model


def default_h_range(args):
    if args.predict_model == "TFI":
        return 0.0, 1.0
    if args.predict_model == "J1J2":
        # generate_j1j2_annni_dataset.py writes alpha = J2 / J1 in column 0.
        return 0.0, 1.0
    if args.predict_model == "ANNNI":
        # generate_j1j2_annni_dataset.py has no implicit params, but the examples
        # and current training data scan either kappa or h over 0..1.
        return 0.0, 1.0
    return -2.0, 2.0


def condition_name(args):
    if args.predict_model == "J1J2":
        return "alpha"
    if args.predict_model == "ANNNI":
        return args.scan
    if args.predict_model == "xxz":
        return "Delta"
    if args.predict_model == "TFI":
        return "g"
    return "h"


def get_h_values(args, device):
    if args.h_values:
        return torch.tensor(args.h_values, dtype=torch.float32, device=device)

    default_min, default_max = default_h_range(args)
    h_min = default_min if args.h_min is None else args.h_min
    h_max = default_max if args.h_max is None else args.h_max
    return torch.linspace(float(h_min), float(h_max), int(args.h_length), device=device)


def generate_oseq_measurements(
    h,
    model,
    num_qubits,
    sample_size,
    diffusion_steps,
    temperature,
    device,
    gen_batch_size,
    use_sampling,
    param_label="h",
):
    length = 1 + 2 * num_qubits
    outputs = []
    remaining = int(sample_size)
    r_positions = torch.arange(1, length, 2, device=device)
    b_positions = torch.arange(2, length, 2, device=device)

    pbar = tqdm.tqdm(
        total=remaining,
        desc=f"Generating {param_label}={float(h):.6g}",
        unit="sample",
        ncols=100,
        colour="green",
    )

    while remaining > 0:
        batch_size = min(int(gen_batch_size), remaining)
        prompt = torch.full((batch_size, length, 1), MASK_TOKEN_ID, device=device)
        prompt[:, 0, 0] = float(h)

        basis_values = torch.randint(2, 5, (batch_size, num_qubits), device=device).float()
        prompt[:, r_positions, 0] = basis_values

        with torch.no_grad():
            completed = generate_oseq_batch(
                model=model,
                prompt=prompt,
                steps=diffusion_steps,
                temperature=temperature,
                use_sampling=use_sampling,
            )

        b_values = (completed[:, b_positions, 0] > 0.5).float()
        rows = torch.zeros((batch_size, length), device=device)
        rows[:, 0] = float(h)
        rows[:, r_positions] = basis_values
        rows[:, b_positions] = b_values
        outputs.append(rows)

        remaining -= batch_size
        pbar.update(batch_size)

    pbar.close()
    return torch.cat(outputs, dim=0) if outputs else torch.empty((0, length), device=device)


def tensor_rows_to_jsonable(rows):
    rows = rows.detach().cpu()
    result = []
    for row in rows.tolist():
        item = []
        for i, value in enumerate(row):
            if i == 0:
                item.append(float(value))
            else:
                item.append(int(round(value)))
        result.append(item)
    return result


def ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def main():
    args = parse_args()
    if not math.isfinite(args.temperature) or args.temperature <= 0.0:
        raise ValueError("--temperature must be a finite positive number.")
    if args.greedy:
        print("Note: --greedy uses argmax decoding, so --temperature does not affect generated tokens.")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if args.sample_size_per_param is not None:
        args.sample_size_per_h = args.sample_size_per_param

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    print("Generation arguments:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    print(f"  resolved_device: {device}")

    model = load_model(args, device=device)
    h_values = get_h_values(args, device=device)
    use_sampling = not args.greedy
    param_label = condition_name(args)

    all_samples = []
    for h in h_values:
        rows = generate_oseq_measurements(
            h=h,
            model=model,
            num_qubits=args.num_qubits,
            sample_size=args.sample_size_per_h,
            diffusion_steps=args.diffusion_steps,
            temperature=args.temperature,
            device=device,
            gen_batch_size=args.gen_batch_size,
            use_sampling=use_sampling,
            param_label=param_label,
        )
        all_samples.extend(tensor_rows_to_jsonable(rows))
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    if args.shuffle:
        random.shuffle(all_samples)

    ensure_parent_dir(args.save_data_path)
    with open(args.save_data_path, "w", encoding="utf-8") as f:
        json.dump(all_samples, f)

    print(f"Saved {len(all_samples)} samples to: {args.save_data_path}")
    print(f"Each sample length: {1 + 2 * args.num_qubits}")
    print(f"Format: [{param_label}, P1, b1, P2, b2, ...], directly loadable by train_oseq.py")


if __name__ == "__main__":
    main()
