"""
预计算 eval 所需的精确（参考）物理量并写入缓存，避免每次评估重复对角化等大计算。
用法:
  python eval_cal_exact.py --predict_model TFI --num_qubits 10
  python eval_cal_exact.py --predict_model Heisenberg --num_qubits 10
  python eval_cal_exact.py --predict_model xxz --num_qubits 10 --J_xy 1.0
  python eval_cal_exact.py --predict_model J1J2 --num_qubits 10 --J1 1.0
  python eval_cal_exact.py --predict_model ANNNI --num_qubits 10 --annni_kappa 0.5
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from eval_utils import (
    exact_annni_structure_factor_z,
    exact_pauli_correlation_from_rho,
    exact_single_pauli_from_rho,
    ground_cal_averge_two_point_pbc,
    ground_cal_averge_Xstring_pbc,
    obtain_eigenrho,
    obtain_ground_rho_from_hamiltonian,
    Hamiltonian_sym_circle,
    Hamiltonian_j1j2_pbc,
    Hamiltonian_annni_pbc,
    calculate_exact_values,
)
from eval_data_heisenberg import compute_exact_heisenberg_values
from eval_data_xxz import compute_exact_xxz_values


SP_I2_REAL = sp.csr_matrix(np.eye(2, dtype=np.float64))
SP_X_REAL = sp.csr_matrix(np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64))
SP_Z_REAL = sp.csr_matrix(np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.float64))
SP_Y_COMPLEX = sp.csr_matrix(np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128))
SPARSE_V0_SEED = 1729
J1J2_ENDPOINT_J2 = 1.0
J1J2_ENDPOINT_SUBSPACE_DIM = 12
J1J2_DEGENERACY_TOL = 1e-8


def _default_cache_dir():
    return os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), ".", "eval_exact_cache")
    )


def npz_path_tfi(cache_dir, num_qubits):
    return os.path.join(cache_dir, f"exact_TFI_N{int(num_qubits)}.npz")


def npz_path_heisenberg(cache_dir, num_qubits):
    return os.path.join(cache_dir, f"exact_Heisenberg_N{int(num_qubits)}.npz")


def npz_path_xxz(cache_dir, num_qubits, J_xy):
    return os.path.join(cache_dir, f"exact_xxz_N{int(num_qubits)}_J{J_xy}.npz")


def npz_path_j1j2(cache_dir, num_qubits, J1):
    return os.path.join(cache_dir, f"exact_J1J2_N{int(num_qubits)}_J1{J1}.npz")


def npz_path_annni(cache_dir, num_qubits, kappa, J1):
    return os.path.join(cache_dir, f"exact_ANNNI_N{int(num_qubits)}_kappa{kappa}_J1{J1}.npz")


def xxz_meta_path(npz_p):
    return npz_p.replace(".npz", "_meta.json")


def npz_has_keys(path, required_keys):
    if not os.path.isfile(path):
        return False
    try:
        with np.load(path) as data:
            return all(key in data.files for key in required_keys)
    except Exception:
        return False


def compute_and_save_tfi(cache_dir, num_qubits, force):
    path = npz_path_tfi(cache_dir, num_qubits)
    if os.path.isfile(path) and not force:
        print(f"已存在缓存，跳过: {path}")
        return path

    h_values = np.linspace(0.0, 1.0, 101)
    print("Calculating true values for TFI...")
    if num_qubits <= 10:
        ZZ_d1 = [ground_cal_averge_two_point_pbc(num_qubits, h, d=1).item() for h in h_values]
        ZZ_d2 = [ground_cal_averge_two_point_pbc(num_qubits, h, d=2).item() for h in h_values]
        ZZ_d3 = [ground_cal_averge_two_point_pbc(num_qubits, h, d=3).item() for h in h_values]
        ZZ_d4 = [ground_cal_averge_two_point_pbc(num_qubits, h, d=4).item() for h in h_values]
        ZZ_d5 = [ground_cal_averge_two_point_pbc(num_qubits, h, d=5).item() for h in h_values]
        ZZ_curves = np.array([ZZ_d1, ZZ_d2, ZZ_d3, ZZ_d4, ZZ_d5])

        Xs_l1 = [ground_cal_averge_Xstring_pbc(num_qubits, h, d=0).item() for h in h_values]
        Xs_l2 = [ground_cal_averge_Xstring_pbc(num_qubits, h, d=1).item() for h in h_values]
        Xs_l3 = [ground_cal_averge_Xstring_pbc(num_qubits, h, d=2).item() for h in h_values]
        Xs_l4 = [ground_cal_averge_Xstring_pbc(num_qubits, h, d=3).item() for h in h_values]
        Xs_l5 = [ground_cal_averge_Xstring_pbc(num_qubits, h, d=4).item() for h in h_values]
        Xs_curves = np.array([Xs_l1, Xs_l2, Xs_l3, Xs_l4, Xs_l5])

        Energy = []
        for h in h_values:
            rho = obtain_eigenrho(num_qubits, h)
            ham = Hamiltonian_sym_circle(num_qubits, h)
            energy = torch.real(torch.trace(rho @ ham)).item()
            Energy.append(energy)
        Energy = np.array(Energy)
    else:
        print("Calculating true values for real data more than 10 qubits...")
        Energy, ZZ_curves, Xs_curves = calculate_exact_values(num_qubits, h_values)

    os.makedirs(cache_dir, exist_ok=True)
    np.savez_compressed(
        path,
        h_values=h_values,
        ZZ_curves=ZZ_curves,
        Xs_curves=Xs_curves,
        Energy=Energy,
    )
    print(f"已保存 TFI 精确值: {path}")
    return path


def compute_and_save_heisenberg(cache_dir, num_qubits, force):
    path = npz_path_heisenberg(cache_dir, num_qubits)
    if os.path.isfile(path) and not force:
        print(f"已存在缓存，跳过: {path}")
        return path

    h_values = np.linspace(-2.0, 2.0, 101)
    print("Calculating true values for Heisenberg model...")
    exact_value = compute_exact_heisenberg_values(num_qubits, h_values)

    os.makedirs(cache_dir, exist_ok=True)
    np.savez_compressed(
        path,
        h_values=h_values,
        energy=exact_value["energy"],
        correlations_XX=exact_value["correlations_XX"],
        correlations_YY=exact_value["correlations_YY"],
        correlations_ZZ=exact_value["correlations_ZZ"],
        correlations_spin_dot=exact_value["correlations_spin_dot"],
    )
    print(f"已保存 Heisenberg 精确值: {path}")
    return path


def _build_xxz_sparse_parts(num_qubits):
    dim = 2**num_qubits
    xx_op = sp.csr_matrix((dim, dim), dtype=np.float64)
    yy_op = sp.csr_matrix((dim, dim), dtype=np.float64)
    zz_op = sp.csr_matrix((dim, dim), dtype=np.float64)

    for i in range(num_qubits):
        j = (i + 1) % num_qubits
        xx_op = xx_op + _two_site_sparse_operator(num_qubits, i, j, SP_X_REAL, SP_X_REAL)
        yy_op = yy_op + _two_site_sparse_operator(num_qubits, i, j, SP_Y_COMPLEX, SP_Y_COMPLEX).real
        zz_op = zz_op + _two_site_sparse_operator(num_qubits, i, j, SP_Z_REAL, SP_Z_REAL)

    return xx_op.tocsr(), yy_op.tocsr(), zz_op.tocsr()


def compute_exact_xxz_c_modality_values(num_qubits, J_xy, delta_values):
    print("Calculating true values for XXZ C modality (<XX>, <YY>, <ZZ>)...")
    xx_op, yy_op, zz_op = _build_xxz_sparse_parts(num_qubits)
    pair_terms = _precompute_two_site_terms(num_qubits, max_distance=1)[1]

    corr_xx = []
    corr_yy = []
    corr_zz = []
    energy_from_c = []
    derivative_from_c = []
    total_energy_from_c = []
    total_derivative_from_c = []
    v0 = _make_sparse_v0(xx_op.shape[0])

    for idx, delta in enumerate(delta_values):
        delta = float(delta)
        print(f"  XXZ C point {idx + 1}/{len(delta_values)}: Delta={delta:.6g}", flush=True)
        ham = (float(J_xy) * (xx_op + yy_op) + float(J_xy) * delta * zz_op).tocsr()
        _e0, psi = _lowest_sparse_state(ham, v0=v0)
        v0 = _normalize_sparse_v0(psi)

        xx, yy, zz = _two_site_pauli_averages_from_state(psi, pair_terms)
        energy = float(J_xy) * (xx + yy + delta * zz)
        derivative = float(J_xy) * zz

        corr_xx.append(xx)
        corr_yy.append(yy)
        corr_zz.append(zz)
        energy_from_c.append(energy)
        derivative_from_c.append(derivative)
        total_energy_from_c.append(float(num_qubits) * energy)
        total_derivative_from_c.append(float(num_qubits) * derivative)

    corr_xx = np.asarray(corr_xx, dtype=float)
    corr_yy = np.asarray(corr_yy, dtype=float)
    corr_zz = np.asarray(corr_zz, dtype=float)
    energy_from_c = np.asarray(energy_from_c, dtype=float)
    derivative_from_c = np.asarray(derivative_from_c, dtype=float)

    return {
        "xx_correlation": corr_xx,
        "yy_correlation": corr_yy,
        "c_corr_XX": corr_xx,
        "c_corr_YY": corr_yy,
        "c_corr_ZZ": corr_zz,
        "c_energy": energy_from_c,
        "c_energy_derivative": derivative_from_c,
        "c_total_energy": np.asarray(total_energy_from_c, dtype=float),
        "c_total_energy_derivative": np.asarray(total_derivative_from_c, dtype=float),
    }


def compute_and_save_xxz(cache_dir, num_qubits, J_xy, force):
    path = npz_path_xxz(cache_dir, num_qubits, J_xy)
    meta = xxz_meta_path(path)
    required_xxz_keys = [
        "energy",
        "magnetization_z",
        "zz_correlation",
        "xx_correlation",
        "yy_correlation",
        "c_corr_XX",
        "c_corr_YY",
        "c_corr_ZZ",
        "c_energy",
        "c_energy_derivative",
        "c_energy_per_bond",
        "c_energy_derivative_per_bond",
        "c_total_energy",
        "c_total_energy_derivative",
        "energy_total",
        "energy_per_bond",
        "energy_derivative_per_bond",
        "energy_derivative_total",
        "single_energy",
        "single_energy_derivative",
        "single_energy_per_bond",
        "single_energy_derivative_per_bond",
        "single_energy_total",
        "single_energy_derivative_total",
        "single_corr_XX",
        "single_corr_YY",
        "single_corr_ZZ",
        "single_local_Sz",
        "single_magnetization_z",
        "single_magnetization_staggered",
    ]
    if (
        os.path.isfile(path)
        and os.path.isfile(meta)
        and not force
        and npz_has_keys(path, required_xxz_keys)
    ):
        print(f"已存在缓存，跳过: {path}")
        return path

    if os.path.isfile(path) and os.path.isfile(meta) and not force:
        print(f"XXZ cache exists but misses multimodal fields; recomputing: {path}")

    delta_values = np.linspace(-2.0, 2.0, 121)
    print("Calculating true values for XXZ model...")
    exact_value = compute_exact_xxz_values(num_qubits, J_xy, delta_values)
    c_exact_value = compute_exact_xxz_c_modality_values(num_qubits, J_xy, delta_values)

    os.makedirs(cache_dir, exist_ok=True)
    np.savez_compressed(
        path,
        delta_values=delta_values,
        N=np.array(exact_value["N"]),
        J_xy=np.array(exact_value["J_xy"]),
        energy=exact_value["energy"],
        magnetization_z=exact_value["magnetization_z"],
        magnetization_staggered=exact_value["magnetization_staggered"],
        local_Sz=exact_value["local_Sz"],
        zz_correlation=exact_value["zz_correlation"],
        energy_derivative_numeric=np.asarray(exact_value["energy_derivative_numeric"], dtype=float),
        energy_derivative_analytic=np.asarray(exact_value["energy_derivative_analytic"], dtype=float),
        energy_second_derivative=np.asarray(exact_value["energy_second_derivative"], dtype=float),
        xx_correlation=c_exact_value["xx_correlation"],
        yy_correlation=c_exact_value["yy_correlation"],
        c_corr_XX=c_exact_value["c_corr_XX"],
        c_corr_YY=c_exact_value["c_corr_YY"],
        c_corr_ZZ=c_exact_value["c_corr_ZZ"],
        c_energy=c_exact_value["c_energy"],
        c_energy_derivative=c_exact_value["c_energy_derivative"],
        c_energy_per_bond=c_exact_value["c_energy"],
        c_energy_derivative_per_bond=c_exact_value["c_energy_derivative"],
        c_total_energy=c_exact_value["c_total_energy"],
        c_total_energy_derivative=c_exact_value["c_total_energy_derivative"],
        energy_total=exact_value["energy"],
        energy_per_bond=c_exact_value["c_energy"],
        energy_derivative_per_bond=c_exact_value["c_energy_derivative"],
        energy_derivative_total=c_exact_value["c_total_energy_derivative"],
        single_energy=c_exact_value["c_energy"],
        single_energy_derivative=c_exact_value["c_energy_derivative"],
        single_energy_per_bond=c_exact_value["c_energy"],
        single_energy_derivative_per_bond=c_exact_value["c_energy_derivative"],
        single_energy_total=exact_value["energy"],
        single_energy_derivative_total=c_exact_value["c_total_energy_derivative"],
        single_corr_XX=c_exact_value["c_corr_XX"],
        single_corr_YY=c_exact_value["c_corr_YY"],
        single_corr_ZZ=c_exact_value["c_corr_ZZ"],
        single_local_Sz=exact_value["local_Sz"],
        single_magnetization_z=exact_value["magnetization_z"],
        single_magnetization_staggered=exact_value["magnetization_staggered"],
    )
    with open(meta, "w", encoding="utf-8") as f:
        json.dump(exact_value["special_points"], f, indent=2)

    print(f"已保存 XXZ 精确值: {path} (+ meta)")
    return path


def _kron_all_sparse(ops):
    result = None
    for op in ops:
        result = op if result is None else sp.kron(result, op, format="csr")
    if result is None:
        raise ValueError("ops must be non-empty")
    return result


def _two_site_sparse_operator(num_qubits, i, j, op_i, op_j):
    i %= num_qubits
    j %= num_qubits
    if i == j:
        raise ValueError("two-site operator needs two distinct sites")
    identity = SP_I2_REAL.astype(np.result_type(op_i.dtype, op_j.dtype), copy=False)
    ops = [identity] * num_qubits
    ops[i] = op_i
    ops[j] = op_j
    return _kron_all_sparse(ops)


def _one_site_sparse_operator(num_qubits, i, op):
    i %= num_qubits
    ops = [SP_I2_REAL] * num_qubits
    ops[i] = op
    return _kron_all_sparse(ops)


def _normalize_sparse_v0(v0):
    v0 = np.asarray(v0)
    if np.iscomplexobj(v0) and np.max(np.abs(v0.imag)) < 1e-12:
        v0 = v0.real
    v0 = v0.astype(np.float64 if not np.iscomplexobj(v0) else np.complex128, copy=False)
    norm = np.linalg.norm(v0)
    if norm == 0:
        raise ValueError("Sparse eigensolver initial vector has zero norm")
    return v0 / norm


def _make_sparse_v0(dim, seed=SPARSE_V0_SEED):
    rng = np.random.default_rng(seed)
    return _normalize_sparse_v0(rng.standard_normal(dim))


def _fix_sparse_state_phase(psi):
    pivot = int(np.argmax(np.abs(psi)))
    if np.abs(psi[pivot]) == 0:
        return psi
    phase = psi[pivot] / np.abs(psi[pivot])
    psi = psi / phase
    if np.iscomplexobj(psi) and np.max(np.abs(psi.imag)) < 1e-12:
        psi = psi.real
    return psi


def _lowest_sparse_state(ham, v0=None, subspace_dim=1, degeneracy_tol=J1J2_DEGENERACY_TOL):
    if v0 is not None:
        v0 = _normalize_sparse_v0(v0)
    k = max(1, min(int(subspace_dim), ham.shape[0] - 2))
    eigenvalues, eigenvectors = spla.eigsh(
        ham,
        k=k,
        which="SA",
        v0=v0,
        tol=1e-10,
        maxiter=max(2000, 20 * ham.shape[0]),
    )
    order = np.argsort(eigenvalues.real)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    e0 = float(eigenvalues[0].real)
    ground_mask = eigenvalues.real <= e0 + degeneracy_tol * max(1.0, abs(e0))

    if k > 1 and np.count_nonzero(ground_mask) > 1 and v0 is not None:
        ground_vectors = eigenvectors[:, ground_mask]
        coeffs = ground_vectors.conj().T @ v0
        projected = ground_vectors @ coeffs
        if np.linalg.norm(projected) > 1e-12:
            psi = projected
        else:
            psi = eigenvectors[:, 0]
    else:
        psi = eigenvectors[:, 0]

    psi = _normalize_sparse_v0(psi)
    psi = _fix_sparse_state_phase(psi)
    return e0, psi


def _build_j1j2_sparse_parts(num_qubits):
    dim = 2**num_qubits
    h1 = sp.csr_matrix((dim, dim), dtype=np.float64)
    h2 = sp.csr_matrix((dim, dim), dtype=np.float64)

    for distance in (1, 2):
        for i in range(num_qubits):
            j = (i + distance) % num_qubits
            xx = _two_site_sparse_operator(num_qubits, i, j, SP_X_REAL, SP_X_REAL)
            yy = _two_site_sparse_operator(num_qubits, i, j, SP_Y_COMPLEX, SP_Y_COMPLEX).real
            zz = _two_site_sparse_operator(num_qubits, i, j, SP_Z_REAL, SP_Z_REAL)
            term = xx + yy + zz
            if distance == 1:
                h1 = h1 + term
            else:
                h2 = h2 + term

    return h1.tocsr(), h2.tocsr()


def _build_annni_sparse_parts(num_qubits):
    dim = 2**num_qubits
    zz1 = sp.csr_matrix((dim, dim), dtype=np.float64)
    zz2 = sp.csr_matrix((dim, dim), dtype=np.float64)
    x_field = sp.csr_matrix((dim, dim), dtype=np.float64)

    for i in range(num_qubits):
        zz1 = zz1 + _two_site_sparse_operator(num_qubits, i, i + 1, SP_Z_REAL, SP_Z_REAL)
        zz2 = zz2 + _two_site_sparse_operator(num_qubits, i, i + 2, SP_Z_REAL, SP_Z_REAL)
        x_field = x_field + _one_site_sparse_operator(num_qubits, i, SP_X_REAL)

    return zz1.tocsr(), zz2.tocsr(), x_field.tocsr()


def _precompute_two_site_terms(num_qubits, max_distance):
    basis = np.arange(2**num_qubits, dtype=np.int64)
    terms_by_distance = {}
    for distance in range(1, max_distance + 1):
        pair_terms = []
        for i in range(num_qubits):
            j = (i + distance) % num_qubits
            mask_i = 1 << (num_qubits - 1 - i)
            mask_j = 1 << (num_qubits - 1 - j)
            flip_indices = basis ^ (mask_i | mask_j)
            z_i = np.where((basis & mask_i) == 0, 1.0, -1.0)
            z_j = np.where((basis & mask_j) == 0, 1.0, -1.0)
            z_product = z_i * z_j
            pair_terms.append((flip_indices, z_product))
        terms_by_distance[distance] = pair_terms
    return terms_by_distance


def _precompute_single_x_terms(num_qubits):
    basis = np.arange(2**num_qubits, dtype=np.int64)
    flips = []
    for i in range(num_qubits):
        mask_i = 1 << (num_qubits - 1 - i)
        flips.append(basis ^ mask_i)
    return flips


def _two_site_pauli_averages_from_state(psi, pair_terms):
    prob = np.abs(psi) ** 2
    xx = 0.0
    yy = 0.0
    zz = 0.0
    for flip_indices, z_product in pair_terms:
        flipped = psi[flip_indices]
        xx += np.vdot(psi, flipped).real
        yy += np.vdot(psi, (-z_product) * flipped).real
        zz += np.dot(prob, z_product).real
    norm = float(len(pair_terms))
    return xx / norm, yy / norm, zz / norm


def _zz_average_from_state(psi, pair_terms):
    prob = np.abs(psi) ** 2
    zz = 0.0
    for _flip_indices, z_product in pair_terms:
        zz += np.dot(prob, z_product).real
    return zz / float(len(pair_terms))


def _single_x_average_from_state(psi, x_flip_terms):
    x_total = 0.0
    for flip_indices in x_flip_terms:
        x_total += np.vdot(psi, psi[flip_indices]).real
    return x_total / float(len(x_flip_terms))


def compute_exact_j1j2_values_sparse(num_qubits, J1, j2_values, max_distance=5):
    print(f"Using sparse Lanczos J1-J2 exact solver for N={num_qubits}.")
    h1, h2 = _build_j1j2_sparse_parts(num_qubits)
    observable_terms = _precompute_two_site_terms(num_qubits, max_distance)

    energy = []
    corr_xx = []
    corr_yy = []
    corr_zz = []
    corr_spin_dot = []
    dimer_proxy = []
    v0 = _make_sparse_v0(h1.shape[0])

    for idx, J2 in enumerate(j2_values):
        print(f"  J2 point {idx + 1}/{len(j2_values)}: J2={float(J2):.6g}", flush=True)
        ham = (float(J1) * h1 + float(J2) * h2).tocsr()
        subspace_dim = J1J2_ENDPOINT_SUBSPACE_DIM if abs(float(J2) - J1J2_ENDPOINT_J2) < 1e-12 else 1
        e0, psi = _lowest_sparse_state(
            ham,
            v0=v0,
            subspace_dim=subspace_dim,
            degeneracy_tol=J1J2_DEGENERACY_TOL,
        )
        v0 = _normalize_sparse_v0(psi)
        energy.append(e0)

        xx_row = []
        yy_row = []
        zz_row = []
        sd_row = []
        for d in range(1, max_distance + 1):
            xx, yy, zz = _two_site_pauli_averages_from_state(psi, observable_terms[d])
            xx_row.append(xx)
            yy_row.append(yy)
            zz_row.append(zz)
            sd_row.append(xx + yy + zz)

        corr_xx.append(xx_row)
        corr_yy.append(yy_row)
        corr_zz.append(zz_row)
        corr_spin_dot.append(sd_row)
        dimer_proxy.append(sd_row[0] - sd_row[1])

    return {
        "J2_values": np.asarray(j2_values, dtype=float),
        "J1": float(J1),
        "energy": np.asarray(energy, dtype=float),
        "correlations_XX": np.asarray(corr_xx, dtype=float).T,
        "correlations_YY": np.asarray(corr_yy, dtype=float).T,
        "correlations_ZZ": np.asarray(corr_zz, dtype=float).T,
        "correlations_spin_dot": np.asarray(corr_spin_dot, dtype=float).T,
        "dimer_proxy": np.asarray(dimer_proxy, dtype=float),
    }


def compute_exact_annni_values_sparse(num_qubits, h_values, kappa, J1, max_distance=5):
    print(f"Using sparse Lanczos ANNNI exact solver for N={num_qubits}.")
    zz1_op, zz2_op, x_op = _build_annni_sparse_parts(num_qubits)
    curve_terms = _precompute_two_site_terms(num_qubits, max_distance)
    sf_terms = _precompute_two_site_terms(num_qubits, num_qubits - 1)
    x_terms = _precompute_single_x_terms(num_qubits)

    energy = []
    zz_curves = []
    mx = []
    sf_pi = []
    sf_pi_over_2 = []
    v0 = _make_sparse_v0(zz1_op.shape[0])

    for idx, h in enumerate(h_values):
        print(f"  h point {idx + 1}/{len(h_values)}: h={float(h):.6g}", flush=True)
        ham = (-float(J1) * zz1_op + float(kappa) * float(J1) * zz2_op - float(h) * x_op).tocsr()
        e0, psi = _lowest_sparse_state(ham, v0=v0)
        v0 = _normalize_sparse_v0(psi)

        energy.append(e0)
        zz_row = [
            _zz_average_from_state(psi, curve_terms[d])
            for d in range(1, max_distance + 1)
        ]
        zz_curves.append(zz_row)
        mx.append(_single_x_average_from_state(psi, x_terms))

        zz_by_distance = {0: 1.0}
        for d in range(1, num_qubits):
            zz_by_distance[d] = _zz_average_from_state(psi, sf_terms[d])
        sf_pi.append(sum(np.cos(np.pi * r) * zz_by_distance[r] for r in range(num_qubits)))
        sf_pi_over_2.append(
            sum(np.cos((np.pi / 2.0) * r) * zz_by_distance[r] for r in range(num_qubits))
        )

    return {
        "h_values": np.asarray(h_values, dtype=float),
        "kappa": float(kappa),
        "J1": float(J1),
        "energy": np.asarray(energy, dtype=float),
        "ZZ_curves": np.asarray(zz_curves, dtype=float).T,
        "X_magnetization": np.asarray(mx, dtype=float),
        "structure_factor_pi": np.asarray(sf_pi, dtype=float),
        "structure_factor_pi_over_2": np.asarray(sf_pi_over_2, dtype=float),
    }


def compute_exact_j1j2_values(num_qubits, J1, j2_values, max_distance=5):
    if num_qubits >= 12:
        return compute_exact_j1j2_values_sparse(num_qubits, J1, j2_values, max_distance=max_distance)

    energy = []
    corr_xx = []
    corr_yy = []
    corr_zz = []
    corr_spin_dot = []
    dimer_proxy = []

    for J2 in j2_values:
        ham = Hamiltonian_j1j2_pbc(num_qubits, J1, float(J2))
        rho, e0 = obtain_ground_rho_from_hamiltonian(ham)
        energy.append(e0.item())

        xx_row = []
        yy_row = []
        zz_row = []
        sd_row = []
        for d in range(1, max_distance + 1):
            xx = exact_pauli_correlation_from_rho(rho, num_qubits, 1, d)
            yy = exact_pauli_correlation_from_rho(rho, num_qubits, 3, d)
            zz = exact_pauli_correlation_from_rho(rho, num_qubits, 2, d)
            xx_row.append(xx)
            yy_row.append(yy)
            zz_row.append(zz)
            sd_row.append(xx + yy + zz)
        corr_xx.append(xx_row)
        corr_yy.append(yy_row)
        corr_zz.append(zz_row)
        corr_spin_dot.append(sd_row)
        dimer_proxy.append(sd_row[0] - sd_row[1])

    return {
        "J2_values": np.asarray(j2_values, dtype=float),
        "J1": float(J1),
        "energy": np.asarray(energy, dtype=float),
        "correlations_XX": np.asarray(corr_xx, dtype=float).T,
        "correlations_YY": np.asarray(corr_yy, dtype=float).T,
        "correlations_ZZ": np.asarray(corr_zz, dtype=float).T,
        "correlations_spin_dot": np.asarray(corr_spin_dot, dtype=float).T,
        "dimer_proxy": np.asarray(dimer_proxy, dtype=float),
    }


def compute_and_save_j1j2(cache_dir, num_qubits, J1, force):
    path = npz_path_j1j2(cache_dir, num_qubits, J1)
    if os.path.isfile(path) and not force:
        print(f"已存在缓存，跳过: {path}")
        return path

    j2_values = np.linspace(0.0, 1.0, 101)
    print("Calculating true values for J1-J2 frustrated chain...")
    exact_value = compute_exact_j1j2_values(num_qubits, J1, j2_values)

    os.makedirs(cache_dir, exist_ok=True)
    np.savez_compressed(path, N=int(num_qubits), **exact_value)
    print(f"已保存 J1-J2 精确值: {path}")
    return path


def compute_exact_annni_values(num_qubits, h_values, kappa, J1, max_distance=5):
    if num_qubits >= 12:
        return compute_exact_annni_values_sparse(
            num_qubits,
            h_values,
            kappa,
            J1,
            max_distance=max_distance,
        )

    energy = []
    zz_curves = []
    mx = []
    sf_pi = []
    sf_pi_over_2 = []

    for h in h_values:
        ham = Hamiltonian_annni_pbc(num_qubits, float(h), kappa=float(kappa), J1=float(J1))
        rho, e0 = obtain_ground_rho_from_hamiltonian(ham)
        energy.append(e0.item())
        zz_curves.append([
            exact_pauli_correlation_from_rho(rho, num_qubits, 2, d)
            for d in range(1, max_distance + 1)
        ])
        mx.append(exact_single_pauli_from_rho(rho, num_qubits, 1))
        sf_pi.append(exact_annni_structure_factor_z(rho, num_qubits, np.pi))
        sf_pi_over_2.append(exact_annni_structure_factor_z(rho, num_qubits, np.pi / 2.0))

    return {
        "h_values": np.asarray(h_values, dtype=float),
        "kappa": float(kappa),
        "J1": float(J1),
        "energy": np.asarray(energy, dtype=float),
        "ZZ_curves": np.asarray(zz_curves, dtype=float).T,
        "X_magnetization": np.asarray(mx, dtype=float),
        "structure_factor_pi": np.asarray(sf_pi, dtype=float),
        "structure_factor_pi_over_2": np.asarray(sf_pi_over_2, dtype=float),
    }


def compute_and_save_annni(cache_dir, num_qubits, kappa, J1, force):
    path = npz_path_annni(cache_dir, num_qubits, kappa, J1)
    if os.path.isfile(path) and not force:
        print(f"已存在缓存，跳过: {path}")
        return path

    h_values = np.linspace(0.0, 2.0, 101)
    print("Calculating true values for ANNNI / NNN-TFI model...")
    exact_value = compute_exact_annni_values(num_qubits, h_values, kappa, J1)

    os.makedirs(cache_dir, exist_ok=True)
    np.savez_compressed(path, N=int(num_qubits), **exact_value)
    print(f"已保存 ANNNI 精确值: {path}")
    return path


def main():
    parser = argparse.ArgumentParser(description="Precompute exact values for eval cache")
    parser.add_argument(
        "--predict_model",
        type=str,
        required=True,
        choices=["TFI", "Heisenberg", "xxz", "J1J2", "ANNNI"],
        help="Physical model (same as eval.py)",
    )
    parser.add_argument("--num_qubits", type=int, required=True)
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Directory for .npz cache (default: <repo>/data/exact_cache)",
    )
    parser.add_argument("--J_xy", type=float, default=1.0, help="XXZ J_xy (default 1.0, same as eval.py)")
    parser.add_argument("--J1", type=float, default=1.0, help="J1 coupling for J1J2/ANNNI")
    parser.add_argument("--annni_kappa", type=float, default=0.5, help="ANNNI next-nearest-neighbor coupling ratio")
    parser.add_argument("--force", action="store_true", help="Recompute even if cache exists")
    args = parser.parse_args()

    cache_dir = args.cache_dir or _default_cache_dir()

    if args.predict_model == "TFI":
        compute_and_save_tfi(cache_dir, args.num_qubits, args.force)
    elif args.predict_model == "Heisenberg":
        compute_and_save_heisenberg(cache_dir, args.num_qubits, args.force)
    elif args.predict_model == "xxz":
        compute_and_save_xxz(cache_dir, args.num_qubits, args.J_xy, args.force)
    elif args.predict_model == "J1J2":
        compute_and_save_j1j2(cache_dir, args.num_qubits, args.J1, args.force)
    elif args.predict_model == "ANNNI":
        compute_and_save_annni(cache_dir, args.num_qubits, args.annni_kappa, args.J1, args.force)

    print("True values calculation complete...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
