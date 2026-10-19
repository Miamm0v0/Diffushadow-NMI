# Here are functions to calculate physical properties (both for true values and predicted values using shadow tomography)
import numpy as np
import torch
import scipy.sparse.linalg as spla
import scipy.sparse as sp
from typing import List
import time
import os
import gc

try:
    import psutil
except ImportError:
    psutil = None



# 您现有的定义
X = torch.tensor([[0.,1.],[1.,0.]], dtype = torch.cfloat)
Z = torch.tensor([[1.,0.],[0.,-1.]], dtype = torch.cfloat)
I = torch.tensor([[1.,0.],[0.,1.]], dtype = torch.cfloat)
Y = torch.tensor([[0., -1.j], [1.j, 0.]], dtype=torch.cfloat)

operator_map = {0:I, 1:X, 2:Z, 3:Y}  # 3对应Y矩阵

def correlation_func(sqe_op):
    func = operator_map[int(sqe_op[0])]
    for i in range(len(sqe_op)-1):
        func = torch.kron(func,operator_map[int(sqe_op[i+1])])
    return func

# periodic boundary condition
def Hamiltonian_sym_circle(num_qubits, h):
    ham = 0.+0.j
    for i in range(num_qubits-1):
        Z_index = np.zeros(num_qubits)
        X_index = np.zeros(num_qubits)
        Z_index[i] = 2
        Z_index[i+1] = 2
        X_index[i] = 1
        ham+= -((1-h)*correlation_func(Z_index)+h*correlation_func(X_index))
    X_index = np.zeros(num_qubits)
    X_index[-1] = 1
    ham += -h*correlation_func(X_index)
    Z_index = np.zeros(num_qubits)
    Z_index[0], Z_index[-1] = 2, 2
    ham -= (1-h)*correlation_func(Z_index)
    return ham


# ==================== 海森堡模型哈密顿量 ====================
# 一维周期边界海森堡模型: H = J * Σ_i (X_i X_{i+1} + Y_i Y_{i+1} + Z_i Z_{i+1})
def Hamiltonian_heisenberg_pbc(num_qubits, J):
    """
    生成海森堡模型哈密顿量
    """
    ham = torch.zeros((2**num_qubits, 2**num_qubits), dtype=torch.cfloat)
    
    Y = torch.tensor([[0., -1.j], [1.j, 0.]], dtype=torch.cfloat)
    
    
    # 使用您现有的 Pauli 矩阵定义
    X = torch.tensor([[0., 1.], [1., 0.]], dtype=torch.cfloat)
    Y = torch.tensor([[0., -1.j], [1.j, 0.]], dtype=torch.cfloat)
    Z = torch.tensor([[1., 0.], [0., -1.]], dtype=torch.cfloat)
    I = torch.eye(2, dtype=torch.cfloat)
    
    for i in range(num_qubits):
        j = (i + 1) % num_qubits  # 周期边界
        
        # XX 项
        xx_index = np.zeros(num_qubits, dtype=int)
        xx_index[i] = 1  # X 在您代码中对应 1
        xx_index[j] = 1
        ham += J * correlation_func(xx_index)
        
        # YY 项 - 需要自定义，因为您的 correlation_func 可能不支持 Y
        # 直接构建 YY 算符
        term_yy = torch.eye(1, dtype=torch.cfloat)
        for pos in range(num_qubits):
            if pos == i or pos == j:
                term_yy = torch.kron(term_yy, Y)
            else:
                term_yy = torch.kron(term_yy, I)
        ham += J * term_yy
        
        # ZZ 项
        zz_index = np.zeros(num_qubits, dtype=int)
        zz_index[i] = 2  # Z 在您代码中对应 2
        zz_index[j] = 2
        ham += J * correlation_func(zz_index)
    
    return ham

def Hamiltonian_xxz_pbc(num_qubits, J_xy, Delta):
    """
    生成XXZ模型哈密顿量（周期边界条件）
    
    参数:
        num_qubits: 自旋个数
        J_xy: XY平面内的耦合强度，通常设为1.0
        Delta: 各向异性参数，Z方向耦合为 J_xy * Delta
        哈密顿量: H = J_xy Σ_i (S^x_i S^x_{i+1} + S^y_i S^y_{i+1}) + J_xy*Delta Σ_i S^z_i S^z_{i+1}
    """
    ham = torch.zeros((2**num_qubits, 2**num_qubits), dtype=torch.cfloat)
    
    for i in range(num_qubits):
        j = (i + 1) % num_qubits  # 周期边界
        
        # XX 项：系数 J_xy
        xx_index = np.zeros(num_qubits, dtype=int)
        xx_index[i] = 1  # X对应1
        xx_index[j] = 1
        ham += J_xy * correlation_func(xx_index)
        
        # YY 项：系数 J_xy
        yy_index = np.zeros(num_qubits, dtype=int)
        yy_index[i] = 3  # Y对应3
        yy_index[j] = 3
        ham += J_xy * correlation_func(yy_index)
        
        # ZZ 项：系数 J_xy * Delta
        zz_index = np.zeros(num_qubits, dtype=int)
        zz_index[i] = 2  # Z对应2
        zz_index[j] = 2
        ham += J_xy * Delta * correlation_func(zz_index)
    
    return ham


def Hamiltonian_j1j2_pbc(num_qubits, J1, J2):
    """
    Periodic frustrated J1-J2 spin chain in the Pauli convention used by this
    codebase:

        H = J1 sum_i sigma_i.sigma_{i+1}
          + J2 sum_i sigma_i.sigma_{i+2}.

    This uses Pauli matrices directly, not S = sigma/2. Convert by a factor 1/4
    if comparing to spin-operator conventions in the literature.
    """
    ham = torch.zeros((2**num_qubits, 2**num_qubits), dtype=torch.float64)
    for distance, coupling in ((1, J1), (2, J2)):
        for i in range(num_qubits):
            j = (i + distance) % num_qubits
            for pauli_idx in (1, 3, 2):  # X, Y, Z in operator_map
                op_index = np.zeros(num_qubits, dtype=int)
                op_index[i] = pauli_idx
                op_index[j] = pauli_idx
                ham += coupling * correlation_func(op_index).real.double()
    return ham


def Hamiltonian_annni_pbc(num_qubits, h, kappa=0.5, J1=1.0):
    """
    Periodic quantum ANNNI / next-nearest-neighbor transverse Ising model:

        H = -J1 sum_i Z_i Z_{i+1}
            + kappa*J1 sum_i Z_i Z_{i+2}
            - h sum_i X_i.

    kappa controls frustration/antiphase tendency; h is the transverse field and
    is the scalar condition generated by Diffushadow.
    """
    ham = torch.zeros((2**num_qubits, 2**num_qubits), dtype=torch.float64)
    for i in range(num_qubits):
        j1 = (i + 1) % num_qubits
        j2 = (i + 2) % num_qubits

        zz1 = np.zeros(num_qubits, dtype=int)
        zz1[i] = 2
        zz1[j1] = 2
        ham += -J1 * correlation_func(zz1).real.double()

        zz2 = np.zeros(num_qubits, dtype=int)
        zz2[i] = 2
        zz2[j2] = 2
        ham += kappa * J1 * correlation_func(zz2).real.double()

        x_idx = np.zeros(num_qubits, dtype=int)
        x_idx[i] = 1
        ham += -h * correlation_func(x_idx).real.double()
    return ham


def symmetry_proj(rho, num_qubits):
    op1_index = np.zeros(num_qubits)
    op1 = correlation_func(op1_index)
    op2_index = np.ones(num_qubits)
    op2 = correlation_func(op2_index)
    rho_ = rho + op2 @ rho @ op2.conj().transpose(-2, -1)
    return rho_ / torch.trace(rho_)

def obtain_eigenrho(num_bits,h):
    Ham = Hamiltonian_sym_circle(num_bits,h)
    eigenvalues, eigenvectors = torch.linalg.eigh(Ham)
    ground = eigenvectors[:,0].view(-1,1)
    rho = ground @ ground.conj().transpose(-2, -1)
    rho = symmetry_proj(rho,num_bits)
    return rho


def obtain_ground_rho_from_hamiltonian(ham):
    eigenvalues, eigenvectors = torch.linalg.eigh(ham)
    ground = eigenvectors[:, 0].view(-1, 1).to(torch.cfloat)
    e0 = eigenvalues[0]
    if torch.is_complex(e0):
        e0 = e0.real
    return ground @ ground.conj().transpose(-2, -1), e0


def exact_pauli_correlation_from_rho(rho, num_bits, pauli_idx, distance):
    if distance % num_bits == 0:
        return 1.0
    value = 0.0 + 0.0j
    for i in range(num_bits):
        op_index = np.zeros(num_bits, dtype=int)
        op_index[i] = pauli_idx
        op_index[(i + distance) % num_bits] = pauli_idx
        value += torch.trace(rho @ correlation_func(op_index))
    return torch.real(value / num_bits).item()


def exact_single_pauli_from_rho(rho, num_bits, pauli_idx):
    value = 0.0 + 0.0j
    for i in range(num_bits):
        op_index = np.zeros(num_bits, dtype=int)
        op_index[i] = pauli_idx
        value += torch.trace(rho @ correlation_func(op_index))
    return torch.real(value / num_bits).item()


def exact_annni_structure_factor_z(rho, num_bits, q):
    total = 0.0
    for r in range(num_bits):
        corr = exact_pauli_correlation_from_rho(rho, num_bits, 2, r)
        total += np.cos(q * r) * corr
    return float(total)

# ground truth
# Two point correlation function <Z_i Z_i+d>
def ground_cal_averge_two_point_pbc(num_bits, h, d):
    rho = obtain_eigenrho(num_bits,h)
    nmz = 0
    for i in range(num_bits):
        ZZ_index = np.zeros(num_bits)
        ZZ_index[i] = 2
        ZZ_index[(i+d)%num_bits] = 2
        ope = correlation_func(ZZ_index)
        nmz += torch.trace(rho@ope)
    return nmz/num_bits
# <X>
def ground_cal_magnezition_xpbc(num_bits,h):
    rho = obtain_eigenrho(num_bits,h)
    mz = 0
    for i in range(num_bits):
        Z_index = np.zeros(num_bits)
        Z_index[i] = 1
        ope = correlation_func(Z_index)
        mz += torch.trace(rho@ope)
    return mz/num_bits

# <XX...X> 
def ground_cal_averge_Xstring_pbc(num_bits, h, d):
    rho = obtain_eigenrho(num_bits,h)
    nmx = 0
    for i in range(num_bits):
        Xs_index = np.zeros(num_bits)
        for j in range(i, i + d + 1):
            Xs_index[j % num_bits] = 1
        ope = correlation_func(Xs_index)
        nmx += torch.trace(rho@ope)
    return nmx/num_bits

# GPT
# calculate <X> given sequences generated by transformer
def cal_magnezition_x(sqe):
    num_bits = int(sqe.shape[1]/2)
    mz = 0
    for i in range(num_bits):
        cond_Z_p = (sqe[:,i*2]==2) & (sqe[:,i*2+1]==1)
        cond_Z_n = (sqe[:,i*2]==2) & (sqe[:,i*2+1]==0)
        tempz = torch.sum(cond_Z_p)*3-torch.sum(cond_Z_n)*3
        mz += tempz
    return mz/(num_bits*sqe.shape[0])

# calculate E(<Z_i Z_i+d>) given sequences generated by transformer
def cal_averge_two_point_pbc(sqe, d):
    num_bits = int(sqe.shape[1]/2)
    nmz = 0
    expZ = torch.zeros(sqe.shape[0], sqe.shape[1] // 2, dtype=sqe.dtype)
    cond_3_Z = (sqe[:, :-1] == 4) & (sqe[:, 1:] == 1)
    cond_neg_3_Z = (sqe[:, :-1] == 4) & (sqe[:, 1:] == 0)
    for i in range(0, expZ.shape[1] * 2, 2):
        expZ[:, i // 2][cond_3_Z[:, i]] = 3
        expZ[:, i // 2][cond_neg_3_Z[:, i]] = -3
    expZZ = expZ* torch.cat((expZ[:,d:],expZ[:,:d]),dim=1)
    return torch.sum(expZZ)/(num_bits*sqe.shape[0])

def cal_averge_two_point_Y_pbc(sqe, d):
    num_bits = int(sqe.shape[1]/2)
    nmz = 0
    expZ = torch.zeros(sqe.shape[0], sqe.shape[1] // 2, dtype=sqe.dtype)
    cond_3_Z = (sqe[:, :-1] == 3) & (sqe[:, 1:] == 1)
    cond_neg_3_Z = (sqe[:, :-1] == 3) & (sqe[:, 1:] == 0)
    for i in range(0, expZ.shape[1] * 2, 2):
        expZ[:, i // 2][cond_3_Z[:, i]] = 3
        expZ[:, i // 2][cond_neg_3_Z[:, i]] = -3
    expZZ = expZ* torch.cat((expZ[:,d:],expZ[:,:d]),dim=1)
    return torch.sum(expZZ)/(num_bits*sqe.shape[0])

def cal_averge_two_point_X_pbc(sqe, d):
    num_bits = int(sqe.shape[1]/2)
    nmz = 0
    expZ = torch.zeros(sqe.shape[0], sqe.shape[1] // 2, dtype=sqe.dtype)
    cond_3_Z = (sqe[:, :-1] == 2) & (sqe[:, 1:] == 1)
    cond_neg_3_Z = (sqe[:, :-1] == 2) & (sqe[:, 1:] == 0)
    for i in range(0, expZ.shape[1] * 2, 2):
        expZ[:, i // 2][cond_3_Z[:, i]] = 3
        expZ[:, i // 2][cond_neg_3_Z[:, i]] = -3
    expZZ = expZ* torch.cat((expZ[:,d:],expZ[:,:d]),dim=1)
    return torch.sum(expZZ)/(num_bits*sqe.shape[0])


def cal_ZZ_nearest_from_c(sqe_c):
    """计算最近邻ZZ关联"""
    batch_size = sqe_c.shape[0]
    num_qubits = 10
    
    total = 0.0
    count = 0
    
    for i in range(num_qubits):
        r1_idx = i * 3
        r2_idx = r1_idx + 1
        p_idx = r2_idx + 1
        
        r1 = sqe_c[:, r1_idx]
        r2 = sqe_c[:, r2_idx]
        p_val = sqe_c[:, p_idx]
        
        # 如果是Z基测量，直接使用p值
        z_mask = (r1 == 4) & (r2 == 4)
        if torch.any(z_mask):
            zz_values = (2 * p_val[z_mask] - 1)  # 直接就是ZZ关联值
            zz_values = zz_values.float()
            total += torch.mean(zz_values)
            count += 1
    result = total / count if count > 0 else 0.0
    
    return torch.tensor(result, dtype=torch.float32)

# calculate E(<X...X>), here d = l-1, d=0 means calculating <X> given sequences generated by transformer
def cal_averge_Xstring_pbc(sqe, d):
    if d == 0:
        return cal_magnezition_x(sqe)
    else:
        num_bits = int(sqe.shape[1] / 2)
        nmx = 0
        expX = torch.zeros(sqe.shape[0], sqe.shape[1] // 2, dtype=sqe.dtype)  
        cond_3_X = (sqe[:, :-1] == 2) & (sqe[:, 1:] == 1)
        cond_neg_3_X = (sqe[:, :-1] == 2) & (sqe[:, 1:] == 0)    
        for i in range(0, expX.shape[1] * 2, 2):
            expX[:, i // 2][cond_3_X[:, i]] = 3
            expX[:, i // 2][cond_neg_3_X[:, i]] = -3   
        expXs = expX.clone()
        for shift in range(1, d + 1):
            expXs *= torch.cat((expX[:, shift:], expX[:, :shift]), dim=1)
        return torch.sum(expXs) / (num_bits * sqe.shape[0])
    
# calculate ground energy given sequences generated by transformer
def cal_ground_pbc(h,sqe):
    expX = torch.zeros(sqe.shape[0], sqe.shape[1] // 2, dtype=sqe.dtype)
    cond_3_X = (sqe[:, :-1] == 2) & (sqe[:, 1:] == 1)
    cond_neg_3_X = (sqe[:, :-1] == 2) & (sqe[:, 1:] == 0)
    for i in range(0, expX.shape[1] * 2, 2):
        expX[:, i // 2][cond_3_X[:, i]] = 3
        expX[:, i // 2][cond_neg_3_X[:, i]] = -3
        
    expZ = torch.zeros(sqe.shape[0], sqe.shape[1] // 2, dtype=sqe.dtype)
    cond_3_Z = (sqe[:, :-1] == 4) & (sqe[:, 1:] == 1)
    cond_neg_3_Z = (sqe[:, :-1] == 4) & (sqe[:, 1:] == 0)
    for i in range(0, expZ.shape[1] * 2, 2):
        expZ[:, i // 2][cond_3_Z[:, i]] = 3
        expZ[:, i // 2][cond_neg_3_Z[:, i]] = -3
    expZZ = expZ[:,:-1]*expZ[:,1:]

    T = sqe.shape[0]
    return -((1-h)*torch.sum(expZZ) + (1-h)*torch.sum(expZ[:,-1]*expZ[:,0]) + h*torch.sum(expX))/T


def _shadow_single_pauli(sqe, pauli_code):
    expP = torch.zeros(sqe.shape[0], sqe.shape[1] // 2, dtype=torch.float32, device=sqe.device)
    cond_pos = (sqe[:, :-1] == pauli_code) & (sqe[:, 1:] == 1)
    cond_neg = (sqe[:, :-1] == pauli_code) & (sqe[:, 1:] == 0)
    for i in range(0, expP.shape[1] * 2, 2):
        expP[:, i // 2][cond_pos[:, i]] = 3.0
        expP[:, i // 2][cond_neg[:, i]] = -3.0
    return expP


def cal_single_pauli_average(sqe, pauli_type='X'):
    pauli_code = {'X': 2, 'Y': 3, 'Z': 4}[pauli_type]
    expP = _shadow_single_pauli(sqe, pauli_code)
    return torch.sum(expP) / (expP.shape[0] * expP.shape[1])


def cal_spin_dot_correlation(sqe, d):
    return (
        cal_averge_two_point_X_pbc(sqe, d)
        + cal_averge_two_point_Y_pbc(sqe, d)
        + cal_averge_two_point_pbc(sqe, d)
    )


def cal_energy_j1j2(J1, J2, sqe):
    num_bits = int(sqe.shape[1] // 2)
    return num_bits * (J1 * cal_spin_dot_correlation(sqe, 1) + J2 * cal_spin_dot_correlation(sqe, 2))


def cal_dimer_proxy_j1j2(sqe):
    """
    Translation-invariant finite systems often have zero signed dimer order.
    This proxy reports the difference between nearest- and next-nearest spin-dot
    correlations, which is a useful shadow-estimable frustration diagnostic.
    """
    return cal_spin_dot_correlation(sqe, 1) - cal_spin_dot_correlation(sqe, 2)


def cal_energy_annni(h, sqe, kappa=0.5, J1=1.0):
    num_bits = int(sqe.shape[1] // 2)
    zz1 = cal_averge_two_point_pbc(sqe, 1)
    zz2 = cal_averge_two_point_pbc(sqe, 2)
    mx = cal_single_pauli_average(sqe, 'X')
    return num_bits * (-J1 * zz1 + kappa * J1 * zz2 - h * mx)


def cal_annni_structure_factor_z(sqe, q=np.pi):
    num_bits = int(sqe.shape[1] // 2)
    total = torch.tensor(1.0, dtype=torch.float32, device=sqe.device)
    for r in range(1, num_bits):
        total = total + float(np.cos(q * r)) * cal_averge_two_point_pbc(sqe, r)
    return total


def cal_energy_heisenberg(J, sqe):
    """
    从影子数据估计海森堡模型能量 H = J * Σ_i (X_i X_{i+1} + Y_i Y_{i+1} + Z_i Z_{i+1})
    
    Args:
        J (float): 耦合常数
        sqe (torch.Tensor): 影子数据，形状为 [T, 2N]，格式为 [P1, b1, P2, b2, ..., PN, bN]
                           P_i ∈ {2,3,4} 对应 {X,Y,Z}，b_i ∈ {0,1} 对应 {-1,+1}
    Returns:
        energy (torch.Tensor): 估计的基态能量期望值
    """
    T, seq_len = sqe.shape
    num_bits = seq_len // 2
    
    # 初始化三个方向的关联值累加器
    xx_sum = torch.tensor(0.0)
    yy_sum = torch.tensor(0.0)
    zz_sum = torch.tensor(0.0)
    
    # 遍历所有影子样本 (T个)
    for t in range(T):
        # 遍历所有最近邻对 (i, i+1)，周期边界
        for i in range(num_bits):
            j = (i + 1) % num_bits  # 相邻位点
            
            # 提取测量信息
            P_i = sqe[t, 2*i]      # 位点i的测量基
            b_i = sqe[t, 2*i+1]    # 位点i的测量结果
            P_j = sqe[t, 2*j]      # 位点j的测量基
            b_j = sqe[t, 2*j+1]    # 位点j的测量结果
            
            # 经典影子估计规则：只有当两个位点都测了相同Pauli基时，才对相应的关联有贡献
            # 贡献值为 b_i * b_j * 3 * 3 （因为每个单点估计器有因子3）
            if P_i == P_j:
                sign = 1.0 if b_i == b_j else -1.0
                contribution = sign * 9.0  # 3 * 3
                
                if P_i == 2:    # X基
                    xx_sum += contribution
                elif P_i == 3:  # Y基
                    yy_sum += contribution
                elif P_i == 4:  # Z基
                    zz_sum += contribution
    
    # 计算每个方向的关联平均值，然后组合成能量
    total_pairs = T * num_bits
    avg_xx = xx_sum / total_pairs
    avg_yy = yy_sum / total_pairs
    avg_zz = zz_sum / total_pairs
    
    # 能量期望值 = J * ( <XX> + <YY> + <ZZ> )
    energy = J * (avg_xx + avg_yy + avg_zz)
    
    return energy


def cal_correlation_general(sqe, d, pauli_type='Z'):
    """
    通用关联函数估计器 - 修复版本
    
    Args:
        sqe: 影子数据 [T, 2N]，格式: [P1, b1, P2, b2, ..., PN, bN]
            您的编码: P∈{1:X, 2:Z, 3:Y}, b∈{0:result=-1, 1:result=+1}
        d: 关联距离
        pauli_type: 'X', 'Y', 或 'Z'
    Returns:
        估计的关联函数值 <P_i P_{i+d}> 对所有i的平均
    """
    # 修正：根据您的operator_map定义正确的映射
    # 您的operator_map: {0:I, 1:X, 2:Z, 3:Y}
    pauli_code = {'X': 2, 'Y': 3, 'Z': 4}  # 注意：这里对应您的编码！
    code = pauli_code[pauli_type]
    
    T, seq_len = sqe.shape
    num_bits = seq_len // 2
    
    # 初始化总相关值和计数器
    total = 0.0
    count = 0
    
    # 遍历所有影子
    for t in range(T):
        # 遍历所有起始位点
        for i in range(num_bits):
            j = (i + d) % num_bits  # 周期边界
            
            P_i = sqe[t, 2*i].item()      # 位点i的测量基
            b_i = sqe[t, 2*i+1].item()    # 位点i的测量结果
            P_j = sqe[t, 2*j].item()      # 位点j的测量基
            b_j = sqe[t, 2*j+1].item()    # 位点j的测量结果
            
            # 只有当两个位点都测了指定Pauli基时才有贡献
            if P_i == code and P_j == code:
                # 测量结果: 0→-1, 1→+1
                sign = 1 if b_i == b_j else -1
                # 经典影子估计因子：对于单比特Pauli测量，(3λ_i-1)/2
                # 这里λ_i = (1 if b_i==1 else -1)，所以估计值为 3*b_sign
                # 对于两个位点：3*b_sign_i * 3*b_sign_j = 9*sign
                total += 9.0 * sign
                count += 1
    
    # 计算平均值，按位点数量平均
    if count > 0:
        # 注意：我们需要按量子比特数平均，而不只是有效测量数
        # 因为对于每个i，如果测量基不匹配，贡献为0
        return torch.tensor(total / (num_bits * T), dtype=torch.float32)
    else:
        return torch.tensor(0.0, dtype=torch.float32)


# code to implement median of means

def median_of_means(h, sqe, num_parts):
    T, _ = sqe.shape
    part_size = T // num_parts
    
    means = []
    for i in range(num_parts):
        part_sqe = sqe[i*part_size : (i+1)*part_size]
        mean_value = cal_ground_pbc(h, part_sqe)
        means.append(mean_value.item())
    
    median_value = torch.median(torch.tensor(means))
    return median_value
def median_of_means_two(d, sqe, num_parts):
    T, _ = sqe.shape
    part_size = T // num_parts
    
    means = []
    for i in range(num_parts):
        part_sqe = sqe[i*part_size : (i+1)*part_size]
        mean_value = cal_averge_two_point_pbc(part_sqe,d)
        means.append(mean_value.item())
    
    median_value = torch.median(torch.tensor(means))
    return median_value


def median_of_means_X(d, sqe, num_parts):
    T, _ = sqe.shape
    part_size = T // num_parts
    
    means = []
    for i in range(num_parts):
        part_sqe = sqe[i*part_size : (i+1)*part_size]
        mean_value = cal_averge_Xstring_pbc(part_sqe,d)
        means.append(mean_value.item())
    
    median_value = torch.median(torch.tensor(means))
    return median_value

def median_of_means_two_from_c(sqe_c, num_parts):
    """专门用于c序列的median_of_means"""
    T, _ = sqe_c.shape
    part_size = T // num_parts
    
    means = []
    for i in range(num_parts):
        part_sqe_c = sqe_c[i*part_size : (i+1)*part_size]
        mean_value = cal_ZZ_nearest_from_c(part_sqe_c)  # 使用c序列计算函数
        means.append(mean_value.item())
    
    median_value = torch.median(torch.tensor(means))
    return median_value


def median_of_means_heisenberg(J, sqe, num_parts=10):
    """海森堡模型能量的中位数平均法估计"""
    T, _ = sqe.shape
    part_size = T // num_parts
    
    means = []
    for i in range(num_parts):
        part_sqe = sqe[i*part_size : (i+1)*part_size]
        mean_value = cal_energy_heisenberg(J, part_sqe)
        means.append(mean_value.item())
    
    return torch.median(torch.tensor(means))

def median_of_means_correlation(sqe, d, pauli_type='Z', num_parts=10):
    """关联函数的中位数平均法估计"""
    T, _ = sqe.shape
    part_size = T // num_parts
    
    means = []
    for i in range(num_parts):
        part_sqe = sqe[i*part_size : (i+1)*part_size]
        if pauli_type == 'Z':
            mean_value = cal_averge_two_point_pbc(part_sqe, d)
            means.append(mean_value.item())
        elif pauli_type == 'X':
            mean_value = cal_averge_two_point_X_pbc(part_sqe, d)
            means.append(mean_value.item())
        else:
            mean_value = cal_averge_two_point_Y_pbc(part_sqe, d)
            means.append(mean_value.item())            
        

    
    return torch.median(torch.tensor(means))


def median_of_means_j1j2_energy(J1, J2, sqe, num_parts=10):
    T, _ = sqe.shape
    part_size = max(1, T // num_parts)
    means = []
    for i in range(num_parts):
        part_sqe = sqe[i * part_size:(i + 1) * part_size]
        if part_sqe.numel() == 0:
            continue
        means.append(cal_energy_j1j2(J1, J2, part_sqe).item())
    return torch.median(torch.tensor(means)) if means else torch.tensor(0.0)


def median_of_means_j1j2_dimer_proxy(sqe, num_parts=10):
    T, _ = sqe.shape
    part_size = max(1, T // num_parts)
    means = []
    for i in range(num_parts):
        part_sqe = sqe[i * part_size:(i + 1) * part_size]
        if part_sqe.numel() == 0:
            continue
        means.append(cal_dimer_proxy_j1j2(part_sqe).item())
    return torch.median(torch.tensor(means)) if means else torch.tensor(0.0)


def median_of_means_annni_energy(h, sqe, kappa=0.5, J1=1.0, num_parts=10):
    T, _ = sqe.shape
    part_size = max(1, T // num_parts)
    means = []
    for i in range(num_parts):
        part_sqe = sqe[i * part_size:(i + 1) * part_size]
        if part_sqe.numel() == 0:
            continue
        means.append(cal_energy_annni(h, part_sqe, kappa=kappa, J1=J1).item())
    return torch.median(torch.tensor(means)) if means else torch.tensor(0.0)


def median_of_means_single_pauli(sqe, pauli_type='X', num_parts=10):
    T, _ = sqe.shape
    part_size = max(1, T // num_parts)
    means = []
    for i in range(num_parts):
        part_sqe = sqe[i * part_size:(i + 1) * part_size]
        if part_sqe.numel() == 0:
            continue
        means.append(cal_single_pauli_average(part_sqe, pauli_type).item())
    return torch.median(torch.tensor(means)) if means else torch.tensor(0.0)


def median_of_means_annni_structure_factor_z(sqe, q=np.pi, num_parts=10):
    T, _ = sqe.shape
    part_size = max(1, T // num_parts)
    means = []
    for i in range(num_parts):
        part_sqe = sqe[i * part_size:(i + 1) * part_size]
        if part_sqe.numel() == 0:
            continue
        means.append(cal_annni_structure_factor_z(part_sqe, q=q).item())
    return torch.median(torch.tensor(means)) if means else torch.tensor(0.0)









def cal_energy_and_derivative_xxz(J_xy, Delta, sqe):
    """
    同时估计XXZ模型能量和能量导数（更高效的实现）
    
    Args:
        J_xy (float): XY耦合强度
        Delta (float): 各向异性参数
        sqe (torch.Tensor): 影子数据
    
    Returns:
        energy (torch.Tensor): 估计的基态能量
        zz_corr (torch.Tensor): ZZ关联期望值（用于计算导数）
    """
    T, seq_len = sqe.shape
    num_bits = seq_len // 2
    
    xx_sum = torch.tensor(0.0)
    yy_sum = torch.tensor(0.0)
    zz_sum = torch.tensor(0.0)
    
    for t in range(T):
        for i in range(num_bits):
            j = (i + 1) % num_bits
            
            P_i = sqe[t, 2*i]
            b_i = sqe[t, 2*i+1]
            P_j = sqe[t, 2*j]
            b_j = sqe[t, 2*j+1]
            
            if P_i == P_j:
                sign = 1.0 if b_i == b_j else -1.0
                contribution = sign * 9.0
                
                if P_i == 2:
                    xx_sum += contribution
                elif P_i == 3:
                    yy_sum += contribution
                elif P_i == 4:
                    zz_sum += contribution
    
    total_pairs = T * num_bits
    avg_xx = xx_sum / total_pairs
    avg_yy = yy_sum / total_pairs
    avg_zz = zz_sum / total_pairs
    
    energy = J_xy * (avg_xx + avg_yy) + J_xy * Delta * avg_zz
    energy_derivative = J_xy * avg_zz  # dE/dΔ = J_xy * ⟨ZZ⟩
    
    return energy, energy_derivative


def median_of_means_xxz_energy_and_derivative(J_xy, Delta, sqe, num_parts=10):
    """
    同时估计能量和能量导数的中位数平均法
    """
    T, _ = sqe.shape
    part_size = T // num_parts
    
    energy_means = []
    derivative_means = []
    
    for i in range(num_parts):
        part_sqe = sqe[i*part_size : (i+1)*part_size]
        energy, derivative = cal_energy_and_derivative_xxz(J_xy, Delta, part_sqe)
        energy_means.append(energy.item())
        derivative_means.append(derivative.item())
    
    return (torch.median(torch.tensor(energy_means)), 
            torch.median(torch.tensor(derivative_means)))


def cal_magnetization_xxz(sqe):
    """
    从影子数据估计XXZ模型的序参量（总磁化和交错磁化）
    
    Args:
        sqe (torch.Tensor): 影子数据，形状为 [T, 2N]，格式为 [P1, b1, P2, b2, ..., PN, bN]
                           P_i ∈ {2,3,4} 对应 {X,Y,Z}，b_i ∈ {0,1} 对应 {-1,+1}
    Returns:
        m_z (torch.Tensor): 总磁化
        m_s (torch.Tensor): 交错磁化
    """
    T, seq_len = sqe.shape
    num_bits = seq_len // 2
    
    # 初始化磁化累加器
    m_z_sum = torch.tensor(0.0)
    m_s_sum = torch.tensor(0.0)
    
    # 遍历所有影子样本
    for t in range(T):
        # 遍历所有位点
        for i in range(num_bits):
            # 提取测量信息
            P_i = sqe[t, 2*i]      # 位点i的测量基
            b_i = sqe[t, 2*i+1]    # 位点i的测量结果
            
            # 只有测量Z基时才能得到S^z信息
            # 经典影子估计：单点可观测量 O = 3 * U† |b⟩⟨b| U - I
            if P_i == 4:  # Z基测量
                # σ^z的期望值：b_i = 1 -> +1, b_i = 0 -> -1
                sigma_z_exp = 1.0 if b_i == 1 else -1.0
                # S^z = σ^z/2
                sz_value = sigma_z_exp / 2.0
                
                # 累加总磁化
                m_z_sum += sz_value
                # 累加交错磁化（乘以交错因子 (-1)^i）
                m_s_sum += ((-1) ** i) * sz_value
            # 注意：测量X或Y基时，S^z的经典影子估计为0，不需要处理
    
    # 计算平均值
    total_measurements = T * num_bits
    m_z = m_z_sum / total_measurements
    m_s = m_s_sum / total_measurements
    
    return m_z, m_s



def median_of_means_xxz_magnetization(sqe, num_parts=10):
    """
    XXZ模型序参量的中位数平均法估计
    
    Args:
        sqe (torch.Tensor): 影子数据
        num_parts (int): 分区数
    
    Returns:
        median_m_z (torch.Tensor): 中位数平均法估计的总磁化
        median_m_s (torch.Tensor): 中位数平均法估计的交错磁化
    """
    T, _ = sqe.shape
    part_size = T // num_parts
    
    m_z_means = []
    m_s_means = []
    
    for i in range(num_parts):
        part_sqe = sqe[i*part_size : (i+1)*part_size]
        m_z, m_s = cal_magnetization_xxz(part_sqe)
        m_z_means.append(m_z.item())
        m_s_means.append(m_s.item())
    
    return (torch.median(torch.tensor(m_z_means)), 
            torch.median(torch.tensor(m_s_means)))

















Y = torch.tensor([[0., -1j],[1j, 0.]], dtype=torch.cfloat)

operator_map_C = {0: I, 1: X, 2: Z, 3: Y}

def build_two_site_op(num_qubits, i, alpha, j, beta):
    """
    构造算符：σ_i^α ⊗ σ_j^β （其余位置为 I）
    alpha, beta: 'X'=1, 'Z'=2, 'Y'=3
    """
    op_index = np.zeros(num_qubits)
    op_index[i] = alpha
    op_index[j] = beta
    return correlation_func(op_index)

def compute_correlation_tensor(num_bits, h, i, j):
    """
    计算量子比特 i 和 j 之间的自旋关联张量 T_ij
    返回 3x3 torch.Tensor，行=α (i), 列=β (j)
    """
    rho = obtain_eigenrho(num_bits, h)  # 你的基态密度矩阵
    T = torch.zeros(3, 3, dtype=torch.cfloat)  # α,β ∈ {x,y,z}

    # 映射方向到 operator_map 编号
    pauli_idx = {'X': 1, 'Y': 3, 'Z': 2}
    directions = ['X', 'Y', 'Z']

    for a_idx, alpha in enumerate(directions):
        for b_idx, beta in enumerate(directions):
            op = build_two_site_op(num_bits, i, pauli_idx[alpha], j, pauli_idx[beta])
            # 计算期望值 ⟨σ_i^α σ_j^β⟩
            T[a_idx, b_idx] = torch.trace(rho @ op)
    
    return T.real  # 由于哈密顿量是实的，虚部应接近 0


def find_g_key(query_g, g_dict, tol=1e-4):
    keys = np.array(list(g_dict.keys()))
    diffs = np.abs(keys - query_g)
    if diffs.min() < tol:
        return keys[diffs.argmin()]  # 返回最接近的 key
    else:
        raise KeyError(f"No g within {tol} of {query_g}")



class SparseTFI:
    def __init__(self, num_qubits: int):
        self.num_qubits = num_qubits
        self.dim = 2 ** num_qubits
        self.X = sp.csr_matrix([[0, 1], [1, 0]], dtype=complex)
        self.Z = sp.csr_matrix([[1, 0], [0, -1]], dtype=complex)
        self.I = sp.csr_matrix([[1, 0], [0, 1]], dtype=complex)
    
    def _tensor_product(self, matrices: List[sp.csr_matrix]) -> sp.csr_matrix:
        result = matrices[0]
        for mat in matrices[1:]:
            result = sp.kron(result, mat, format='csr')
        return result
    
    def build_hamiltonian(self, h: float) -> sp.csr_matrix:
        H = sp.csr_matrix((self.dim, self.dim), dtype=complex)
        
        # ZZ相互作用项
        for i in range(self.num_qubits - 1):
            ops = [self.I] * self.num_qubits
            ops[i] = self.Z
            ops[i + 1] = self.Z
            ZZ_term = self._tensor_product(ops)
            H += -(1 - h) * ZZ_term
        
        # 周期性边界条件
        ops = [self.I] * self.num_qubits
        ops[0] = self.Z
        ops[-1] = self.Z
        ZZ_boundary = self._tensor_product(ops)
        H += -(1 - h) * ZZ_boundary
        
        # 横向场项
        for i in range(self.num_qubits):
            ops = [self.I] * self.num_qubits
            ops[i] = self.X
            X_term = self._tensor_product(ops)
            H += -h * X_term
        
        return H
    
    def get_ground_state(self, h: float):
        H = self.build_hamiltonian(h)
        eigenvalues, eigenvectors = spla.eigsh(H, k=1, which='SA', maxiter=2000)
        return eigenvalues[0].real, eigenvectors[:, 0]
    
    def expectation_value(self, state: np.ndarray, operator: sp.csr_matrix) -> float:
        """计算算子在给定态下的期望值 - 修复版本"""
        # operator @ state 返回向量
        temp = operator @ state
        # state.conj().T @ temp 返回标量
        result = state.conj().T @ temp
        # 直接取实部，不需要索引
        return np.real(result)
    
    def ZZ_correlation(self, state: np.ndarray, i: int, j: int) -> float:
        ops = [self.I] * self.num_qubits
        ops[i] = self.Z
        ops[j] = self.Z
        ZZ_op = self._tensor_product(ops)
        return self.expectation_value(state, ZZ_op)
    
    def X_string_correlation(self, state: np.ndarray, start: int, length: int) -> float:
        ops = [self.I] * self.num_qubits
        for i in range(start, start + length):
            ops[i % self.num_qubits] = self.X
        X_string_op = self._tensor_product(ops)
        return self.expectation_value(state, X_string_op)
    
    def average_ZZ_correlation(self, state: np.ndarray, distance: int) -> float:
        total = 0.0
        for i in range(self.num_qubits):
            j = (i + distance) % self.num_qubits
            total += self.ZZ_correlation(state, i, j)
        return total / self.num_qubits
    
    def average_X_string_correlation(self, state: np.ndarray, length: int) -> float:
        total = 0.0
        for start in range(self.num_qubits):
            total += self.X_string_correlation(state, start, length)
        return total / self.num_qubits




def calculate_exact_values(num_qubits: int, h_values: np.ndarray):
    tfi = SparseTFI(num_qubits)
    
    results = {
        'num_qubits': num_qubits,
        'h_values': h_values,
        'energy': [],
        'ZZ_d1': [], 'ZZ_d2': [], 'ZZ_d3': [], 'ZZ_d4': [], 'ZZ_d5': [],
        'Xs_l1': [], 'Xs_l2': [], 'Xs_l3': [], 'Xs_l4': [], 'Xs_l5': []
    }
    
    print(f"计算 {num_qubits} 量子比特系统...")
    start_time = time.time()
    
    for i, h in enumerate(h_values):
        energy, ground_state = tfi.get_ground_state(h)
        results['energy'].append(energy)
        
        for d in range(1, 6):
            zz_val = tfi.average_ZZ_correlation(ground_state, d)
            results[f'ZZ_d{d}'].append(zz_val)
        
        for l in range(1, 6):
            xs_val = tfi.average_X_string_correlation(ground_state, l)
            results[f'Xs_l{l}'].append(xs_val)
        
        print(f"进度: {i+1}/{len(h_values)}, h={h:.3f}")
    
    total_time = time.time() - start_time
    print(f"计算完成！耗时: {total_time:.1f}秒")
    ZZ_curves = np.array([results[f'ZZ_d{d}'] for d in range(1, 6)])
    Xs_curves = np.array([results[f'Xs_l{l}'] for l in range(1, 6)])
    
    return results['energy'], ZZ_curves, Xs_curves




# ==================== 内存测量工具函数 ====================


def get_memory_usage(description: str = ""):
    """获取当前内存使用情况 - 新增函数"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    result = {
        'rss_mb': memory_info.rss / 1024**2,  # 物理内存
        'vms_mb': memory_info.vms / 1024**2,  # 虚拟内存
        'description': description
    }
    
    if description:
        print(f"Memory [{description}]: RSS={result['rss_mb']:.1f}MB")
    
    return result

def measure_memory_delta(func, *args, **kwargs):
    """测量函数执行前后的内存变化 - 新增函数"""
    gc.collect()
    
    start_memory = get_memory_usage("Before")
    result = func(*args, **kwargs)
    end_memory = get_memory_usage("After")
    
    memory_delta = {
        'rss_delta_mb': end_memory['rss_mb'] - start_memory['rss_mb'],
        'peak_rss_mb': end_memory['rss_mb']
    }
    
    return result, memory_delta
