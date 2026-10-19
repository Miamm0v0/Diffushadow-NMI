import torch 
import torch.nn as nn
from numericalmodel import NumericalDiffusionModel
import json
import torch.nn.functional as F
# from Qdmodel_oseq import QuantumDiffusionModel_oseq
from Qdmodel_nseq import QuantumDiffusionModel_nseq
from Qmultimodel import QuantumMultimodalDModel_nseq

MASK_TOKEN_ID = -1.0
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# model_path = r'/home/fyp26lyh/QuantumLLADA/DshadowGPT/model_train/Quantum_diffusion_model.pth'
# model_path = r'/home/fyp26lyh/QuantumLLADA/DshadowGPT/model_train/Qdmodel_diffusion_model_mini_wo_amask_6002.pth'
# # model = NumericalDiffusionModel()
# model = QuantumDiffusionModel_nseq()
# model.load_state_dict(torch.load(model_path, map_location=device))

# model.to(device)
# model.eval()

    


def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def generate_nseq(model, prompt, steps, C=None, mask_id=MASK_TOKEN_ID, num_qubits=10, temperature=1.0, use_sampling=False, conf_decode=True):
    """
    从全掩码状态逐步生成：每一步只 unmask 最自信的 num_unmask 个
    Args:
        model: Diffusion model
        prompt: (1, L, 1)
        steps: 总步数
        mask_id: MASK 标识
    Returns:
        生成结果 (1, L, 1)
    """
    model.eval()
    prompt = prompt.clone()  # [1, L, 1]
    if C is not None:
        C = C.clone()
    is_masked = (prompt.squeeze(-1) == mask_id)  # 初始 mask 位置 [L]
    total_masked = is_masked.sum().item()
    num_unmask_per_step = max(1, total_masked // steps)  # 每步恢复多少个

    if conf_decode:
        for i in range(steps):
            # print(f"\n=== Step {i+1}/{steps} ===")
            is_still_masked = (prompt.squeeze(-1) == mask_id)  # [1, L], 当前仍被 mask 的位置
            if not is_still_masked[:, 1+num_qubits:2*num_qubits+1].any():
                print("All target positions unmasked, stopping early.")
                break

            # --- 模型前向 ---
            with torch.no_grad():
                if C is not None:
                    # print("C is not None")
                    # print("C:", C)
                    logits, logits_c = model(x_single=prompt, x_pair=C, mask_indices=is_still_masked[:, 1+num_qubits:2*num_qubits+1])
                else:
                    logits = model(prompt, mask_indices=is_still_masked[:, 1+num_qubits:2*num_qubits+1])  # 传入 mask_indices
            # print(logits)
            logits = logits / temperature 
            # p = torch.sigmoid(logits)  # P(y=1) [1, L, 1]
            probs = torch.softmax(logits, dim=-1)
            p_b1 = probs[..., 1]

            eps = 1e-12
            entropy = - (probs * torch.log(probs + eps)).sum(dim=-1)  # [1, num_qubits], H(p)
            # print("shape of entropy:", entropy.shape)
            # print("entropy:", entropy)

            # 只看仍然被 mask 的位置
            current_masked_b = is_still_masked[:, 1+num_qubits:2*num_qubits+1]  # [1, num_qubits]
            # print("shape of current_masked_b:", current_masked_b.shape)
            # print("current_masked_b:", current_masked_b)
            entropy_masked = entropy[current_masked_b]  # [K], 只取被 mask 的位置

            num_unmask = min(num_unmask_per_step, entropy_masked.size(0))
            if num_unmask == 0:
                break
            # print("confidence:", confidence.cpu().tolist())

            # if i == 0:
            #     # 第一步：随机选一些位置恢复（打破先验）
            #     idx_in_masked = torch.randperm(len(entropy_masked))[:num_unmask]
            # else:
            #     # 后续步骤：低熵优先
            _, idx_in_masked = torch.topk(entropy_masked, num_unmask, largest=False)

            # 映射回 b_i 局部索引 (0~9)
            all_masked_local = torch.nonzero(current_masked_b[0], as_tuple=False).squeeze(1)  # [K]
            selected_local_b = all_masked_local[idx_in_masked]  # [num_unmask]

            # 转为全局索引 (11~20)
            selected_global_idx = 1 + num_qubits + selected_local_b  # [num_unmask]

            # 采样预测值
            p_selected = p_b1[0, selected_local_b]  # [num_unmask]
            if use_sampling:
                pred_values = torch.bernoulli(p_selected).float()  # 0. 或 1.
            else:
                pred_values = (p_selected > 0.5).float()

            # 更新 prompt
            prompt[0, selected_global_idx, 0] = pred_values  # 直接赋值
    else:
        for i in range(steps):
            # print(f"\n=== Step {i+1}/{steps} ===")
            is_still_masked = (prompt.squeeze(-1) == mask_id)  # [1, L], 当前仍被 mask 的位置
            if not is_still_masked[:, 1+num_qubits:2*num_qubits+1].any():
                print("All target positions unmasked, stopping early.")
                break

            # 获取当前仍被 mask 的 b 区域位置（局部索引 0～num_qubits-1）
            current_masked_b = is_still_masked[:, 1+num_qubits:2*num_qubits+1]  # [1, num_qubits]
            all_masked_local = torch.nonzero(current_masked_b[0], as_tuple=False).squeeze(1)  # [K]

            if all_masked_local.numel() == 0:
                break

            num_unmask = min(num_unmask_per_step, all_masked_local.size(0))

            # 随机打乱并选取前 num_unmask 个
            rand_perm = torch.randperm(all_masked_local.size(0))
            selected_local_b = all_masked_local[rand_perm[:num_unmask]]  # [num_unmask]

            # 转为全局索引
            selected_global_idx = 1 + num_qubits + selected_local_b  # [num_unmask]

            # 模型前向（用于获取预测值）
            with torch.no_grad():
                if C is not None:
                    logits = model(x_single=prompt, x_pair=C, mask_indices=current_masked_b)
                else:
                    logits = model(prompt, mask_indices=current_masked_b)

            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            p_b1 = probs[..., 1]

            # 采样或确定性预测
            p_selected = p_b1[0, selected_local_b]
            if use_sampling:
                pred_values = torch.bernoulli(p_selected).float()
            else:
                pred_values = (p_selected > 0.5).float()

            # 更新 prompt
            prompt[0, selected_global_idx, 0] = pred_values

    return prompt


def generate_nseq_batch(
    model,
    prompt,
    steps,
    C=None,
    mask_id=MASK_TOKEN_ID,
    num_qubits=10,
    temperature=1.0,
    use_sampling=False,
    conf_decode=True,
):
    """
    batched 版本的 nseq 生成：
    - prompt: [B, L, 1]
    - C: 可选，[B, ...]（与模型 forward 保持一致）
    - 返回: [B, L, 1]
    """
    model.eval()
    prompt = prompt.clone()
    if C is not None:
        C = C.clone()

    B, L, _ = prompt.shape
    b_slice = slice(1 + num_qubits, 1 + 2 * num_qubits)

    is_masked = (prompt.squeeze(-1) == mask_id)  # [B, L]
    total_masked = is_masked[:, b_slice].sum(dim=1)  # [B]
    # 取 batch 内最大，保证每步至少能推进；小样本也不会被 0 卡住
    num_unmask_per_step = torch.clamp(total_masked.max() // max(1, steps), min=1).item()

    if conf_decode:
        for _ in range(steps):
            is_still_masked = (prompt.squeeze(-1) == mask_id)  # [B, L]
            current_masked_b = is_still_masked[:, b_slice]  # [B, N]
            if not current_masked_b.any():
                break

            with torch.no_grad():
                if C is not None:
                    logits, _logits_c = model(
                        x_single=prompt,
                        x_pair=C,
                        mask_indices=current_masked_b,
                    )
                else:
                    logits = model(prompt, mask_indices=current_masked_b)

            probs = torch.softmax(logits / temperature, dim=-1)  # [B, N, 2]
            p_b1 = probs[..., 1]  # [B, N]
            eps = 1e-12
            entropy = - (probs * torch.log(probs + eps)).sum(dim=-1)  # [B, N]

            # 对于未 mask 的位置，用 +inf 屏蔽掉
            entropy_masked = entropy.masked_fill(~current_masked_b, float("inf"))  # [B, N]

            # 每个样本各自挑选 k 个最低熵的位置
            k = min(num_unmask_per_step, num_qubits)
            idx = torch.topk(entropy_masked, k=k, largest=False, dim=1).indices  # [B, k]

            # 可能存在该样本 mask 数 < k，topk 会返回 inf 对应的位置；过滤掉这些
            gathered_entropy = entropy_masked.gather(1, idx)  # [B, k]
            valid = torch.isfinite(gathered_entropy)  # [B, k]
            if not valid.any():
                break

            # 采样/阈值预测
            p_selected = p_b1.gather(1, idx)  # [B, k]
            if use_sampling:
                pred_values = torch.bernoulli(p_selected).float()
            else:
                pred_values = (p_selected > 0.5).float()

            # 写回到 prompt 的全局索引
            global_idx = (1 + num_qubits) + idx  # [B, k]

            b_idx = torch.arange(B, device=prompt.device).unsqueeze(1).expand_as(global_idx)  # [B, k]
            b_idx = b_idx[valid]
            g_idx = global_idx[valid]
            v = pred_values[valid]
            prompt[b_idx, g_idx, 0] = v
    else:
        for _ in range(steps):
            is_still_masked = (prompt.squeeze(-1) == mask_id)  # [B, L]
            current_masked_b = is_still_masked[:, b_slice]  # [B, N]
            if not current_masked_b.any():
                break

            # 随机选取每个样本要 unmask 的位置
            # 用随机噪声 +inf 屏蔽未 mask，然后取最小 k 个
            rand = torch.rand((B, num_qubits), device=prompt.device)
            rand = rand.masked_fill(~current_masked_b, float("inf"))
            k = min(num_unmask_per_step, num_qubits)
            idx = torch.topk(rand, k=k, largest=False, dim=1).indices  # [B, k]
            valid = torch.isfinite(rand.gather(1, idx))
            if not valid.any():
                break

            with torch.no_grad():
                if C is not None:
                    logits = model(x_single=prompt, x_pair=C, mask_indices=current_masked_b)
                else:
                    logits = model(prompt, mask_indices=current_masked_b)

            probs = torch.softmax(logits / temperature, dim=-1)
            p_b1 = probs[..., 1]
            p_selected = p_b1.gather(1, idx)
            if use_sampling:
                pred_values = torch.bernoulli(p_selected).float()
            else:
                pred_values = (p_selected > 0.5).float()

            global_idx = (1 + num_qubits) + idx
            b_idx = torch.arange(B, device=prompt.device).unsqueeze(1).expand_as(global_idx)
            b_idx = b_idx[valid]
            g_idx = global_idx[valid]
            v = pred_values[valid]
            prompt[b_idx, g_idx, 0] = v

    return prompt





def generate_nseq_both(model, prompt, prompt_c, steps, mask_id=MASK_TOKEN_ID, num_qubits=10, temperature=1.0, use_sampling=False):
    """
    同时生成b和c：每一步只 unmask 最自信的 num_unmask 个位置
    """
    model.eval()
    prompt = prompt.clone()  # [1, 21, 1]
    prompt_c = prompt_c.clone()  # [1, 31, 1]

    # 初始mask位置
    is_masked_b = (prompt.squeeze(-1) == mask_id)  # [1, 21]
    total_masked_b = is_masked_b[:, 1+num_qubits:2*num_qubits+1].sum().item()  # b的mask数
    
    prompt_flat_c = prompt_c.squeeze(-1)  
    p_positions = torch.arange(3, 31, 3, device=device)
    p_values = prompt_flat_c[:, p_positions]  # [1, 10]
    is_masked_c = (p_values == mask_id)  # [1, 10]
    total_masked_c = is_masked_c.sum().item()  # c的mask数
    
    num_unmask_per_step = max(1, (total_masked_b + total_masked_c) // steps)

    for i in range(steps):
        # 当前仍被mask的位置
        is_still_masked_b = (prompt.squeeze(-1) == mask_id)[:, 1+num_qubits:2*num_qubits+1]  # [1, 10]
        prompt_flat_c = prompt_c.squeeze(-1)  
        p_values = prompt_flat_c[:, p_positions]
        is_still_masked_c = (p_values == mask_id)  # [1, 10]
        
        # 检查是否还有需要生成的位置
        if not is_still_masked_b.any() and not is_still_masked_c.any():
            # print("All target positions unmasked, stopping early.")
            break

        # print(f"prompt c: {prompt_c.cpu().numpy().flatten()}")
        # print(f"prompt: {prompt.cpu().numpy().flatten()}")
        # --- 模型前向 ---
        with torch.no_grad():
            logits_b, logits_c = model(
                x_single=prompt, 
                x_pair=prompt_c, 
                mask_indices=is_still_masked_b, 
                mask_indices_pair=is_still_masked_c
            )
        
        # 处理b的预测
        probs_b = torch.softmax(logits_b / temperature, dim=-1)  # [1, 10, 2]
        p_b1 = probs_b[..., 1]  # [1, 10]
        
        # 处理c的预测  
        probs_c = torch.softmax(logits_c / temperature, dim=-1)  # [1, 10, 2]
        p_c1 = probs_c[..., 1]  # [1, 10]
        
        # 计算熵（不确定性）
        eps = 1e-12
        entropy_b = - (probs_b * torch.log(probs_b + eps)).sum(dim=-1)  # [1, 10]
        entropy_c = - (probs_c * torch.log(probs_c + eps)).sum(dim=-1)  # [1, 10]
        
        # 合并b和c的熵，选择最确定的位置
        combined_entropy = torch.cat([entropy_b[is_still_masked_b], entropy_c[is_still_masked_c]], dim=0)
        
        num_unmask = min(num_unmask_per_step, combined_entropy.size(0))
        if num_unmask == 0:
            break
            
        # 选择熵最低（最确定）的位置
        _, idx_in_combined = torch.topk(combined_entropy, num_unmask, largest=False)
        
        # 确定选中的是b还是c位置
        num_masked_b = is_still_masked_b.sum().item()
        selected_b_mask = idx_in_combined < num_masked_b
        selected_c_mask = ~selected_b_mask
        
        # 处理选中的b位置
        if selected_b_mask.any():
            selected_b_idx_in_combined = idx_in_combined[selected_b_mask]
            all_masked_b_local = torch.nonzero(is_still_masked_b[0], as_tuple=False).squeeze(1)  # [K_b]
            selected_b_local = all_masked_b_local[selected_b_idx_in_combined]  # [num_selected_b]
            
            # 转为全局索引并更新prompt
            selected_b_global = 1 + num_qubits + selected_b_local
            p_selected_b = p_b1[0, selected_b_local]
            
            if use_sampling:
                pred_b_values = torch.bernoulli(p_selected_b).float()
            else:
                pred_b_values = (p_selected_b > 0.5).float()
                
            prompt[0, selected_b_global, 0] = pred_b_values
        
        # 处理选中的c位置
        if selected_c_mask.any():
            selected_c_idx_in_combined = idx_in_combined[selected_c_mask] - num_masked_b
            all_masked_c_local = torch.nonzero(is_still_masked_c[0], as_tuple=False).squeeze(1)  # [K_c]
            selected_c_local = all_masked_c_local[selected_c_idx_in_combined]  # [num_selected_c]
            
            # 转为c的全局索引并更新prompt_c
            selected_c_global = p_positions[selected_c_local]  # 3,6,9,...,30
            p_selected_c = p_c1[0, selected_c_local]
            
            if use_sampling:
                pred_c_values = torch.bernoulli(p_selected_c).float()
            else:
                pred_c_values = (p_selected_c > 0.5).float()
                
            prompt_c[0, selected_c_global, 0] = pred_c_values

    return prompt, prompt_c






def generate_oseq(model, prompt, steps, mask_id=MASK_TOKEN_ID, temperature=1.0, use_sampling=False):
    """
    从全掩码状态逐步生成：每一步只 unmask 最自信的 num_unmask 个
    Args:
        model: Diffusion model
        prompt: (1, L, 1)，oseq 布局 [g, P1,b1, ..., PN,bN]，L=1+2N
        steps: 总步数
        mask_id: MASK 标识
    Returns:
        生成结果 (1, L, 1)
    """
    model.eval()
    prompt = prompt.clone()  # [1, L, 1]
    L = prompt.shape[1]
    b_mask_slice = slice(2, L, 2)

    is_masked = (prompt.squeeze(-1) == mask_id)  # 初始 mask 位置 [L]
    total_masked = is_masked.sum().item()
    num_unmask_per_step = max(1, total_masked // steps)  # 每步恢复多少个

    for i in range(steps):
        # print(f"\n=== Step {i+1}/{steps} ===")
        is_still_masked = (prompt.squeeze(-1) == mask_id)  # [1, L], 当前仍被 mask 的位置
        if not is_still_masked[:, b_mask_slice].any():
            print("All target positions unmasked, stopping early.")
            break

        # --- 模型前向 ---
        with torch.no_grad():
            logits = model(prompt, mask_indices=is_still_masked[:, b_mask_slice])  # 传入 mask_indices
        logits = logits / temperature 
        # p = torch.sigmoid(logits)  # P(y=1) [1, L, 1]
        probs = torch.softmax(logits, dim=-1)
        p_b1 = probs[..., 1]

        eps = 1e-12
        entropy = - (probs * torch.log(probs + eps)).sum(dim=-1)  # [1, N], H(p)

        # 只看仍然被 mask 的位置
        current_masked_b = is_still_masked[:, b_mask_slice]  # [1, N]
        entropy_masked = entropy[current_masked_b]  # [K], 只取被 mask 的位置

        num_unmask = min(num_unmask_per_step, entropy_masked.size(0))
        if num_unmask == 0:
            break
        # print("confidence:", confidence.cpu().tolist())

        # topk 最大置信度 → 最自信的
        # _, idx_in_masked = torch.topk(entropy_masked, num_unmask, largest=False)

        # if i == 0:
        #     # 第一步：随机选一些位置恢复（打破先验）
        #     idx_in_masked = torch.randperm(len(entropy_masked))[:num_unmask]
        # else:
        #     # 后续步骤：低熵优先
        _, idx_in_masked = torch.topk(entropy_masked, num_unmask, largest=False)

        # 映射回 b_i 局部索引 (0~9)
        all_masked_local = torch.nonzero(current_masked_b[0], as_tuple=False).squeeze(1)  # [K]
        selected_local_b = all_masked_local[idx_in_masked]  # [num_unmask]

        # 转为全局索引 (11~20)
        selected_global_idx = 2 + 2 * selected_local_b  # [num_unmask]

        # 采样预测值
        p_selected = p_b1[0, selected_local_b]  # [num_unmask]
        if use_sampling:
            pred_values = torch.bernoulli(p_selected).float()  # 0. 或 1.
        else:
            pred_values = (p_selected > 0.5).float()

        # 更新 prompt
        prompt[0, selected_global_idx, 0] = pred_values  # 直接赋值

    return prompt


def generate_oseq_batch(model, prompt, steps, mask_id=MASK_TOKEN_ID, temperature=1.0, use_sampling=False):
    """
    batched 版本的 oseq 生成：
    - prompt: [B, L, 1]，oseq 布局 [g, P1,b1, ..., PN,bN]，L=1+2N
    - 返回: [B, L, 1]
    """
    model.eval()
    prompt = prompt.clone()
    B, L, _ = prompt.shape
    b_positions = torch.arange(2, L, 2, device=prompt.device)  # [N]
    N = b_positions.numel()

    is_masked = (prompt.squeeze(-1) == mask_id)  # [B, L]
    total_masked = is_masked[:, b_positions].sum(dim=1)
    num_unmask_per_step = torch.clamp(total_masked.max() // max(1, steps), min=1).item()

    for _ in range(steps):
        is_still_masked = (prompt.squeeze(-1) == mask_id)  # [B, L]
        current_masked_b = is_still_masked[:, b_positions]  # [B, N]
        if not current_masked_b.any():
            break

        with torch.no_grad():
            logits = model(prompt, mask_indices=current_masked_b)  # [B, N, 2]

        probs = torch.softmax(logits / temperature, dim=-1)
        p_b1 = probs[..., 1]  # [B, N]
        eps = 1e-12
        entropy = - (probs * torch.log(probs + eps)).sum(dim=-1)  # [B, N]
        entropy_masked = entropy.masked_fill(~current_masked_b, float("inf"))

        k = min(num_unmask_per_step, N)
        idx = torch.topk(entropy_masked, k=k, largest=False, dim=1).indices  # [B, k]
        gathered_entropy = entropy_masked.gather(1, idx)
        valid = torch.isfinite(gathered_entropy)
        if not valid.any():
            break

        p_selected = p_b1.gather(1, idx)
        if use_sampling:
            pred_values = torch.bernoulli(p_selected).float()
        else:
            pred_values = (p_selected > 0.5).float()

        # idx 是局部 b_i 索引，映射到全局序列位置：2 + 2*i
        global_pos = 2 + 2 * idx  # [B, k]
        b_idx = torch.arange(B, device=prompt.device).unsqueeze(1).expand_as(global_pos)
        b_idx = b_idx[valid]
        g_idx = global_pos[valid]
        v = pred_values[valid]
        prompt[b_idx, g_idx, 0] = v

    return prompt







# with open(r'/home/fyp26lyh/QuantumLLADA/data/TFI_test_data_nseq.json', 'r') as f:
#     test_data = json.load(f)
# test_sample = test_data[14]
# test_tensor = torch.tensor(test_sample, dtype=torch.float32).unsqueeze(0).to(device)
# cond = test_tensor[:, :11]
# seq = test_tensor[:, 11:]
# mask_indices = torch.ones_like(seq, dtype=torch.bool)
# seq_masked = seq.clone()
# seq_masked[mask_indices] = MASK_TOKEN_ID
# test_tensor_masked = torch.cat([cond, seq_masked], dim=1)



# x_input = test_tensor_masked.unsqueeze(-1)  # [1, 21, 1]


# with torch.no_grad():
#     pred = generate(model=model, prompt=x_input, steps=5, mask_id=MASK_TOKEN_ID,  temperature=1.0, use_sampling=True)  # 输出 [1, 21, 1]

# gen_seq = pred.cpu().numpy()[0]


# # ================================
# # 6. 输出结果
# # ================================
# print("原始后10个值:", test_tensor[0, 11:].cpu().tolist())
# print("掩码后输入  :", test_tensor_masked[0, 11:].cpu().tolist())
# # print("预测概率   :", pred_prob[0].cpu().tolist())
# print("预测结果(0/1):", gen_seq)

# # 如果你想得到 Python list
# pred_list = pred_binary[0].cpu().numpy().astype(int).tolist()
# print("预测结果 (list):", pred_list)