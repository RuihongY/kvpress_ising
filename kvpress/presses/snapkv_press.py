# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from transformers.models.llama.modeling_llama import repeat_kv, rotate_half

from kvpress.presses.scorer_press import ScorerPress
from kvpress.qubo_solver import GPUQUBOSolver
from kvpress.utils import get_prerope_query_states

@dataclass
class SnapKVPress(ScorerPress):
    """
    SnapKV + QUBO compression
    
    During Prefill stage:
    1. Use SnapKV for initial compression
    2. Use QUBO to remove redundant tokens
    
    During Generation stage:
    1. No compression (controlled by framework)
    """
    compression_ratio: float = 0.0
    window_size: int = 64
    kernel_size: int = 5
    debug: bool = False
    last_scores: torch.Tensor = None
    
    # QUBO parameters
    qubo_ratio: float = 0.0  # Default 0 means QUBO is not used
    similarity_threshold: float = 0.85
    lambda_penalty: float = 2.0
    lagrange_multiplier: float = 10.0
    
    # QUBO Solver configuration
    qubo_solver_method: str = 'parallel_annealing'
    qubo_num_iterations: int = 100
    qubo_initial_temp: float = 10.0
    qubo_cooling_rate: float = 0.95
    qubo_num_chains: int = 20
    
    def __post_init__(self):
        super().__post_init__()
        # Initialize solver only when QUBO is used
        if self.qubo_ratio > 0:
            self.qubo_solver = GPUQUBOSolver(
                method=self.qubo_solver_method,
                num_iterations=self.qubo_num_iterations,
                initial_temp=self.qubo_initial_temp,
                cooling_rate=self.qubo_cooling_rate,
                num_chains=self.qubo_num_chains,
                debug=self.debug
            )

    def _ensure_qubo_solver(self):
        """Ensure qubo_solver is initialized if qubo_ratio > 0.
        
        This is needed because qubo_ratio might be set after object creation
        (e.g., in evaluate.py's _setup_press method), and __post_init__ only
        runs once at object creation time.
        """
        if self.qubo_ratio > 0 and not hasattr(self, 'qubo_solver'):
            self.qubo_solver = GPUQUBOSolver(
                method=self.qubo_solver_method,
                num_iterations=self.qubo_num_iterations,
                initial_temp=self.qubo_initial_temp,
                cooling_rate=self.qubo_cooling_rate,
                num_chains=self.qubo_num_chains,
                debug=self.debug
            )

    @staticmethod
    def compute_window_attention(module, hidden_states, keys, window_size, position_embeddings):
        """Compute window attention for scoring"""
        bsz, _, k_len, _ = keys.shape
        num_heads = module.config.num_attention_heads
        head_dim = module.head_dim
        num_key_value_groups = num_heads // module.config.num_key_value_heads
        
        query_states = get_prerope_query_states(module, hidden_states[:, -window_size:])
        cos, sin = position_embeddings
        cos, sin = cos[:, -window_size:], sin[:, -window_size:]
        query_states = (query_states * cos.unsqueeze(1)) + (rotate_half(query_states) * sin.unsqueeze(1))

        key_states = repeat_kv(keys, num_key_value_groups)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)
        attention_mask = torch.ones_like(attn_weights) * float("-inf")
        attention_mask = torch.triu(attention_mask, diagonal=k_len - window_size + 1)
        attn_weights += attention_mask
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = attn_weights[..., :-window_size]
        return attn_weights

    def score(self, module, hidden_states, keys, values, attentions, kwargs) -> torch.Tensor:
        """Compute token importance scores"""
        bsz, num_key_value_heads, k_len, _ = keys.shape
        num_key_value_groups = module.config.num_attention_heads // num_key_value_heads

        if attentions is not None:
            attn_weights = attentions[..., -self.window_size:, :-self.window_size]
        else:
            attn_weights = self.compute_window_attention(
                module, hidden_states, keys, self.window_size, kwargs["position_embeddings"]
            )

        scores = attn_weights.mean(dim=-2)
        scores = F.avg_pool1d(scores, kernel_size=self.kernel_size, padding=self.kernel_size // 2, stride=1)
        scores = scores.view(bsz, num_key_value_heads, num_key_value_groups, k_len - self.window_size)
        scores = scores.mean(2)
        scores = F.pad(scores, (0, self.window_size), value=scores.max().item())

        # Save scores for QUBO - keep on GPU to avoid data transfer and dtype issues
        self.last_scores = scores[0, 0].detach()

        return scores

    def compress(self, module, hidden_states, keys, values, attentions, kwargs):
        """
        Compress KV cache
        
        Prefill stage: Execute SnapKV + QUBO compression
        Generation stage: No compression (framework default, not called)
        """
        # If QUBO is not used, directly use parent class's standard SnapKV implementation
        if self.qubo_ratio == 0:
            return super().compress(module, hidden_states, keys, values, attentions, kwargs)
        
        # ========== SnapKV + QUBO Compression ==========
        bsz, num_heads, k_len, head_dim = keys.shape
        device = keys.device
        
        # 1. SnapKV compression (manual implementation to track indices)
        # Compute scores
        scores = self.score(module, hidden_states, keys, values, attentions, kwargs)
        
        # Get indices of tokens to keep
        n_kept_snapkv = int(k_len * (1 - self.compression_ratio))
        snapkv_indices = scores.topk(n_kept_snapkv, dim=-1).indices  # [bsz, num_heads, n_kept_snapkv]
        
        # Critical: Sort indices by original position to maintain temporal order
        # topk returns indices sorted by score, need to re-sort to maintain token temporal order
        snapkv_indices = torch.sort(snapkv_indices, dim=2).values
        
        # Use indices to select keys and values
        snapkv_indices_expanded = snapkv_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
        snap_keys = keys.gather(2, snapkv_indices_expanded).contiguous()
        snap_values = values.gather(2, snapkv_indices_expanded).contiguous()
        
        # 2. QUBO redundancy removal
        if n_kept_snapkv > 20 and self.qubo_ratio > 0:
            # Ensure qubo_solver is initialized (in case qubo_ratio was set after object creation)
            self._ensure_qubo_solver()
            
            # Compute similarity matrix (using average value across all heads)
            v = snap_values[0].mean(dim=0)  # [n_kept_snapkv, head_dim]
            v_norm = F.normalize(v, p=2, dim=-1)
            sim_matrix = torch.matmul(v_norm, v_norm.t())  #  [n_kept_snapkv, n_kept_snapkv]
            
            # Get importance scores (using scores corresponding to tokens selected by SnapKV)
            # Use snapkv_indices to extract correct scores from self.last_scores
            snapkv_indices_first_head = snapkv_indices[0, 0]  # [n_kept_snapkv] - keep on GPU
            importance = self.last_scores[snapkv_indices_first_head]  # Already on correct device
            target_keep = int(n_kept_snapkv * (1 - self.qubo_ratio))
            
            # Build QUBO matrix
            Q = self.qubo_solver.build_qubo_matrix(
                importance=importance,
                sim_matrix=sim_matrix,
                target_keep=target_keep,
                similarity_threshold=self.similarity_threshold,
                lambda_penalty=self.lambda_penalty,
                lagrange_multiplier=self.lagrange_multiplier
            )
            
            # Solve QUBO
            solution, energy = self.qubo_solver.solve(Q, target_keep)
            qubo_kept_indices = torch.where(solution > 0.5)[0]
            
            # Fix constraint (ensure exactly target_keep tokens are selected)
            # Use pure GPU operations to avoid CPU-GPU transfer and dtype issues
            if len(qubo_kept_indices) != target_keep:
                kept_mask = torch.zeros(n_kept_snapkv, dtype=torch.bool, device=device)
                kept_mask[qubo_kept_indices] = True
                
                if len(qubo_kept_indices) < target_keep:
                    # Need to add more tokens: select highest importance from remaining
                    remaining_importance = importance.clone()
                    remaining_importance[kept_mask] = -float('inf')  # Mask already selected
                    
                    n_to_add = target_keep - len(qubo_kept_indices)
                    additional_indices = remaining_importance.topk(n_to_add).indices
                    qubo_kept_indices = torch.cat([qubo_kept_indices, additional_indices])
                else:
                    # Need to reduce tokens: keep top importance ones
                    selected_importance = importance[qubo_kept_indices]
                    top_k_in_selected = selected_importance.topk(target_keep).indices
                    qubo_kept_indices = qubo_kept_indices[top_k_in_selected]
                
                # Sort to maintain temporal order
                qubo_kept_indices = qubo_kept_indices.sort().values
            
            # Extract QUBO-selected tokens from SnapKV results
            qubo_indices_expanded = qubo_kept_indices.unsqueeze(0).unsqueeze(0).unsqueeze(-1).expand(bsz, num_heads, -1, head_dim)
            final_keys = snap_keys.gather(2, qubo_indices_expanded).contiguous()
            final_values = snap_values.gather(2, qubo_indices_expanded).contiguous()
            
            return final_keys, final_values
        else:
            # SnapKV result too small, skip QUBO
            return snap_keys, snap_values
