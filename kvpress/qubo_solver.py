# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple, Optional
import math
import numpy as np


class GPUQUBOSolver:
    """GPU加速的QUBO求解器"""
    
    def __init__(
        self,
        method: str = 'parallel_annealing',  # 'parallel_annealing', 'simulated_annealing', 'greedy'
        num_iterations: int = 100,
        initial_temp: float = 10.0,
        cooling_rate: float = 0.95,
        num_chains: int = 20,
        debug: bool = False
    ):
        """
        Args:
            method: 求解方法 ('parallel_annealing', 'simulated_annealing', 'greedy')
            num_iterations: 迭代次数
            initial_temp: 初始温度
            cooling_rate: 降温速率
            num_chains: 并行链数量
            debug: 是否打印调试信息
        """
        self.method = method
        self.num_iterations = num_iterations
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.num_chains = num_chains
        self.debug = debug
        
    def solve(
        self, 
        Q: torch.Tensor, 
        target_keep: int,
        initial_solution: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, float]:
        """
        求解QUBO问题
        
        Args:
            Q: QUBO矩阵 [n, n]
            target_keep: 目标选择数量
            initial_solution: 初始解 [n] (可选)
            
        Returns:
            solution: 二进制解 [n]
            energy: 最优能量值
        """
        device = Q.device
        
        if self.method == 'parallel_annealing':
            solution, energy = self._parallel_annealing(Q, target_keep, device)
        elif self.method == 'simulated_annealing':
            solution, energy = self._simulated_annealing(Q, target_keep, device, initial_solution)
        elif self.method == 'greedy':
            solution, energy = self._greedy_solve(Q, target_keep, device)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        if self.debug:
            print(f"[QUBOSolver] Method={self.method}, Energy={energy:.4f}, Selected={solution.sum().item()}/{target_keep}")
        
        return solution, energy
    
    def _compute_energy(self, x: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
        """计算QUBO能量: x^T * Q * x"""
        return torch.sum(x * (Q @ x), dim=-1)
    
    def _simulated_annealing(
        self, 
        Q: torch.Tensor, 
        target_keep: int, 
        device: torch.device,
        initial_solution: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, float]:
        """单链模拟退火"""
        n = Q.shape[0]
        
        # 初始化解
        if initial_solution is not None:
            x = initial_solution.clone().to(device)
        else:
            x = torch.zeros(n, device=device)
            indices = torch.randperm(n, device=device)[:target_keep]
            x[indices] = 1.0
        
        best_x = x.clone()
        best_energy = self._compute_energy(x, Q)
        current_energy = best_energy.clone()
        
        temp = self.initial_temp
        
        for iteration in range(self.num_iterations):
            # 随机翻转一个比特
            flip_idx = torch.randint(0, n, (1,), device=device).item()
            x_new = x.clone()
            x_new[flip_idx] = 1.0 - x_new[flip_idx]
            
            # 计算新能量
            energy_new = self._compute_energy(x_new, Q)
            energy_diff = energy_new - current_energy
            
            # Metropolis准则
            if energy_diff < 0 or torch.rand(1, device=device) < torch.exp(-energy_diff / temp):
                x = x_new
                current_energy = energy_new
                
                if energy_new < best_energy:
                    best_x = x.clone()
                    best_energy = energy_new
            
            # 降温
            temp *= self.cooling_rate
        
        return best_x, best_energy.item()
    
    def _parallel_annealing(
        self, 
        Q: torch.Tensor, 
        target_keep: int, 
        device: torch.device
    ) -> Tuple[torch.Tensor, float]:
        """并行多链模拟退火"""
        n = Q.shape[0]
        num_chains = self.num_chains
        
        # 并行初始化多条链
        x = torch.zeros(num_chains, n, device=device)
        for i in range(num_chains):
            indices = torch.randperm(n, device=device)[:target_keep]
            x[i, indices] = 1.0
        
        # 计算初始能量
        current_energies = torch.stack([self._compute_energy(x[i], Q) for i in range(num_chains)])
        best_energies = current_energies.clone()
        best_solutions = x.clone()
        
        temp = self.initial_temp
        
        for iteration in range(self.num_iterations):
            # 每条链随机翻转
            flip_indices = torch.randint(0, n, (num_chains,), device=device)
            x_new = x.clone()
            
            # 批量翻转
            chain_indices = torch.arange(num_chains, device=device)
            x_new[chain_indices, flip_indices] = 1.0 - x_new[chain_indices, flip_indices]
            
            # 批量计算能量
            energies_new = torch.stack([self._compute_energy(x_new[i], Q) for i in range(num_chains)])
            energy_diffs = energies_new - current_energies
            
            # 批量Metropolis判断
            accept_probs = torch.exp(-energy_diffs / temp)
            random_vals = torch.rand(num_chains, device=device)
            accepts = (energy_diffs < 0) | (random_vals < accept_probs)
            
            # 更新接受的链
            x[accepts] = x_new[accepts]
            current_energies[accepts] = energies_new[accepts]
            
            # 更新最优解
            improved = energies_new < best_energies
            best_energies[improved] = energies_new[improved]
            best_solutions[improved] = x_new[improved]
            
            # 降温
            temp *= self.cooling_rate
        
        # 返回最优链的解
        best_chain_idx = torch.argmin(best_energies)
        return best_solutions[best_chain_idx], best_energies[best_chain_idx].item()
    
    def _greedy_solve(
        self, 
        Q: torch.Tensor, 
        target_keep: int, 
        device: torch.device
    ) -> Tuple[torch.Tensor, float]:
        """贪心算法求解"""
        n = Q.shape[0]
        
        # 从对角线获取重要性（负对角线元素）
        importance = -Q.diagonal()
        
        # 按重要性排序
        sorted_indices = torch.argsort(importance, descending=True)
        
        # 贪心选择
        selected = sorted_indices[:target_keep]
        
        # 构建解向量
        solution = torch.zeros(n, device=device)
        solution[selected] = 1.0
        
        # 计算能量
        energy = self._compute_energy(solution, Q)
        
        return solution, energy.item()
    
    def build_qubo_matrix(
        self,
        importance: torch.Tensor,
        sim_matrix: torch.Tensor,
        target_keep: int,
        similarity_threshold: float = 0.95,
        lambda_penalty: float = 2.0,
        lagrange_multiplier: float = 10.0
    ) -> torch.Tensor:
        """
        构建QUBO矩阵
        
        Args:
            importance: 重要性分数 [n]
            sim_matrix: 相似度矩阵 [n, n]
            target_keep: 目标保留数量
            similarity_threshold: 相似度阈值
            lambda_penalty: 相似度惩罚系数
            lagrange_multiplier: Lagrange乘子
            
        Returns:
            Q: QUBO矩阵 [n, n]
        """
        device = importance.device
        n = len(importance)
        
        # 初始化QUBO矩阵
        Q = torch.zeros(n, n, device=device)
        
        # 对角线：-importance（最大化重要性）
        Q.diagonal().copy_(-importance)
        
        # 非对角线：相似度惩罚
        mask = (sim_matrix > similarity_threshold) & ~torch.eye(n, dtype=torch.bool, device=device)
        Q = Q + (mask.float() * sim_matrix * lambda_penalty)
        
        # 添加基数约束（选择恰好target_keep个）
        Q.diagonal().add_(lagrange_multiplier * (1 - 2 * target_keep))
        upper_triangle = torch.triu(torch.ones(n, n, device=device), diagonal=1)
        Q = Q + 2 * lagrange_multiplier * upper_triangle
        
        return Q
