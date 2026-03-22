"""
Quantum Router MCTS (Monte Carlo Tree Search)
=============================================
V4: 结合 PPO 策略价值网络的 MCTS 搜索器。
用于在真实编译（非训练）阶段，通过前瞻搜索打破局部极值，寻找全局最优路由段。
"""

import math
import copy
import torch
import numpy as np

class MCTSNode:
    def __init__(self, state_obs, state_info, env_state, parent=None, action=None, prior_prob=1.0):
        self.state_obs = state_obs
        self.state_info = state_info
        self.env_state = env_state  # 深拷贝的环境状态
        self.parent = parent
        self.action = action        # 导致此节点的 action
        self.children = {}          # action -> MCTSNode
        
        self.N = 0                  # 访问次数
        self.W = 0.0                # 累计价值
        self.Q = 0.0                # 平均价值
        self.P = prior_prob         # 先验概率 (来自 PPO Actor)
        
        self.is_terminal = env_state.unwrapped._dag.is_done() if env_state else False
        self.reward = 0.0           # 环境单步 reward

    def expand(self, action_probs, valid_actions):
        """展开子节点"""
        for a in valid_actions:
            if a not in self.children:
                # 初始不实例化 env，等 select 到了再 step
                self.children[a] = MCTSNode(
                    state_obs=None, state_info=None, env_state=None,
                    parent=self, action=a, prior_prob=action_probs[a]
                )

    def is_expanded(self):
        return len(self.children) > 0

    def get_ucb_action(self, c_puct=1.5):
        """应用 PUCT 算法选择最佳子节点"""
        best_u = -float('inf')
        best_a = -1
        
        for a, child in self.children.items():
            if child.N == 0:
                q_value = 0.0
            else:
                q_value = child.Q
            
            # PUCT formula
            u = q_value + c_puct * child.P * math.sqrt(self.N) / (1 + child.N)
            if u > best_u:
                best_u = u
                best_a = a
                
        return best_a

    def backup(self, value):
        """回溯更新节点价值"""
        self.N += 1
        self.W += value
        self.Q = self.W / self.N
        if self.parent:
            # 价值随步数衰减 (Gamma)
            self.parent.backup(self.reward + 0.99 * value)


class RouterMCTS:
    def __init__(self, policy, num_simulations=50, c_puct=1.5):
        self.policy = policy
        self.num_simulations = num_simulations
        self.c_puct = c_puct

    def search(self, initial_env, initial_obs, initial_info):
        """对当前环境状态进行 MCTS 搜索并返回最佳 Action"""
        root = MCTSNode(initial_obs, initial_info, copy.deepcopy(initial_env))
        
        # 预取根节点的先验概率和 Value
        with torch.no_grad():
            action_mask = root.env_state.unwrapped.get_action_mask()
            gnn_input = root.state_info.get('gnn_input')
            
            # evaluate 返回 action, log_prob, entropy, values
            # 直接使用 PPO 的 actor logits 计算 probs
            obs_t = torch.tensor(initial_obs, dtype=torch.float32).unsqueeze(0)
            
            if self.policy.use_gnn and gnn_input is not None:
                from torch_geometric.data import Batch
                d_b = Batch.from_data_list([gnn_input['dag']])
                c_b = Batch.from_data_list([gnn_input['coupling']])
                gnn_feat = self.policy.gnn(d_b, c_b)
                obs_t = torch.cat([obs_t, gnn_feat], dim=-1)
            features = self.policy.shared(obs_t)
            logits = self.policy.actor(features)
            value = self.policy.critic(features).squeeze(-1)
            
            mask_t = torch.tensor(action_mask, dtype=torch.float32).unsqueeze(0)
            logits = logits.masked_fill(mask_t == 0, -1e8)
            probs = torch.softmax(logits, dim=-1)[0].numpy()
            
        valid_actions = np.where(action_mask > 0)[0]
        root.expand(probs, valid_actions)
        
        for _ in range(self.num_simulations):
            node = root
            search_path = [node]
            
            # 1. Select
            while node.is_expanded() and not node.is_terminal:
                action = node.get_ucb_action(self.c_puct)
                node = node.children[action]
                search_path.append(node)
                
                # Lazy initialization: 如果子环境未建立，则 Step 并建立
                if node.env_state is None:
                    node.env_state = copy.deepcopy(node.parent.env_state)
                    obs, reward, term, trunc, info = node.env_state.step(action)
                    node.state_obs = obs
                    node.state_info = info
                    node.reward = reward
                    node.is_terminal = term or trunc
                    break
                    
            # 2. Evaluate
            value = 0.0
            if not node.is_terminal:
                with torch.no_grad():
                    action_mask = node.env_state.unwrapped.get_action_mask()
                    obs_t = torch.tensor(node.state_obs, dtype=torch.float32).unsqueeze(0)
                    gnn_input = node.state_info.get('gnn_input')
                    
                    if self.policy.use_gnn and gnn_input is not None:
                        from torch_geometric.data import Batch
                        d_b = Batch.from_data_list([gnn_input['dag']])
                        c_b = Batch.from_data_list([gnn_input['coupling']])
                        gnn_feat = self.policy.gnn(d_b, c_b)
                        obs_t = torch.cat([obs_t, gnn_feat], dim=-1)
                    features = self.policy.shared(obs_t)
                    logits = self.policy.actor(features)
                    v_t = self.policy.critic(features).squeeze(-1)
                    value = v_t.item()
                    
                    mask_t = torch.tensor(action_mask, dtype=torch.float32).unsqueeze(0)
                    logits = logits.masked_fill(mask_t == 0, -1e8)
                    probs = torch.softmax(logits, dim=-1)[0].numpy()
                    
                valid_actions = np.where(action_mask > 0)[0]
                # 3. Expand
                node.expand(probs, valid_actions)
            elif node.is_terminal and node.env_state.unwrapped._dag.is_done():
                value = node.env_state.unwrapped.reward_done
                
            # 4. Backup
            node.backup(value)
            
        # 选择访问次数最多的动作
        best_a = max(root.children.items(), key=lambda x: x[1].N)[0]
        return best_a
