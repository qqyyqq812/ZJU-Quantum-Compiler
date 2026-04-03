"""
V12 蒙特卡洛树搜索 + 神经网络并行推理引擎 (AlphaZero 架构)
专供宿主机 CPU 挖掘局部极限最优解
"""
import copy
import math
import torch
import numpy as np
from src.compiler.env import QuantumRoutingEnv
from src.compiler.inference_v8 import load_policy
from qiskit.transpiler import CouplingMap

class MCTSNode:
    def __init__(self, env: QuantumRoutingEnv, parent=None, action=None, prior_prob=0.0):
        self.env = env
        self.parent = parent
        self.action = action
        self.prior_prob = prior_prob
        
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0.0
        self.is_expanded = False
        
        self.reward_from_parent = 0.0
        self.is_terminal = False

    @property
    def q_value(self):
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def expand(self, action_probs: dict, is_terminal: bool):
        self.is_terminal = is_terminal
        self.is_expanded = True
        if not is_terminal:
            for action, prob in action_probs.items():
                if action not in self.children:
                    # 只有真正访问时再深拷贝环境，节省内存
                    self.children[action] = MCTSNode(env=None, parent=self, action=action, prior_prob=prob)

class MCTSSearcher:
    def __init__(self, policy_net, cm: CouplingMap, c_puct=1.5, num_simulations=50):
        self.policy_net = policy_net
        self.cm = cm
        self.c_puct = c_puct
        self.num_simulations = num_simulations
        self.device = next(policy_net.parameters()).device

    def search(self, initial_env: QuantumRoutingEnv) -> int:
        root = MCTSNode(copy.deepcopy(initial_env))
        obs = root.env._get_obs()
        action_probs, value = self._evaluate_env(root.env, obs)
        root.expand(action_probs, False)

        for _ in range(self.num_simulations):
            node = root
            search_path = [node]

            # 1. Selection
            while node.is_expanded and not node.is_terminal:
                action, node = self._select_child(node)
                search_path.append(node)
                
                # 懒加载环境拷贝
                if node.env is None:
                    node.env = copy.deepcopy(node.parent.env)
                    # 先行使其前进一步
                    obs, reward, term, trunc, info = node.env.step(node.action)
                    node.reward_from_parent = reward
                    node.is_terminal = term or trunc
                    
                    if not node.is_terminal:
                        probs, val = self._evaluate_env(node.env, obs)
                        node.expand(probs, node.is_terminal)
                        leaf_value = val
                    else:
                        leaf_value = 0.0 # 终局
                    
                    # 2. Backpropagation 面向本次新增节点的价值传播
                    self._backpropagate(search_path, leaf_value)
                    break
            else:
                # 碰到终端节点，直接回传
                self._backpropagate(search_path, 0.0)

        # 3. Play - 挑选访问次数最多的根节点动作
        best_action = max(root.children.items(), key=lambda item: item[1].visit_count)[0]
        return best_action
        
    def _select_child(self, node: MCTSNode) -> tuple[int, MCTSNode]:
        best_score = -float('inf')
        best_action = -1
        best_child = None
        
        for action, child in node.children.items():
            u = self.c_puct * child.prior_prob * math.sqrt(node.visit_count) / (1 + child.visit_count)
            score = child.q_value + u
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
        return best_action, best_child

    def _backpropagate(self, search_path: list[MCTSNode], value: float):
        for node in reversed(search_path):
            node.value_sum += value
            node.visit_count += 1
            # 价值逆向传播（包含跳跃的 reward）
            value = node.reward_from_parent + 0.99 * value

    def _evaluate_env(self, env: QuantumRoutingEnv, obs: np.ndarray) -> tuple[dict, float]:
        mask = env.get_action_mask()
        info = env._get_info()
        gnn_input = info.get('gnn_input')
        
        from torch_geometric.data import Batch
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            if gnn_input and 'graph' in gnn_input:
                d_b = Batch.from_data_list([gnn_input['graph']]).to(self.device)
                dist, value = self.policy_net(obs_t, d_b, [gnn_input['swap_edges']])
                logits = dist.logits.squeeze(0)
            else:
                logits = torch.zeros(env.action_space.n, device=self.device)
                value = torch.tensor([[0.0]], device=self.device)

            mask_t = torch.tensor(mask, dtype=torch.float32, device=self.device)
            logits = logits.masked_fill(mask_t == 0, -1e8)
            probs = torch.softmax(logits, dim=0).cpu().numpy()
            
            valid_actions = np.where(mask > 0)[0]
            action_probs = {act: probs[act] for act in valid_actions}
            
            # 强化探索：给所有合法动作叠加轻微 Dirichlet 噪声
            if len(action_probs) > 0:
                noise = np.random.dirichlet([0.3] * len(action_probs))
                for i, act in enumerate(valid_actions):
                    action_probs[act] = 0.75 * action_probs[act] + 0.25 * noise[i]
            
            return action_probs, value.item()


def run_mcts_eval(model_path: str, topology_name: str, num_sims: int = 20):
    policy, cm = load_policy(model_path, topology_name)
    policy.eval()
    
    from src.benchmarks.circuits import generate_qft
    qc = generate_qft(5)
    
    env = QuantumRoutingEnv(coupling_map=cm, soft_mask=False, tabu_size=4, max_steps=100)
    env.set_circuit(qc)
    obs, info = env.reset()
    
    searcher = MCTSSearcher(policy, cm, num_simulations=num_sims)
    
    print(f"🌲 正在启动 MCTS 主机引擎 (模型: {model_path})")
    
    done = False
    step = 0
    while not done:
        # 对每一步进行树搜索
        action = searcher.search(env)
        obs, reward, term, trunc, info = env.step(action)
        step += 1
        done = term or trunc
        print(f"推演步数 {step:2d} | 树决策选择了动作: {action} | 最新 SWAP 消耗: {info.get('total_swaps')}")
        
    print(f"🎉 MCTS 路由完成！最终 SWAP 消耗: {info.get('total_swaps')}")

if __name__ == '__main__':
    run_mcts_eval("models/v10_hard_mask/v7_ibm_tokyo_best.pt", "ibm_tokyo", num_sims=30)
