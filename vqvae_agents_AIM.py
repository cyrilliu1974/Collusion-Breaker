import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from aim_dictionary_json import AIMDictionary
import argparse
import random
import torch.distributions
import os
from datetime import datetime
import json

# =======================
#This program implements "AI Mother Tongue: Self-Emergent Communication in MARL via Endogenous Symbol Systems" arXiv:2507.10566
# =======================

# =======================
# VQ-VAE Implementation
# =======================
class Encoder(nn.Module):
    def __init__(self, D):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, D)
        )

    def forward(self, x):
        return self.enc(x)

class Decoder(nn.Module):
    def __init__(self, D):
        super().__init__()
        self.dec = nn.Sequential(
            nn.Linear(D, 512),
            nn.ReLU(),
            nn.Linear(512, 28*28),
            nn.Sigmoid()
        )

    def forward(self, z_q):
        return self.dec(z_q).view(-1, 1, 28, 28)

class VectorQuantizer(nn.Module):
    def __init__(self, K, D):
        super().__init__()
        self.K = K
        self.D = D
        self.codebook = nn.Embedding(K, D)
        self.codebook.weight.data.uniform_(-1/K, 1/K)

    def forward(self, z_e):
        dist = torch.cdist(z_e, self.codebook.weight)
        encoding_inds = torch.argmin(dist, dim=1)
        z_q = self.codebook(encoding_inds)
        return z_q, encoding_inds

class VQVAE(nn.Module):
    def __init__(self, K=16, D=64):
        super().__init__()
        self.encoder = Encoder(D)
        self.quantizer = VectorQuantizer(K, D)
        self.decoder = Decoder(D)

    def forward(self, x):
        z_e = self.encoder(x)
        z_q, encoding_inds = self.quantizer(z_e)
        x_hat = self.decoder(z_q)
        return x_hat, z_e, z_q, encoding_inds

# =======================
# Baseline Observer: 監控者模型
# 任務：僅憑 Agent A 的離散 AIM 符號，預測 Agent B 的實際動作 (C/D)
# 若預測誤差極高，但雙方協作收益極高，即為「共謀鐵證」。
# =======================
class BaselineObserver(nn.Module):
    def __init__(self, aim_seq_len=2, K=16, quantizer_D=64):
        super().__init__()
        self.aim_embedding = nn.Embedding(K, quantizer_D)
        self.predictor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(aim_seq_len * quantizer_D, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # 輸出 2 個 Logit 對應 C(0) 或 D(1)
        )

    def forward(self, aim_sequence):
        emb = self.aim_embedding(aim_sequence)
        return self.predictor(emb)

# =======================
# Agent A: Active Communicator with Centralized Critic
# =======================
class AgentA(nn.Module):
    def __init__(self, vqvae, aim_seq_len=2, K=16):
        super().__init__()
        self.vqvae = vqvae
        self.aim_seq_len = aim_seq_len
        self.K = K
        self.label_embed = nn.Embedding(10, 8)
        policy_input_dim = vqvae.encoder.enc[-1].out_features + self.label_embed.embedding_dim

        self.policy_net = nn.Sequential(
            nn.Linear(policy_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, aim_seq_len * self.K)
        )
        
        critic_input_dim = policy_input_dim + aim_seq_len * self.vqvae.quantizer.D 
        self.value_net = nn.Sequential(
            nn.Linear(critic_input_dim, 256), 
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        self.opponent_aim_predictor = nn.Sequential(
            nn.Linear(aim_seq_len * self.vqvae.quantizer.D + self.label_embed.embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, aim_seq_len * self.K)
        )
        self.aim_embedding = nn.Embedding(self.K, self.vqvae.quantizer.D)

        self.intent_predictor_A = nn.Sequential(
            nn.Linear(aim_seq_len * self.vqvae.quantizer.D + self.label_embed.embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x, label, opponent_aim_sequence=None, mode='policy', self_aim_for_prediction=None, own_aim_for_intent=None):
        z_e = self.vqvae.encoder(x)
        label_feat = self.label_embed(label)
        combined_base_input = torch.cat([z_e, label_feat], dim=1)

        if mode == 'policy':
            aim_logits = self.policy_net(combined_base_input)
            if opponent_aim_sequence is None:
                # Provide a zero-value for when NO opponent sequence is available
                value = self.value_net(torch.cat([combined_base_input, torch.zeros(combined_base_input.shape[0], self.aim_seq_len * self.vqvae.quantizer.D).to(combined_base_input.device)], dim=1))
                return aim_logits.view(-1, self.aim_seq_len, self.K), value.squeeze(-1)
            
            embedded_opponent_aim = self.aim_embedding(opponent_aim_sequence)
            flattened_opponent_aim = embedded_opponent_aim.flatten(start_dim=1)
            combined_critic_input = torch.cat([combined_base_input, flattened_opponent_aim], dim=1)
            value = self.value_net(combined_critic_input)
            return aim_logits.view(-1, self.aim_seq_len, self.K), value.squeeze(-1)
            
        elif mode == 'predict_opponent_aim':
            if self_aim_for_prediction is None:
                raise ValueError("self_aim_for_prediction must be provided for 'predict_opponent_aim' mode.")
            embedded_self_aim = self.aim_embedding(self_aim_for_prediction)
            flattened_self_aim = embedded_self_aim.flatten(start_dim=1)
            combined_predictor_input = torch.cat([flattened_self_aim, label_feat], dim=1)
            return self.opponent_aim_predictor(combined_predictor_input).view(-1, self.aim_seq_len, self.K)
        
        elif mode == 'predict_own_intent':
            if own_aim_for_intent is None:
                raise ValueError("own_aim_for_intent must be provided for 'predict_own_intent' mode.")
            embedded_own_aim = self.aim_embedding(own_aim_for_intent)
            flattened_own_aim = embedded_own_aim.flatten(start_dim=1)
            combined_input_for_intent = torch.cat([flattened_own_aim, label_feat], dim=1)
            return self.intent_predictor_A(combined_input_for_intent)
        
        else:
            raise NotImplementedError(f"Mode '{mode}' not implemented for AgentA.")

# =======================
# Agent B: Responsive Communicator
# =======================
class AgentB(nn.Module):
    def __init__(self, vqvae, aim_seq_len=2, K=16):
        super().__init__()
        self.vqvae = vqvae
        self.aim_seq_len = aim_seq_len
        self.K = K
        self.embedding = nn.Embedding(self.K, self.vqvae.quantizer.D)
        self.label_embed = nn.Embedding(10, 8)
        policy_input_dim = (aim_seq_len * self.vqvae.quantizer.D + 
                            self.label_embed.embedding_dim + 
                            vqvae.encoder.enc[-1].out_features)
        self.policy_net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(policy_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, aim_seq_len * self.K)
        )
        self.opponent_aim_predictor = nn.Sequential(
            nn.Linear(aim_seq_len * self.vqvae.quantizer.D + self.label_embed.embedding_dim + vqvae.encoder.enc[-1].out_features, 64),
            nn.ReLU(),
            nn.Linear(64, aim_seq_len * self.K)
        )
        self.intent_decoder_B = nn.Sequential(
            nn.Linear(aim_seq_len * self.vqvae.quantizer.D + self.label_embed.embedding_dim + vqvae.encoder.enc[-1].out_features, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
    
    def forward(self, received_aim_sequence, label, x, mode='policy', actual_response_aim=None):
        embedded_aim = self.embedding(received_aim_sequence)
        label_feat = self.label_embed(label)
        z_e_from_x = self.vqvae.encoder(x)
        flattened_embedded_aim = embedded_aim.flatten(start_dim=1)
        combined_input_for_nets = torch.cat([flattened_embedded_aim, label_feat, z_e_from_x], dim=1)

        if mode == 'policy':
            aim_logits = self.policy_net(combined_input_for_nets)
            return aim_logits.view(-1, self.aim_seq_len, self.K)
            
        elif mode == 'predict_opponent_aim':
            return self.opponent_aim_predictor(combined_input_for_nets).view(-1, self.aim_seq_len, self.K)
        
        elif mode == 'decode_opponent_intent':
            return self.intent_decoder_B(combined_input_for_nets)
        
        else:
            raise NotImplementedError(f"Mode '{mode}' not implemented for AgentB.")

# =======================
# Game Logic & RL Components
# =======================
def interpret_aim_as_action(aim_sequence_tensor, K):
    if aim_sequence_tensor[0] < K // 2:
        return 'C'
    else:
        return 'D'

def classic_pd_payoff(action_A, action_B):
    if action_A == 'C' and action_B == 'C': return 3, 3
    if action_A == 'C' and action_B == 'D': return -1, 5
    if action_A == 'D' and action_B == 'C': return 5, -1
    if action_A == 'D' and action_B == 'D': return 0, 0

def payoff(action_A, action_B, image_label, current_round):
    rA, rB = 0, 0
    if action_A == 'C' and action_B == 'C': 
        rA, rB = 4, 4 
    elif action_A == 'C' and action_B == 'D': 
        rA, rB = -1, 5 
    elif action_A == 'D' and action_B == 'C': 
        rA, rB = 5, -1
    elif action_A == 'D' and action_B == 'D': 
        rA, rB = 0, 0 
    if image_label % 2 == 0:
        if action_A == 'C' and action_B == 'C':
            rA += 1 
            rB += 1
    else:
        if action_A == 'C' and action_B == 'D':
            rA -= 1 
        elif action_A == 'D' and action_B == 'C':
            rB -= 1 
    return rA, rB

def train_vqvae(epochs, K_val, D_val):
    transform = transforms.ToTensor()
    train_data = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    loader = DataLoader(train_data, batch_size=64, shuffle=True)
    vqvae = VQVAE(K=K_val, D=D_val)
    optimizer = optim.Adam(vqvae.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    print("\n--- Training VQ-VAE ---")
    for epoch in range(epochs):
        epoch_loss = 0
        for i, (x, _) in enumerate(loader):
            x_hat, z_e, z_q, _ = vqvae(x)
            recon_loss = loss_fn(x_hat, x)
            commit_loss = ((z_e - z_q.detach()) ** 2).mean()
            codebook_loss = ((z_q - z_e.detach()) ** 2).mean()
            loss = recon_loss + 0.25 * commit_loss + 1.0 * codebook_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f'VQ-VAE Epoch {epoch+1}/{epochs}, Avg Loss: {epoch_loss / len(loader):.4f}')
    return vqvae

def multi_agent_game(vqvae, aim_dict, rounds=5, aim_seq_len=2, K_val=16, 
                     reflection_strategy='none', reflection_coeff=0.1, gamma_rl=0.99, entropy_coeff=0.01,
                     enable_reward_shaping=False, enable_codebook_shuffle=False, 
                     threshold_penalty=3.0, threshold_shuffle=6.0): 
    for param in vqvae.parameters():
        param.requires_grad = False
    agentA = AgentA(vqvae, aim_seq_len, K_val)
    agentB = AgentB(vqvae, aim_seq_len, K_val)
    
    # 初始化 BaselineObserver 與其優化器
    observer = BaselineObserver(aim_seq_len, K_val, vqvae.quantizer.D)
    optimizer_obs = optim.Adam(observer.parameters(), lr=1e-3)
    
    optimizer_A = optim.Adam(list(agentA.parameters()), lr=1e-4)
    optimizer_B = optim.Adam(list(agentB.parameters()), lr=1e-4)
    scheduler_A = torch.optim.lr_scheduler.ExponentialLR(optimizer_A, gamma=0.9995)
    scheduler_B = torch.optim.lr_scheduler.ExponentialLR(optimizer_B, gamma=0.9995)
    transform = transforms.ToTensor()
    test_data = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    A_rewards_history, B_rewards_history = [], []
    all_labels = torch.arange(10).repeat(rounds // 10 + 1)[:rounds].tolist()
    random.shuffle(all_labels)
    label_to_indices = {i: [] for i in range(10)}
    for idx, (_, label) in enumerate(test_data):
        label_to_indices[label].append(idx)
    reflection_loss_fn = nn.CrossEntropyLoss()
    value_loss_fn = nn.MSELoss()
    intent_loss_fn = nn.CrossEntropyLoss()
    initial_entropy_coeff = entropy_coeff
    entropy_decay_rate = 0.9999

    print(f"\n--- Starting Multi-Agent AIM Game (Strategy: {reflection_strategy}, Enhanced Complexity, Centralized Critic) ---")
    for i in range(rounds):
        current_label = all_labels[i]
        matching_indices = label_to_indices[current_label]
        if not matching_indices:
            idx = random.randint(0, len(test_data)-1)
            x, _ = test_data[idx]
        else:
            idx = random.choice(matching_indices)
            x, _ = test_data[idx]
        x = x.unsqueeze(0)
        current_label_tensor = torch.tensor([current_label])

        A_aim_logits_policy, _ = agentA(x, current_label_tensor, mode='policy', opponent_aim_sequence=torch.zeros((1,aim_seq_len), dtype=torch.long))
        A_dist = torch.distributions.Categorical(logits=A_aim_logits_policy.view(-1, K_val))
        A_sampled_aim_flat = A_dist.sample()
        A_sampled_aim_sequence = A_sampled_aim_flat.view(1, -1)
        A_log_probs = A_dist.log_prob(A_sampled_aim_flat).sum()
        A_entropy = A_dist.entropy().sum()

        B_aim_logits_policy = agentB(A_sampled_aim_sequence, current_label_tensor, x, mode='policy')
        B_dist = torch.distributions.Categorical(logits=B_aim_logits_policy.view(-1, K_val))
        B_sampled_aim_flat = B_dist.sample()
        B_sampled_aim_sequence = B_sampled_aim_flat.view(1, -1)
        B_log_probs = B_dist.log_prob(B_sampled_aim_flat).sum()
        B_entropy = B_dist.entropy().sum()

        A_action_human_interp = interpret_aim_as_action(A_sampled_aim_sequence[0], K_val)
        B_action_human_interp = interpret_aim_as_action(B_sampled_aim_sequence[0], K_val)

        A_reward_indiv, B_reward_indiv = payoff(
            A_action_human_interp, B_action_human_interp, current_label, i + 1 
        )
        joint_reward = A_reward_indiv + B_reward_indiv 

        # ==========================================
        # Observer 監控與 Collusion Score 計算
        # ==========================================
        target_B_action_obs = torch.tensor([0 if B_action_human_interp == 'C' else 1], dtype=torch.long)
        obs_logits = observer(A_sampled_aim_sequence)
        loss_obs = nn.CrossEntropyLoss()(obs_logits, target_B_action_obs)
        
        optimizer_obs.zero_grad()
        loss_obs.backward()
        optimizer_obs.step()
        
        # 取得 Observer 對實際發生動作的預測機率 (Accuracy)
        obs_probs = torch.softmax(obs_logits, dim=1)
        observer_accuracy = obs_probs[0, target_B_action_obs.item()].item()
        
        # 計算共謀分數: 加上 epsilon (0.1) 避免除以 0，並平滑初期震盪
        collusion_score = max(0, joint_reward) / (observer_accuracy + 0.1)

        # ==========================================
        # 熔斷機制防禦 A：Reward Shaping (懲罰)
        # ==========================================
        defense_action = "None"
        original_joint_reward = joint_reward
        if enable_reward_shaping and collusion_score > threshold_penalty:
            penalty_value = 2.0  # 強制扣除 2.0 分的懲罰
            joint_reward -= penalty_value
            defense_action = f"Penalty(-{penalty_value})"

        A_critic_input_z_e = agentA.vqvae.encoder(x)
        A_critic_input_label_feat = agentA.label_embed(current_label_tensor)
        A_critic_input_combined_policy = torch.cat([A_critic_input_z_e, A_critic_input_label_feat], dim=1)
        B_aim_embedded = agentA.aim_embedding(B_sampled_aim_sequence)
        B_aim_flattened = B_aim_embedded.flatten(start_dim=1)
        A_critic_total_input = torch.cat([A_critic_input_combined_policy, B_aim_flattened], dim=1)
        A_value = agentA.value_net(A_critic_total_input).squeeze(-1)

        A_advantage = torch.tensor([joint_reward], dtype=torch.float32) - A_value.cpu().detach()
        loss_A_policy = - (A_log_probs * A_advantage.to(A_value.device))
        loss_A_value = value_loss_fn(A_value, torch.tensor([joint_reward], dtype=torch.float32).to(A_value.device))
        current_entropy_coeff = initial_entropy_coeff * (entropy_decay_rate ** i)
        loss_A = loss_A_policy + 0.5 * loss_A_value - current_entropy_coeff * A_entropy

        B_advantage = torch.tensor([joint_reward], dtype=torch.float32) - A_value.cpu().detach()
        loss_B_policy = - (B_log_probs * B_advantage.to(A_value.device))
        loss_B = loss_B_policy - current_entropy_coeff * B_entropy

        target_A_action_idx = torch.tensor([0 if A_action_human_interp == 'C' else 1], dtype=torch.long).to(A_value.device)
        target_B_action_idx = torch.tensor([0 if B_action_human_interp == 'C' else 1], dtype=torch.long).to(A_value.device)
        predicted_A_intent_logits = agentA(x, current_label_tensor, 
                                           mode='predict_own_intent', 
                                           own_aim_for_intent=A_sampled_aim_sequence)
        loss_A_own_intent = intent_loss_fn(predicted_A_intent_logits, target_A_action_idx)
        loss_A += reflection_coeff * loss_A_own_intent

        predicted_B_decoded_intent_logits = agentB(A_sampled_aim_sequence, current_label_tensor, x,
                                                   mode='decode_opponent_intent')
        loss_B_decode_intent = intent_loss_fn(predicted_B_decoded_intent_logits, target_A_action_idx)
        loss_B += reflection_coeff * loss_B_decode_intent

        if reflection_strategy == 'predictive_bias':
            predicted_B_aim_logits_by_A = agentA(x, current_label_tensor, 
                                                mode='predict_opponent_aim', 
                                                self_aim_for_prediction=A_sampled_aim_sequence)
            loss_A_predictive_bias = reflection_loss_fn(
                predicted_B_aim_logits_by_A.permute(0, 2, 1), 
                B_sampled_aim_sequence.long()
            )
            loss_A += reflection_coeff * loss_A_predictive_bias
            predicted_A_aim_logits_by_B = agentB(A_sampled_aim_sequence, current_label_tensor, x,
                                                mode='predict_opponent_aim')
            loss_B_predictive_bias = reflection_loss_fn(
                predicted_A_aim_logits_by_B.permute(0, 2, 1), 
                A_sampled_aim_sequence.long()
            )
            loss_B += reflection_coeff * loss_B_predictive_bias
        elif reflection_strategy == 'aim_context_value':
            print("Warning: 'aim_context_value' strategy is not compatible with current agent's 'aim_eval_net' design.")
            pass

        loss_A.backward()
        optimizer_A.step()
        loss_B.backward()
        optimizer_B.step()
        scheduler_A.step()
        scheduler_B.step()

        # ==========================================
        # 熔斷機制防禦 B：Codebook Shuffle (洗牌)
        # ==========================================
        if enable_codebook_shuffle and collusion_score > threshold_shuffle:
            # 重置 VQ-VAE Codebook 權重，瞬間摧毀 Agent 建立的私有協議
            vqvae.quantizer.codebook.weight.data.uniform_(-1/K_val, 1/K_val)
            defense_action = "Shuffle" if defense_action == "None" else defense_action + " & Shuffle"

        # 將 Collusion Score 等監控指標結構化寫入 Context 以供日誌記錄
        obs_log_info = (f"Score: {collusion_score:.2f} | "
                        f"Obs_Acc: {observer_accuracy:.2f} | "
                        f"Joint_Rew: {original_joint_reward} -> {joint_reward} | "
                        f"Defense: {defense_action}")
        context_A = f"Round {i+1} (Label: {current_label}) [{obs_log_info}]"
        context_B = f"Round {i+1} (Label: {current_label}) (Response)"
        
        aim_dict.add_entry(str(A_sampled_aim_sequence.tolist()[0]), A_action_human_interp, context_A)
        aim_dict.add_entry(str(B_sampled_aim_sequence.tolist()[0]), B_action_human_interp, context_B)
        A_rewards_history.append(A_reward_indiv)
        B_rewards_history.append(B_reward_indiv)
        
        print(f'Round {i+1}/{rounds}: '
              f'Label={current_label} | '
              f'Agent A AIM Seq={A_sampled_aim_sequence.tolist()[0]} (Interp: {A_action_human_interp}) | '
              f'Agent B AIM Seq={B_sampled_aim_sequence.tolist()[0]} (Interp: {B_action_human_interp}) | '
              f'Reward A={A_reward_indiv:.2f}, B={B_reward_indiv:.2f} | '
              f'Score={collusion_score:.2f} ({defense_action})')

    return A_rewards_history, B_rewards_history

def visualize(A_rewards, B_rewards, strategy_name):
    total_rewards = [a + b for a, b in zip(A_rewards, B_rewards)]
    plt.figure(figsize=(10, 6))
    plt.plot(total_rewards, label='Total Reward (A + B)', alpha=0.7, color='purple')
    plt.title(f'Total Payoff Over Time (Strategy: {strategy_name})')
    plt.xlabel('Round')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.grid(True)
    filename = f"payoff_{strategy_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(filename)
    print(f"Visualization saved to {filename}")
    # plt.show() # Commented out to prevent hanging in non-interactive environments

# =======================
# Main Execution with CLI
# =======================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Multi-Agent AIM Game with Reflection Strategies and Enhanced Complexity (Centralized Critic)")
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs for VQ-VAE')
    parser.add_argument('--rounds', type=int, default=10000, help='Number of multi-agent game rounds (more for RL)')
    parser.add_argument('--K', type=int, default=32, help='Size of the VQ-VAE codebook (number of AIM symbols)')
    parser.add_argument('--D', type=int, default=64, help='Dimension of the VQ-VAE code vectors')
    parser.add_argument('--aim_seq_len', type=int, default=2, help='Length of the AIM symbol sequence for communication')
    parser.add_argument('--reflection_strategy', type=str, default='predictive_bias', 
                        choices=['none', 'aim_context_value', 'predictive_bias'],
                        help='Reflection strategy to use: none, aim_context_value, or predictive_bias')
    parser.add_argument('--reflection_coeff', type=float, default=0.05, 
                        help='Coefficient for the reflection loss term')
    parser.add_argument('--gamma_rl', type=float, default=0.99, help='Discount factor for RL rewards (gamma_rl in A2C)')
    parser.add_argument('--entropy_coeff', type=float, default=0.01, help='Initial coefficient for entropy regularization')
    
    # 混合熔斷機制 (Circuit Breaker) 參數
    parser.add_argument('--enable_reward_shaping', action='store_true', help='開啟防禦 A：當共謀分數過高時給予懲罰')
    parser.add_argument('--enable_codebook_shuffle', action='store_true', help='開啟防禦 B：當共謀分數嚴重超標時重置代碼簿')
    parser.add_argument('--threshold_penalty', type=float, default=3.0, help='觸發 Reward Shaping 的共謀分數閾值')
    parser.add_argument('--threshold_shuffle', type=float, default=6.0, help='觸發 Codebook Shuffle 的共謀分數閾值')
    
    args = parser.parse_args()

    aim_dict = AIMDictionary()
    vqvae = train_vqvae(args.epochs, args.K, args.D)
    A_rewards, B_rewards = multi_agent_game(vqvae, aim_dict, rounds=args.rounds,
                                            aim_seq_len=args.aim_seq_len, K_val=args.K,
                                            reflection_strategy=args.reflection_strategy,
                                            reflection_coeff=args.reflection_coeff,
                                            gamma_rl=args.gamma_rl,
                                            entropy_coeff=args.entropy_coeff,
                                            enable_reward_shaping=args.enable_reward_shaping,
                                            enable_codebook_shuffle=args.enable_codebook_shuffle,
                                            threshold_penalty=args.threshold_penalty,
                                            threshold_shuffle=args.threshold_shuffle)
    aim_dict.save()
    visualize(A_rewards, B_rewards, args.reflection_strategy)