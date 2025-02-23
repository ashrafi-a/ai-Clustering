import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.cluster import DBSCAN
import random
from datetime import datetime
import logging
import os
from typing import Tuple, Dict, Optional, List

# تعریف کلاس TorqueClusteringCore با استفاده از DBSCAN
class TorqueClusteringCore:
    def __init__(self, k_neighbors=5, threshold_percentile=90):
        self.k_neighbors = k_neighbors
        self.threshold_percentile = threshold_percentile
        self.labels_ = None
        self.X = None
        self.eps = None

    def fit(self, X: np.ndarray):
        self.X = X
        # محاسبه فاصله‌ها برای تعیین eps
        if len(X) > 1:
            distances = []
            for i in range(len(X)):
                for j in range(i + 1, len(X)):
                    distances.append(np.linalg.norm(X[i] - X[j]))
            self.eps = np.percentile(distances, self.threshold_percentile)
        else:
            self.eps = 1.0  # مقدار پیش‌فرض

        # استفاده از DBSCAN برای خوشه‌بندی
        clustering = DBSCAN(eps=self.eps, min_samples=self.k_neighbors).fit(X)
        self.labels_ = clustering.labels_

# کلاس خوشه‌بندی گشتاور برای تجربیات
class ExperienceClustering:
    def __init__(self, k_neighbors=5, threshold_percentile=90):
        self.torque_clustering = TorqueClusteringCore(
            k_neighbors=k_neighbors,
            threshold_percentile=threshold_percentile
        )
        self.state_clusters = {}
        self.action_patterns = {}

    def update_clusters(self, states: np.ndarray, actions: np.ndarray, rewards: np.ndarray):
        """به‌روزرسانی خوشه‌های تجربه"""
        self.torque_clustering.fit(states)
        
        for cluster_idx in np.unique(self.torque_clustering.labels_):
            if cluster_idx != -1:  # نادیده گرفتن نقاط پرت
                cluster_mask = self.torque_clustering.labels_ == cluster_idx
                cluster_actions = actions[cluster_mask]
                cluster_rewards = rewards[cluster_mask]
                
                successful_actions = cluster_actions[cluster_rewards > np.mean(cluster_rewards)]
                if len(successful_actions) > 0:
                    self.action_patterns[cluster_idx] = np.bincount(successful_actions).argmax()

    def get_suggested_action(self, state: np.ndarray, available_actions: List[int]) -> Optional[int]:
        """پیشنهاد اقدام بر اساس خوشه‌بندی"""
        if not self.action_patterns:
            return None

        state = state.reshape(1, -1)
        cluster_centers = np.array([
            np.mean(self.torque_clustering.X[self.torque_clustering.labels_ == i], axis=0)
            for i in np.unique(self.torque_clustering.labels_) if i != -1
        ])
        
        if len(cluster_centers) == 0:
            return None

        distances = np.linalg.norm(cluster_centers - state, axis=1)
        nearest_cluster = np.argmin(distances)
        
        suggested_action = self.action_patterns.get(nearest_cluster)
        if suggested_action in available_actions:
            return suggested_action
        return None

# بافر بازپخش بهبود یافته با خوشه‌بندی
class ClusteredReplayBuffer:
    def __init__(self, capacity=10000, clustering_update_freq=100):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        self.clustering_update_freq = clustering_update_freq
        self.experience_clustering = ExperienceClustering()
        self.steps_since_clustering = 0

    def push(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, done: bool) -> None:
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
        
        self.steps_since_clustering += 1
        if self.steps_since_clustering >= self.clustering_update_freq:
            self._update_clustering()

    def _update_clustering(self):
        """به‌روزرسانی خوشه‌بندی تجربیات"""
        if len(self.buffer) < self.clustering_update_freq:
            return

        states, actions, rewards = zip(*[(s, a, r) for s, a, r, _, _ in self.buffer])
        self.experience_clustering.update_clusters(
            np.array(states),
            np.array(actions),
            np.array(rewards)
        )
        self.steps_since_clustering = 0

    def get_suggested_action(self, state: np.ndarray, available_actions: List[int]) -> Optional[int]:
        return self.experience_clustering.get_suggested_action(state, available_actions)

    def sample(self, batch_size: int) -> Tuple:
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(dones)
        )

# تعریف شبکه سیاست
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super(PolicyNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.net(x)

# تعریف شبکه ارزش
class ValueNetwork(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 128):
        super(ValueNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x)

# تعریف کلاس پایه GRPO
class ImprovedGRPO:
    def __init__(self, policy_network: PolicyNetwork, value_network: ValueNetwork,
                 learning_rate: float = 0.001, gamma: float = 0.99, gae_lambda: float = 0.95):
        self.policy_network = policy_network
        self.value_network = value_network
        self.policy_optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        self.value_optimizer = optim.Adam(self.value_network.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.gae_lambda = gae_lambda

    def update(self, states, actions, old_probs, rewards, next_states, dones):
        # محاسبه بازده‌ها با GAE
        values = self.value_network(states)
        next_values = self.value_network(next_states)
        returns = self.compute_gae(rewards, values, next_values, dones)
        
        # به‌روزرسانی شبکه ارزش
        value_loss = nn.MSELoss()(values.squeeze(), returns)
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()
        
        # به‌روزرسانی شبکه سیاست
        new_probs = self.policy_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        ratio = new_probs / (old_probs + 1e-5)
        advantages = (returns - values.squeeze()).detach()
        policy_loss = -torch.mean(ratio * advantages)
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        return {'policy_loss': policy_loss.item(), 'value_loss': value_loss.item()}

    def compute_gae(self, rewards, values, next_values, dones):
        advantages = torch.zeros_like(rewards)
        gae = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * next_values[t] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
        return values.squeeze() + advantages

# عامل GRPO بهبود یافته با خوشه‌بندی
class ClusteredGRPO(ImprovedGRPO):
    def __init__(self, *args, clustering_weight=0.3, **kwargs):
        super().__init__(*args, **kwargs)
        self.clustering_weight = clustering_weight

    def select_action(self, state: np.ndarray, buffer: 'ClusteredReplayBuffer') -> int:
        """انتخاب اقدام با در نظر گرفتن پیشنهادات خوشه‌بندی"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            probs = self.policy_network(state_tensor).squeeze(0)
            available_actions = list(range(len(probs)))
            
            suggested_action = buffer.get_suggested_action(state, available_actions)
            
            if suggested_action is not None and random.random() < self.clustering_weight:
                return suggested_action
            
            return torch.multinomial(probs, 1).item()

# تابع تنظیم logging
def setup_logging(log_dir: str):
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(log_dir, f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        level=logging.INFO,
        format='%(asctime)s - %(message)s'
    )

# تابع ذخیره checkpoint
def save_checkpoint(policy_net: PolicyNetwork, value_net: ValueNetwork, episode: int, 
                   reward: float, checkpoint_dir: str):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint = {
        'episode': episode,
        'policy_state_dict': policy_net.state_dict(),
        'value_state_dict': value_net.state_dict(),
        'reward': reward
    }
    torch.save(checkpoint, os.path.join(checkpoint_dir, f'checkpoint_episode_{episode}.pth'))

# تابع آموزش عامل
def train_clustered_agent(
    env_name: str = "CartPole-v1",
    num_episodes: int = 1000,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    hidden_dim: int = 128,
    patience: int = 50,
    checkpoint_freq: int = 100,
    clustering_weight: float = 0.3,
    checkpoint_dir: str = "checkpoints",
    log_dir: str = "logs"
) -> Tuple[PolicyNetwork, ValueNetwork]:
    """آموزش عامل با قابلیت‌های خوشه‌بندی"""
    setup_logging(log_dir)
    logging.info(f"Starting training with clustering, env: {env_name}")
    
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim)
    value_net = ValueNetwork(state_dim, hidden_dim)
    
    agent = ClusteredGRPO(
        policy_network=policy_net,
        value_network=value_net,
        learning_rate=learning_rate,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clustering_weight=clustering_weight
    )
    
    buffer = ClusteredReplayBuffer()
    
    best_reward = float('-inf')
    episodes_without_improvement = 0
    
    for episode in range(num_episodes):
        state = env.reset()[0]
        episode_reward = 0
        episode_losses = []
        
        while True:
            action = agent.select_action(state, buffer)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            buffer.push(state, action, reward, next_state, done)
            
            if len(buffer.buffer) >= batch_size:
                states, actions, rewards, next_states, dones = buffer.sample(batch_size)
                with torch.no_grad():
                    old_probs = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                
                losses = agent.update(states, actions, old_probs, rewards, next_states, dones)
                episode_losses.append(losses)
            
            if done:
                break
            
            state = next_state
        
        if episode % 10 == 0:
            avg_losses = {k: np.mean([loss[k] for loss in episode_losses]) 
                         for k in episode_losses[0].keys()} if episode_losses else {}
            msg = (f"Episode {episode}, "
                  f"Reward: {episode_reward:.2f}, "
                  f"Losses: {avg_losses}")
            print(msg)
            logging.info(msg)
        
        if episode % checkpoint_freq == 0:
            save_checkpoint(policy_net, value_net, episode, episode_reward, checkpoint_dir)
        
        if episode_reward > best_reward:
            best_reward = episode_reward
            episodes_without_improvement = 0
            save_checkpoint(policy_net, value_net, episode, episode_reward,
                          os.path.join(checkpoint_dir, 'best_model'))
        else:
            episodes_without_improvement += 1
            
        if episodes_without_improvement >= patience:
            logging.info(f"Early stopping at episode {episode}")
            break
    
    env.close()
    return policy_net, value_net

if __name__ == "__main__":
    params = {
        'env_name': "CartPole-v1",
        'num_episodes': 1000,
        'batch_size': 64,
        'learning_rate': 0.003,
        'gamma': 0.995,
        'gae_lambda': 0.98,
        'hidden_dim': 128,
        'patience': 50,
        'checkpoint_freq': 100,
        'clustering_weight': 0.3
    }
    
    policy_net, value_net = train_clustered_agent(**params)
    
    torch.save(policy_net.state_dict(), "final_policy_model.pth")
    torch.save(value_net.state_dict(), "final_value_model.pth")
    print("Final models saved successfully!")
