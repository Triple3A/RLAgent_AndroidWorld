import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from android_world.agents import base_agent
from android_world.env import interface, json_action



class ReplayBuffer:
    def __init__(self, capacity: int, obs_shape: Tuple[int, ...]):
        self.capacity = capacity
        self.obs_shape = obs_shape
        self.ptr = 0
        self.size = 0

        self.states = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.next_states = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)

    def add(self, s, a, r, s2, done):
        self.states[self.ptr] = s
        self.actions[self.ptr] = a
        self.rewards[self.ptr] = r
        self.next_states[self.ptr] = s2
        self.dones[self.ptr] = float(done)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(
            states=self.states[idxs],
            actions=self.actions[idxs],
            rewards=self.rewards[idxs],
            next_states=self.next_states[idxs],
            dones=self.dones[idxs],
        )



class ConvFeatureNet(nn.Module):
    """
    Simple CNN feature extractor for screen pixels.
    Outputs a feature vector φ_θ(s) of size feature_dim.
    """

    def __init__(self, in_channels: int = 3, feature_dim: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
        )
        self._conv_out_size = None
        self.fc = None
        self.feature_dim = feature_dim

    def _build_head(self, x):
        if self._conv_out_size is None:
            with torch.no_grad():
                c_out = self.conv(x)
                self._conv_out_size = int(np.prod(c_out.shape[1:]))
            self.fc = nn.Linear(self._conv_out_size, self.feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W], uint8 in [0,255]
        x = x / 255.0
        x = self.conv(x)
        if self._conv_out_size is None or self.fc is None:
            self._build_head(x)
        x = x.view(x.size(0), -1)
        feat = torch.relu(self.fc(x))
        return feat 


class BDQNCore(nn.Module):
    """
    BDQN with:
      - shared ConvFeatureNet φ_θ(s)
      - Bayesian linear layer per action:
            Q(s,a) = w_a^T φ_θ(s)
        with a *diagonal* covariance approximation for simplicity.
    """

    def __init__(
        self,
        num_actions: int,
        in_channels: int = 3,
        feature_dim: int = 128,
        prior_var: float = 1.0,
        noise_var: float = 1.0,
        device: str = "cpu",
    ):
        super().__init__()
        self.device = torch.device(device)
        self.feature_net = ConvFeatureNet(in_channels=in_channels, feature_dim=feature_dim)

        self.num_actions = num_actions
        self.feature_dim = feature_dim

        
        self.register_buffer("weight_mean", torch.zeros(num_actions, feature_dim))
        self.register_buffer("weight_var", torch.ones(num_actions, feature_dim) * prior_var)

        self.prior_var = prior_var
        self.noise_var = noise_var

        self.to(self.device)

    @torch.no_grad()
    def phi(self, obs: torch.Tensor) -> torch.Tensor:
        return self.feature_net(obs)

    @torch.no_grad()
    def q_values(self, obs: torch.Tensor, use_sampled: bool = False) -> torch.Tensor:
        """
        obs: [B, C, H, W]
        Returns Q(s, a) for all actions: [B, num_actions]
        If use_sampled=True, sample weights from posterior (Thompson sampling).
        """
        feat = self.phi(obs)  
        if use_sampled:
            eps = torch.randn_like(self.weight_mean)
            w = self.weight_mean + torch.sqrt(torch.clamp(self.weight_var, min=1e-8)) * eps
        else:
            w = self.weight_mean
        q = torch.matmul(feat, w.t())
        return q

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.q_values(obs, use_sampled=False)

    @torch.no_grad()
    def thompson_q_values(self, obs: torch.Tensor) -> torch.Tensor:
        return self.q_values(obs, use_sampled=True)

    @torch.no_grad()
    def update_posterior(self, batch, gamma: float = 0.99):
        """
        Diagonal Bayesian linear regression update for the last layer.
        This corresponds to lines 10–12 of Algorithm 3 in the BDQN paper.
        """
        states = torch.from_numpy(batch["states"]).to(self.device) 
        next_states = torch.from_numpy(batch["next_states"]).to(self.device)
        actions = torch.from_numpy(batch["actions"]).long().to(self.device)
        rewards = torch.from_numpy(batch["rewards"]).to(self.device)
        dones = torch.from_numpy(batch["dones"]).to(self.device)

        phi_s = self.phi(states) 
        with torch.no_grad():
            q_next = self.q_values(next_states, use_sampled=False)
            max_q_next, _ = q_next.max(dim=1) 

        y = rewards + gamma * (1.0 - dones) * max_q_next

        for a in range(self.num_actions):
            mask = (actions == a)
            if mask.sum() == 0:
                continue
            phi_a = phi_s[mask] 
            y_a = y[mask]       

            # Precision update: Λ_a = 1/prior_var + (phi^T phi)/noise_var
            phi_sq_sum = (phi_a ** 2).sum(dim=0)  # [D]
            prior_prec = 1.0 / self.prior_var
            post_prec = prior_prec + phi_sq_sum / self.noise_var
            post_var = 1.0 / torch.clamp(post_prec, min=1e-8)

            # Mean update: μ_a = Σ_a (phi^T y)/noise_var
            phi_y_sum = (phi_a * y_a.unsqueeze(1)).sum(dim=0)  # [D]
            post_mean = post_var * (phi_y_sum / self.noise_var)

            self.weight_mean[a] = post_mean
            self.weight_var[a] = post_var


class AndroidDiscreteActionSpace:
    """
    Simple discretization of Android actions:
      - grid of CLICK positions
      - 4 SCROLL directions
      - BACK, HOME
    """

    def __init__(self, screen_size: Tuple[int, int], grid_rows: int = 4, grid_cols: int = 4):
        self.screen_w, self.screen_h = screen_size
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols

        self.num_clicks = grid_rows * grid_cols
        self.num_scroll = 4
        self.num_nav = 2
        self.n = self.num_clicks + self.num_scroll + self.num_nav

    def index_to_action(self, idx: int) -> json_action.JSONAction:
        if idx < self.num_clicks:
            row = idx // self.grid_cols
            col = idx % self.grid_cols
            x = int((col + 0.5) * self.screen_w / self.grid_cols)
            y = int((row + 0.5) * self.screen_h / self.grid_rows)
            return json_action.JSONAction(
                action_type=json_action.CLICK,
                x=x,
                y=y,
            )

        idx -= self.num_clicks
        if idx < self.num_scroll:
            direction = ["up", "down", "left", "right"][idx]
            return json_action.JSONAction(
                action_type=json_action.SCROLL,
                direction=direction,
            )

        idx -= self.num_scroll
        if idx == 0:
            return json_action.JSONAction(action_type=json_action.NAVIGATE_BACK)
        else:
            return json_action.JSONAction(action_type=json_action.NAVIGATE_HOME)


class BDQNAgent(base_agent.EnvironmentInteractingAgent):
    """
        - step(): acts in the environment, stores (x_t, a_t, r_t, x_{t+1})
        - _train_step(): samples a minibatch and performs the BDQN update.
    """

    def __init__(
        self,
        env: interface.AsyncEnv,
        name: str = "BDQNAgent",
        device: str = "cpu",
        obs_height: int = 84,
        obs_width: int = 84,
        feature_dim: int = 128,
        replay_capacity: int = 50000,
        batch_size: int = 32,
        gamma: float = 0.99,
        lr: float = 1e-3,
        warmup_steps: int = 1000,
        max_steps: int = 100000,
        train_freq: int = 4,
        target_update_freq: int = 1000,
        prior_var: float = 1.0,
        noise_var: float = 1.0,
        verbose: bool = True,
    ):
        super().__init__(env=env, name=name, transition_pause=1.0)

        self.device = torch.device(device)
        self.obs_height = obs_height
        self.obs_width = obs_width
        self.batch_size = batch_size
        self.gamma = gamma
        self.warmup_steps = warmup_steps
        self._max_steps = max_steps
        self.train_freq = train_freq
        self.target_update_freq = target_update_freq
        self._verbose = verbose

        self.visited_activities = set()

        # Discrete action space crafted from screen size
        self.action_space = AndroidDiscreteActionSpace(env.device_screen_size)
        self.num_actions = self.action_space.n

        # Networks
        self.bdqn = BDQNCore(
            num_actions=self.num_actions,
            in_channels=3,
            feature_dim=feature_dim,
            prior_var=prior_var,
            noise_var=noise_var,
            device=device,
        )
        self.target_bdqn = BDQNCore(
            num_actions=self.num_actions,
            in_channels=3,
            feature_dim=feature_dim,
            prior_var=prior_var,
            noise_var=noise_var,
            device=device,
        )
        self.target_bdqn.load_state_dict(self.bdqn.state_dict())

        self.optimizer = torch.optim.Adam(self.bdqn.parameters(), lr=lr)

        # Replay buffer; obs shape is (C,H,W)
        self.replay = ReplayBuffer(
            capacity=replay_capacity,
            obs_shape=(3, obs_height, obs_width),
        )


        self.total_steps = 0
        self._last_screen_signature: Optional[int] = None

    def _preprocess_obs(self, state: interface.State) -> np.ndarray:
        """
        Convert android_world State into a CHW float32 numpy array.
        """
        pixels = state.pixels  # H,W,3 uint8
        img = torch.from_numpy(pixels).float()  # H,W,C
        if img.dim() == 2:
            img = img.unsqueeze(-1)
        if img.shape[-1] == 4:
            img = img[..., :3]
        img = img.permute(2, 0, 1).unsqueeze(0)  # 1,C,H,W
        img = F.interpolate(
            img,
            size=(self.obs_height, self.obs_width),
            mode="bilinear",
            align_corners=False,
        )
        img = img.squeeze(0)  # C,H,W
        return img.numpy().astype(np.float32)

    def _screen_signature(self, state: interface.State) -> int:
        """
        Cheap hash of the current screen based on UI elements.
        Reward will be 1 when this signature changes.
        """
        # elems = getattr(state, "ui_elements", None)
        # if elems is None:
        #     return hash(state.pixels.tobytes())
        # sig_parts = []
        # for e in elems:
        #     text = getattr(e, "text", "")
        #     bounds = getattr(e, "bounds", None)
        #     sig_parts.append((text, tuple(bounds) if bounds is not None else None))
        # return hash(tuple(sig_parts))
        try:
            return repr(state.ui_elements)
        except Exception:
            return str(type(state))

    def _compute_reward(self, prev_state: Optional[interface.State], new_state: interface.State) -> float:
        """
        Reward = 1 if we transition to a new 'screen signature', else 0.
        You can customize this for your particular task.
        """
        new_sig = self._screen_signature(new_state)
        if prev_state is None or self._last_screen_signature is None:
            self._last_screen_signature = new_sig
            return 0.0
        reward = 1.0 if new_sig not in self.visited_activities else 0.0
        self._last_screen_signature = new_sig
        return reward

    def _select_action(self, obs_np: np.ndarray) -> int:
        """
        Thompson-sampling action selection from BDQN.
        """
        obs = torch.from_numpy(obs_np).unsqueeze(0).to(self.device)  # [1,C,H,W]
        with torch.no_grad():
            q = self.bdqn.thompson_q_values(obs)  # [1,A]
        q_np = q.cpu().numpy()[0]
        return int(np.argmax(q_np))

    def _train_step(self):
        if self.replay.size < self.warmup_steps:
            return

        batch = self.replay.sample(self.batch_size)

        states = torch.from_numpy(batch["states"]).to(self.device)
        next_states = torch.from_numpy(batch["next_states"]).to(self.device)
        actions = torch.from_numpy(batch["actions"]).long().to(self.device)
        rewards = torch.from_numpy(batch["rewards"]).to(self.device)
        dones = torch.from_numpy(batch["dones"]).to(self.device)

        q_all = self.bdqn(states)
        q = q_all.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_all = self.target_bdqn(next_states)
            max_next_q, _ = next_q_all.max(dim=1)
            target = rewards + self.gamma * (1.0 - dones) * max_next_q

        loss = F.mse_loss(q, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.bdqn.update_posterior(batch, gamma=self.gamma)


    def reset(self, go_home: bool = False) -> None:
        super().reset(go_home=go_home)
        self.total_steps = 0
        state = self.get_post_transition_state()
        self._last_screen_signature = self._screen_signature(state)
        self.visited_activities.clear()
        self.visited_activities.add(self._last_screen_signature)


    def step(self, goal: str) -> base_agent.AgentInteractionResult:
        """
        One interaction step with android_world.

        1. Get current state.
        2. Build observation and choose BDQN action (Thompson + ε-greedy).
        3. Execute action.
        4. Observe next state, compute reward, store transition.
        5. Periodically train and update target network.
        """
        # 1. Current state from env
        state = self.get_post_transition_state()

        # 2. Observation
        obs = self._preprocess_obs(state)

        # ε-greedy around Thompson for extra exploration
        eps = max(0.1, 1.0 - self.total_steps / float(self.warmup_steps * 5))
        if np.random.rand() < eps or self.replay.size < self.warmup_steps:
            action_idx = np.random.randint(self.num_actions)
        else:
            action_idx = self._select_action(obs)

        # 3. Execute action
        action = self.action_space.index_to_action(action_idx)
        self.env.execute_action(action)

        # 4. Next state & reward
        next_state = self.get_post_transition_state()
        next_obs = self._preprocess_obs(next_state)
        reward = self._compute_reward(state, next_state)
        
        activity = self._screen_signature(next_state)
        if activity != None:
            self.visited_activities.add(activity)

        done = self.total_steps >= self._max_steps

        # Store in replay
        self.replay.add(obs, action_idx, reward, next_obs, done)
        self.total_steps += 1

        # 5. Training and target updates
        if self.total_steps % self.train_freq == 0:
            self._train_step()

        if self.total_steps % self.target_update_freq == 0:
            self.target_bdqn.load_state_dict(self.bdqn.state_dict())


        if self._verbose:
            print(
                f"[{self.name}] step={self.total_steps} "
                f"reward={reward:.3f} "
                f"VisitedActivities={len(self.visited_activities)}"
            )


        step_data = {
            "raw_screenshot": state.pixels,
            "ui_elements": state.ui_elements,
            "reward": reward,
            "action_index": action_idx,
        }

        # if done:
        #     self.reset(go_home=True)

        return base_agent.AgentInteractionResult(
            done=done,
            data=step_data,
        )
