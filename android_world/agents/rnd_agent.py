import dataclasses
from typing import Any, Dict, Optional

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers

from android_world.agents import base_agent
from android_world.env import interface, json_action


@dataclasses.dataclass
class PpoRndConfig:
    rollout_len: int = 64          # K in RND paper
    gamma_ext: float = 0.99
    gamma_int: float = 0.99
    lam: float = 0.95
    ppo_clip: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.001
    rnd_coef: float = 1.0          # weight on RND loss
    int_coeff: float = 1.0         # scale intrinsic advantage
    ext_coeff: float = 1.0         # scale extrinsic advantage
    lr: float = 3e-4
    n_epochs: int = 4
    batch_size: int = 64
    max_episode_steps: int = 200
    n_actions: int = 13            # must match action mapping below
    obs_height: int = 160          # you can change depending on env
    obs_width: int = 90
    obs_channels: int = 3          # RGB


class PolicyValueNet(Model):
    """Policy + two value heads (ext, int)."""

    def __init__(self, cfg: PpoRndConfig):
        super().__init__()
        self.cfg = cfg
        self.conv1 = layers.Conv2D(32, 8, strides=4, activation="relu")
        self.conv2 = layers.Conv2D(64, 4, strides=2, activation="relu")
        self.conv3 = layers.Conv2D(64, 3, strides=1, activation="relu")
        self.gap = layers.GlobalAveragePooling2D()
        self.fc = layers.Dense(256, activation="relu")
        self.logits_layer = layers.Dense(cfg.n_actions, activation=None)
        self.v_ext_layer = layers.Dense(1, activation=None)
        self.v_int_layer = layers.Dense(1, activation=None)

    def call(self, obs, training=False):
        # obs: [B, H, W, C]
        x = tf.cast(obs, tf.float32) / 255.0
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.gap(x)          # -> [B, channels], no dependence on H/W
        x = self.fc(x)
        logits = self.logits_layer(x)
        v_ext = tf.squeeze(self.v_ext_layer(x), axis=-1)
        v_int = tf.squeeze(self.v_int_layer(x), axis=-1)
        return logits, v_ext, v_int


class RndNet(Model):
    """Feature extractor for RND (target / predictor)."""

    def __init__(self):
        super().__init__()
        self.conv1 = layers.Conv2D(32, 8, strides=4, activation="relu")
        self.conv2 = layers.Conv2D(64, 4, strides=2, activation="relu")
        self.gap = layers.GlobalAveragePooling2D()
        self.fc = layers.Dense(128, activation="relu")

    def call(self, obs, training=False):
        x = tf.cast(obs, tf.float32) / 255.0
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.gap(x)          # -> [B, channels]
        x = self.fc(x)
        return x                 # [B, 128]
    

class RNDAgent(base_agent.EnvironmentInteractingAgent):

    def __init__(
        self,
        env: interface.AsyncEnv,
        cfg: Optional[PpoRndConfig] = None,
        name: str = "RNDAgent",
        transition_pause: float | None = 1.0,
        verbose: bool = True,
    ):
        cfg = cfg or PpoRndConfig()
        super().__init__(env=env, name=name, transition_pause=transition_pause)
        self.cfg = cfg
        self._verbose = verbose

        # models
        self.policy_value = PolicyValueNet(cfg)
        self.rnd_target = RndNet()
        self.rnd_predictor = RndNet()

        # freeze target net
        self.rnd_target.trainable = False

        # optimizers
        self.optimizer = optimizers.Adam(learning_rate=cfg.lr)

        # episode and rollout state
        self._step_in_ep = 0
        self._last_screen_sig: Optional[str] = None

        K = cfg.rollout_len
        H, W, C = cfg.obs_height, cfg.obs_width, cfg.obs_channels

        # rollout buffers (single env)
        self.buf_obs = np.zeros((K, H, W, C), dtype=np.uint8)
        self.buf_actions = np.zeros(K, dtype=np.int32)
        self.buf_logp = np.zeros(K, dtype=np.float32)
        self.buf_rew_ext = np.zeros(K, dtype=np.float32)
        self.buf_rew_int = np.zeros(K, dtype=np.float32)
        self.buf_done = np.zeros(K, dtype=np.float32)
        self.buf_v_ext = np.zeros(K, dtype=np.float32)
        self.buf_v_int = np.zeros(K, dtype=np.float32)

        self._buf_idx = 0

        # build networks by calling once on dummy input
        dummy = tf.zeros((1, H, W, C), dtype=tf.float32)
        _ = self.policy_value(dummy)
        _ = self.rnd_target(dummy)
        _ = self.rnd_predictor(dummy)

    # ---------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------

    def _preprocess_obs(self, state: interface.State) -> np.ndarray:
        """Convert android_world State to (H, W, C) uint8 image with fixed size."""
        img = np.array(state.pixels)

        # Ensure we have 3 channels
        if img.ndim == 2:  # grayscale
            img = np.stack([img] * 3, axis=-1)
        elif img.shape[-1] == 4:  # RGBA -> RGB
            img = img[..., :3]

        # Resize to (obs_height, obs_width)
        img_tf = tf.image.resize(
            tf.convert_to_tensor(img),
            (self.cfg.obs_height, self.cfg.obs_width),
            method="bilinear",
            antialias=True,
        )
        img_np = tf.clip_by_value(img_tf, 0.0, 255.0).numpy().astype(np.uint8)
        return img_np


    def _screen_sig(self, state: interface.State) -> str:
        """Crude signature of the current screen to detect new activities."""
        try:
            return repr(state.ui_elements)
        except Exception:
            return str(type(state))

    def _extrinsic_reward(self, prev_sig: Optional[str], new_sig: str) -> float:
        # simple: reward=1 when screen layout changes (new activity/screen)
        if prev_sig is None:
            return 0.0
        return 1.0 if new_sig != prev_sig else 0.0

    def _index_to_action(self, idx: int) -> json_action.JSONAction:
        """
        Map discrete action index -> AndroidWorld JSONAction.
        Example: 3x3 click grid + 4 extra actions.
        """
        width, height = self.env.device_screen_size
        g = 3
        n_click = g * g
        extra = 4
        assert 0 <= idx < n_click + extra

        if idx < n_click:
            row = idx // g
            col = idx % g
            cell_w = width / g
            cell_h = height / g
            x = int((col + 0.5) * cell_w)
            y = int((row + 0.5) * cell_h)
            return json_action.JSONAction(
                action_type=json_action.CLICK,
                x=x,
                y=y,
            )

        offset = idx - n_click
        if offset == 0:
            return json_action.JSONAction(action_type=json_action.NAVIGATE_BACK)
        if offset == 1:
            return json_action.JSONAction(action_type=json_action.NAVIGATE_HOME)
        if offset == 2:
            return json_action.JSONAction(
                action_type=json_action.SCROLL, direction="up"
            )
        return json_action.JSONAction(
            action_type=json_action.SCROLL, direction="down"
        )

    # ---------------------------------------------------------
    # PPO / RND logic
    # ---------------------------------------------------------

    def _policy_step(self, obs_np: np.ndarray):
        """
        Run policy on a single observation.
        Returns: (action_idx, v_ext, v_int, logp_a)
        """
        obs = obs_np[None, ...]  # [1, H, W, C]
        logits, v_ext, v_int = self.policy_value(obs, training=False)
        logits = logits[0]      # [A]
        v_ext = float(v_ext[0])
        v_int = float(v_int[0])

        # sample from categorical
        # logits -> prob; sample index
        action_idx = tf.random.categorical(
            logits[None, :], num_samples=1
        )[0, 0].numpy().item()

        # log pi(a|s) = -CE
        logp_a = -tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=[action_idx], logits=logits[None, :]
        )[0].numpy().item()

        return action_idx, v_ext, v_int, logp_a

    def _compute_intrinsic_reward(self, next_obs_np: np.ndarray) -> float:
        obs = next_obs_np[None, ...]
        target_feat = self.rnd_target(obs, training=False)
        pred_feat = self.rnd_predictor(obs, training=False)
        err = tf.reduce_sum(tf.square(pred_feat - target_feat), axis=-1)
        return float(err[0].numpy())

    def _update_if_ready(self, last_v_ext: float, last_v_int: float):
        """
        When rollout_len transitions are collected, perform PPO+RND update.
        """
        cfg = self.cfg
        K = cfg.rollout_len
        if self._buf_idx < K:
            return

        # ----- 1. compute GAE returns and advantages (numpy) -----
        v_ext = np.append(self.buf_v_ext, last_v_ext)
        v_int = np.append(self.buf_v_int, last_v_int)
        rext = self.buf_rew_ext.copy()
        rint = self.buf_rew_int.copy()
        done = self.buf_done.copy()

        adv_ext = np.zeros(K, dtype=np.float32)
        adv_int = np.zeros(K, dtype=np.float32)
        gae_ext = 0.0
        gae_int = 0.0
        for t in reversed(range(K)):
            nonterm = 1.0 - done[t]
            delta_ext = rext[t] + cfg.gamma_ext * v_ext[t + 1] * nonterm - v_ext[t]
            delta_int = rint[t] + cfg.gamma_int * v_int[t + 1] * nonterm - v_int[t]
            gae_ext = delta_ext + cfg.gamma_ext * cfg.lam * nonterm * gae_ext
            gae_int = delta_int + cfg.gamma_int * cfg.lam * nonterm * gae_int
            adv_ext[t] = gae_ext
            adv_int[t] = gae_int

        ret_ext = adv_ext + self.buf_v_ext
        ret_int = adv_int + self.buf_v_int

        adv = cfg.ext_coeff * adv_ext + cfg.int_coeff * adv_int
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # ----- 2. PPO + RND optimization (tf.GradientTape) -----
        n_samples = K
        idxs = np.arange(n_samples)

        obs_buf = self.buf_obs.copy()
        act_buf = self.buf_actions.copy()
        old_logp_buf = self.buf_logp.copy()

        for _ in range(cfg.n_epochs):
            np.random.shuffle(idxs)
            for start in range(0, n_samples, cfg.batch_size):
                end = start + cfg.batch_size
                mb_idx = idxs[start:end]

                mb_obs = obs_buf[mb_idx]
                mb_act = act_buf[mb_idx]
                mb_adv = adv[mb_idx]
                mb_ret_ext = ret_ext[mb_idx]
                mb_ret_int = ret_int[mb_idx]
                mb_old_logp = old_logp_buf[mb_idx]

                with tf.GradientTape() as tape:
                    logits, v_ext_pred, v_int_pred = self.policy_value(
                        mb_obs, training=True
                    )

                    # PPO log prob
                    logp_new = -tf.nn.sparse_softmax_cross_entropy_with_logits(
                        labels=mb_act, logits=logits
                    )
                    ratio = tf.exp(logp_new - mb_old_logp)

                    pg_unclipped = ratio * mb_adv
                    pg_clipped = tf.clip_by_value(
                        ratio, 1.0 - cfg.ppo_clip, 1.0 + cfg.ppo_clip
                    ) * mb_adv
                    pg_loss = -tf.reduce_mean(tf.minimum(pg_unclipped, pg_clipped))

                    # value losses
                    v_loss_ext = tf.reduce_mean(
                        tf.square(v_ext_pred - mb_ret_ext)
                    )
                    v_loss_int = tf.reduce_mean(
                        tf.square(v_int_pred - mb_ret_int)
                    )
                    v_loss = 0.5 * (v_loss_ext + v_loss_int)

                    # entropy bonus
                    probs = tf.nn.softmax(logits, axis=-1)
                    log_probs = tf.nn.log_softmax(logits, axis=-1)
                    ent = -tf.reduce_mean(tf.reduce_sum(probs * log_probs, axis=-1))

                    # RND loss (fit predictor to target on next states)
                    # Here we reuse mb_obs as inputs to RND;
                    # if you prefer RND on next_obs, store them separately.
                    t_feat = self.rnd_target(mb_obs, training=False)
                    p_feat = self.rnd_predictor(mb_obs, training=True)
                    rnd_loss = tf.reduce_mean(tf.reduce_sum(
                        tf.square(p_feat - t_feat), axis=-1
                    ))

                    total_loss = (
                        pg_loss
                        + cfg.vf_coef * v_loss
                        - cfg.ent_coef * ent
                        + cfg.rnd_coef * rnd_loss
                    )

                vars_train = (
                    self.policy_value.trainable_variables
                    + self.rnd_predictor.trainable_variables
                )
                grads = tape.gradient(total_loss, vars_train)
                self.optimizer.apply_gradients(zip(grads, vars_train))

        # reset buffer
        self._buf_idx = 0

    # ---------------------------------------------------------
    # EnvironmentInteractingAgent overrides
    # ---------------------------------------------------------

    def reset(self, go_home: bool = False) -> None:
        super().reset(go_home=go_home)
        self._step_in_ep = 0
        self._buf_idx = 0
        state = self.get_post_transition_state()
        self._last_screen_sig = self._screen_sig(state)

    def step(self, goal: str) -> base_agent.AgentInteractionResult:
        """
        One RL step: observe, act via policy, store transition, maybe update.
        """
        cfg = self.cfg

        # current state
        state = self.get_post_transition_state()
        obs = self._preprocess_obs(state)

        # policy
        a_idx, v_ext, v_int, logp_a = self._policy_step(obs)
        action = self._index_to_action(a_idx)

        # apply action
        self.env.execute_action(action)

        # next state and extrinsic reward
        next_state = self.get_post_transition_state()
        next_obs = self._preprocess_obs(next_state)
        new_sig = self._screen_sig(next_state)
        r_ext = self._extrinsic_reward(self._last_screen_sig, new_sig)
        self._last_screen_sig = new_sig

        # intrinsic reward from RND on next state
        r_int = self._compute_intrinsic_reward(next_obs)

        # store transition
        idx = self._buf_idx
        self.buf_obs[idx] = obs
        self.buf_actions[idx] = a_idx
        self.buf_logp[idx] = logp_a
        self.buf_rew_ext[idx] = r_ext
        self.buf_rew_int[idx] = r_int
        self.buf_done[idx] = 0.0   # only horizon-based termination here
        self.buf_v_ext[idx] = v_ext
        self.buf_v_int[idx] = v_int
        self._buf_idx += 1

        # maybe update PPO+RND
        self._update_if_ready(last_v_ext=v_ext, last_v_int=v_int)

        # episode termination rule (simple horizon)
        self._step_in_ep += 1
        done = self._step_in_ep >= cfg.max_episode_steps
        if done:
            self.reset(go_home=True)

        if self._verbose:
            print(
                f"[{self.name}] step={self._step_in_ep} "
                f"a={a_idx} R_ext={r_ext:.3f} R_int={r_int:.3f}"
            )

        data: Dict[str, Any] = {
            "raw_screenshot": state.pixels,
            "ui_elements": state.ui_elements,
            "action_idx": a_idx,
            "android_action": action,
            "reward_ext": r_ext,
            "reward_int": r_int,
            "goal": goal,
        }
        return base_agent.AgentInteractionResult(done=done, data=data)