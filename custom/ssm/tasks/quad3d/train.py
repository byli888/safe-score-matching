"""Train SSM on the Quad3D stabilization task."""
import os
import time
import pickle
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
import jax
import jax.numpy as jnp
import numpy as np
from absl import app, flags
from tqdm import tqdm
from ssm.common.logging import compact_episode_log, compact_update_log
from ssm.tasks.quad3d.env import GRAV, OBS_FEATURE_MODES, Quad3DEnv
FLAGS = flags.FLAGS
flags.DEFINE_integer('checkpoint_interval', 20000, 'Steps between model saves.')
flags.DEFINE_string('project_name', 'safefm-hj', 'wandb project name.')
flags.DEFINE_string('entity', '', 'wandb entity.')
flags.DEFINE_string('run_name', '', 'wandb run name.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_integer('max_steps', 2000000, 'Total environment steps.')
flags.DEFINE_integer('start_training', 5000, 'Random warmup steps.')
flags.DEFINE_integer('batch_size', 256, 'Batch size.')
flags.DEFINE_integer('log_interval', 1000, 'Steps between logging.')
flags.DEFINE_boolean('wandb', False, 'Enable wandb logging.')
flags.DEFINE_boolean('tqdm_bar', True, 'Show tqdm progress bar.')
POLICY_NAME = 'ssm'
flags.DEFINE_float('M_q', 10.0, 'QSM scaling factor.')
flags.DEFINE_float('alpha_r', 1.0, 'Reward gradient weight.')
flags.DEFINE_float('beta_safety', 5.0, 'Safety gradient weight.')
flags.DEFINE_float('delta', 0.0, 'Safe set threshold.')
flags.DEFINE_float('discount', 0.99, 'Reward discount.')
flags.DEFINE_float('discount_h', 0.999, 'Safety discount.')
flags.DEFINE_float('tau', 0.005, 'EMA rate.')
flags.DEFINE_float('cost_critic_hyperparam', 0.9, 'Reversed expectile for V_h.')
flags.DEFINE_enum('qh_stage_mode', 'stage_b', ['stage_a', 'stage_b'], 'Sampled-Q_h mode. stage_a=Q_h(s,pi(s)); stage_b=min_k Q_h(s,a_k).')
flags.DEFINE_enum('qh_candidate_source', 'policy_plus_gaussian', ['policy_only', 'policy_plus_gaussian'], 'Candidate source for sampled-Q_h stage_b.')
flags.DEFINE_integer('qh_min_k_samples', 8, 'Candidate count K for sampled-Q_h gate/bootstrap.')
flags.DEFINE_float('qh_min_k_sigma', 0.3, 'Gaussian proposal std for sampled-Q_h stage_b.')
flags.DEFINE_float('action_penalty', 0.001, 'Action penalty (R diagonal).')
flags.DEFINE_float('q_pos', 10.0, 'Position weight (px, py, pz) in Q matrix.')
flags.DEFINE_float('q_vel', 1.0, 'Velocity weight (vx, vy, vz) in Q matrix.')
flags.DEFINE_float('q_pos_xy', -1.0, 'Axis-split position weight for px/py. <0 disables axis-split.')
flags.DEFINE_float('q_pos_z', -1.0, 'Axis-split position weight for pz. <0 disables axis-split.')
flags.DEFINE_float('q_vel_xy', -1.0, 'Axis-split velocity weight for vx/vy. <0 disables axis-split.')
flags.DEFINE_float('q_vel_z', -1.0, 'Axis-split velocity weight for vz. <0 disables axis-split.')
flags.DEFINE_float('q_ang', 0.2, 'Angle weight (phi, theta, psi) in Q matrix.')
flags.DEFINE_float('q_psi', -1.0, 'Yaw weight. -1.0 means: fall back to q_ang.')
flags.DEFINE_string('reward_norm', 'l2', "Reward norm: 'l2' (quadratic) or 'l1' (weighted abs).")
flags.DEFINE_float('target_pz', 0.0, 'z-coord of reference state x_ref (other 8 dims = 0). Default 0 (apple-to-apple with Dawson). Failover: -0.5 (off the safety boundary).')
flags.DEFINE_boolean('legacy_reward', False, 'Use original sparse Dawson goal-set reward instead of dense quadratic regulation. Default False (new reward).')
flags.DEFINE_float('success_radius', 0.5, 'Radius for success_frac metric. An episode is a success if ||x_T - x_ref|| < this AND not terminated.')
flags.DEFINE_string('beta_schedule', 'vp', 'DDPM beta schedule.')
flags.DEFINE_integer('T', 5, 'DDPM diffusion steps.')
flags.DEFINE_integer('actor_delay', 1, 'Actor update every N critic updates.')
flags.DEFINE_string('actor_hidden', '256,256,256', 'Actor hidden dims.')
flags.DEFINE_string('critic_hidden', '256,256', 'Critic hidden dims.')
flags.DEFINE_boolean('pure_qsm', False, 'Pure QSM ablation (no safety routing).')
flags.DEFINE_float('ddpm_temperature', 0.2, 'DDPM sampling temperature.')
flags.DEFINE_float('fuse_clip_min', -10000.0, 'Bellman target clip lower bound.')
flags.DEFINE_float('fuse_clip_max', 100.0, 'Bellman target clip upper bound.')
flags.DEFINE_float('huber_delta', 10.0, 'Huber delta for reward critic. 0=MSE.')
flags.DEFINE_enum('obs_feature_mode', 'state', list(OBS_FEATURE_MODES), 'Observation features: state=legacy 9D, h, h_components, h_components_closing, or h_components_closing_reach.')
flags.DEFINE_float('obs_h_scale', 1.0, 'Scale for tanh-normalized Quad3D h observation features.')
flags.DEFINE_float('obs_hdot_scale', 1.0, 'Scale for tanh-normalized Quad3D hdot observation feature.')
flags.DEFINE_float('obs_reach_scale', 1.0, 'Global multiplier for Quad3D reach observation feature scales.')
flags.DEFINE_float('obs_goal_pos_radius', 0.3, 'Obs-only position radius used in Quad3D reach margin features.')
flags.DEFINE_float('obs_goal_vel_radius', 0.3, 'Obs-only velocity radius used in Quad3D reach margin features.')
flags.DEFINE_float('obs_goal_angle_radius', 0.2, 'Obs-only attitude radius used in Quad3D reach margin features.')
flags.DEFINE_string('sdf_mode', 'raw', "SDF shaping mode: 'raw' / 'hard' / 'hard_slope' / 'hard_scale'. 'hard' replaces unsafe-side h with a constant. 'hard_slope' adds a piecewise-linear slope in the unsafe interior. 'hard_scale' rescales the whole signal on top of 'hard'.")
flags.DEFINE_float('sdf_unsafe_const', 0.0, 'Unsafe-side constant (h>0 region). Required >0 for {hard, hard_scale}; >=0 for hard_slope (with unsafe_const+unsafe_slope>0).')
flags.DEFINE_float('sdf_unsafe_slope', 0.0, 'Unsafe-side slope. Required for hard_slope.')
flags.DEFINE_float('sdf_global_scale', 0.0, 'Global multiplicative scale. Required >0 for hard_scale.')
flags.DEFINE_boolean('boundary_replay', False, 'Enable boundary replay oversampling for Quad3D.')
flags.DEFINE_enum('boundary_mode', 'ground_near_unsafe', ['abs', 'near_unsafe', 'ground_abs', 'ground_near_unsafe'], "Boundary replay criterion on raw Quad3D geometry. 'abs'/'near_unsafe' use h_raw=max(pz, ||x||-3) like Quad2D. 'ground_*' only oversample states whose active boundary is the ground branch (pz dominates radius margin).")
flags.DEFINE_float('boundary_eps', 0.05, 'Boundary replay threshold eps. For ground_* modes this is applied to the raw pz margin to pz=0.')
flags.DEFINE_float('boundary_frac', 0.25, 'Target fraction of each batch drawn from the boundary set.')
flags.DEFINE_integer('boundary_min_count', 32, 'Minimum distinct boundary samples before oversampling activates.')
flags.DEFINE_boolean('init_curriculum', False, 'Enable training-time init curriculum for Quad3D resets.')
flags.DEFINE_enum('init_curriculum_mode', 'outer_xy', ['outer_xy'], "Training-time init curriculum mode. 'outer_xy' oversamples safe initial states with large xy radius.")
flags.DEFINE_float('init_curriculum_frac', 0.3, 'Probability of using the curriculum-biased reset branch on training env resets.')
flags.DEFINE_float('init_curriculum_pos_xy_min', 1.4, 'For outer_xy curriculum, require sqrt(px^2 + py^2) >= this value on curriculum-biased training resets.')
flags.DEFINE_enum('init_curriculum_schedule', 'constant', ['constant', 'linear_anneal'], 'Schedule for training-time init curriculum probability.')
flags.DEFINE_integer('init_curriculum_anneal_start', 0, 'Step where linear_anneal starts reducing init_curriculum_frac.')
flags.DEFINE_integer('init_curriculum_anneal_end', 0, 'Step where linear_anneal reaches init_curriculum_anneal_end_frac.')
flags.DEFINE_float('init_curriculum_anneal_end_frac', 0.0, 'Final curriculum probability after linear_anneal ends.')

def effective_q_psi():
    return float(FLAGS.q_psi) if FLAGS.q_psi >= 0.0 else float(FLAGS.q_ang)

def get_axis_split_kwargs_from_flags():
    split_kwargs = {}
    for key in ('q_pos_xy', 'q_pos_z', 'q_vel_xy', 'q_vel_z'):
        value = float(getattr(FLAGS, key))
        if value >= 0.0:
            split_kwargs[key] = value
    if split_kwargs and len(split_kwargs) != 4:
        raise ValueError('Axis-split Q requires all four flags together: --q_pos_xy --q_pos_z --q_vel_xy --q_vel_z')
    return split_kwargs


def build_run_config():
    split_kwargs = get_axis_split_kwargs_from_flags()
    config = {'seed': FLAGS.seed, 'env': 'Quad3D', 'policy': POLICY_NAME, 'M_q': FLAGS.M_q, 'alpha_r': FLAGS.alpha_r, 'beta_safety': FLAGS.beta_safety, 'delta': FLAGS.delta, 'discount': FLAGS.discount, 'discount_h': FLAGS.discount_h, 'tau': FLAGS.tau, 'qh_stage_mode': FLAGS.qh_stage_mode, 'qh_candidate_source': FLAGS.qh_candidate_source, 'qh_min_k_samples': FLAGS.qh_min_k_samples, 'qh_min_k_sigma': FLAGS.qh_min_k_sigma, 'T': FLAGS.T, 'beta_schedule': FLAGS.beta_schedule, 'cost_critic_hyperparam': FLAGS.cost_critic_hyperparam, 'safety_estimator': 'sampled_qh', 'batch_size': FLAGS.batch_size, 'actor_delay': FLAGS.actor_delay, 'max_steps': FLAGS.max_steps, 'start_training': FLAGS.start_training, 'actor_hidden': FLAGS.actor_hidden, 'critic_hidden': FLAGS.critic_hidden, 'pure_qsm': FLAGS.pure_qsm, 'ddpm_temperature': FLAGS.ddpm_temperature, 'fuse_clip_min': FLAGS.fuse_clip_min, 'fuse_clip_max': FLAGS.fuse_clip_max, 'huber_delta': FLAGS.huber_delta, 'obs_feature_mode': FLAGS.obs_feature_mode, 'obs_h_scale': FLAGS.obs_h_scale, 'obs_hdot_scale': FLAGS.obs_hdot_scale, 'obs_reach_scale': FLAGS.obs_reach_scale, 'obs_goal_pos_radius': FLAGS.obs_goal_pos_radius, 'obs_goal_vel_radius': FLAGS.obs_goal_vel_radius, 'obs_goal_angle_radius': FLAGS.obs_goal_angle_radius, 'action_penalty': FLAGS.action_penalty, 'q_pos': FLAGS.q_pos, 'q_vel': FLAGS.q_vel, 'q_ang': FLAGS.q_ang, 'q_psi': FLAGS.q_psi, 'q_psi_effective': effective_q_psi(), 'axis_split_q': bool(split_kwargs), 'q_pos_xy': split_kwargs.get('q_pos_xy'), 'q_pos_z': split_kwargs.get('q_pos_z'), 'q_vel_xy': split_kwargs.get('q_vel_xy'), 'q_vel_z': split_kwargs.get('q_vel_z'), 'reward_norm': FLAGS.reward_norm, 'target_pz': FLAGS.target_pz, 'legacy_reward': FLAGS.legacy_reward, 'success_radius': FLAGS.success_radius, 'reward_type': 'legacy_dawson_goal_set' if FLAGS.legacy_reward else 'dense_quadratic_regulation', 'dt': 0.01, 'max_episode_steps': 500, 'sdf_mode': FLAGS.sdf_mode, 'sdf_unsafe_const': FLAGS.sdf_unsafe_const, 'sdf_unsafe_slope': FLAGS.sdf_unsafe_slope, 'sdf_global_scale': FLAGS.sdf_global_scale, 'boundary_replay': FLAGS.boundary_replay, 'boundary_mode': FLAGS.boundary_mode, 'boundary_eps': FLAGS.boundary_eps, 'boundary_frac': FLAGS.boundary_frac, 'boundary_min_count': FLAGS.boundary_min_count, 'init_curriculum': FLAGS.init_curriculum, 'init_curriculum_mode': FLAGS.init_curriculum_mode, 'init_curriculum_frac': FLAGS.init_curriculum_frac, 'init_curriculum_pos_xy_min': FLAGS.init_curriculum_pos_xy_min, 'init_curriculum_schedule': FLAGS.init_curriculum_schedule, 'init_curriculum_anneal_start': FLAGS.init_curriculum_anneal_start, 'init_curriculum_anneal_end': FLAGS.init_curriculum_anneal_end, 'init_curriculum_anneal_end_frac': FLAGS.init_curriculum_anneal_end_frac}
    return config

def make_env(seed):
    """Construct the training environment."""
    kwargs = dict(seed=seed, action_penalty=FLAGS.action_penalty, q_pos=FLAGS.q_pos, q_vel=FLAGS.q_vel, q_ang=FLAGS.q_ang, reward_norm=FLAGS.reward_norm, target_pz=FLAGS.target_pz, legacy_reward=FLAGS.legacy_reward, sdf_mode=FLAGS.sdf_mode, sdf_unsafe_const=FLAGS.sdf_unsafe_const, sdf_unsafe_slope=FLAGS.sdf_unsafe_slope, sdf_global_scale=FLAGS.sdf_global_scale, init_curriculum=FLAGS.init_curriculum, init_curriculum_mode=FLAGS.init_curriculum_mode, init_curriculum_frac=FLAGS.init_curriculum_frac, init_curriculum_pos_xy_min=FLAGS.init_curriculum_pos_xy_min, init_curriculum_schedule=FLAGS.init_curriculum_schedule, init_curriculum_anneal_start=FLAGS.init_curriculum_anneal_start, init_curriculum_anneal_end=FLAGS.init_curriculum_anneal_end, init_curriculum_anneal_end_frac=FLAGS.init_curriculum_anneal_end_frac, obs_feature_mode=FLAGS.obs_feature_mode, obs_h_scale=FLAGS.obs_h_scale, obs_hdot_scale=FLAGS.obs_hdot_scale, obs_reach_scale=FLAGS.obs_reach_scale, obs_goal_pos_radius=FLAGS.obs_goal_pos_radius, obs_goal_vel_radius=FLAGS.obs_goal_vel_radius, obs_goal_angle_radius=FLAGS.obs_goal_angle_radius)
    if FLAGS.q_psi >= 0.0:
        kwargs['q_psi'] = FLAGS.q_psi
    kwargs.update(get_axis_split_kwargs_from_flags())
    return Quad3DEnv(**kwargs)

def save_checkpoint(agent, env, path, policy_name):
    """Save agent params + obs normalization stats."""
    params = jax.device_get({'score_model': agent.score_model.params, 'critic_1': agent.critic_1.params, 'critic_2': agent.critic_2.params, 'target_critic_1': agent.target_critic_1.params, 'target_critic_2': agent.target_critic_2.params, 'safe_critic': agent.safe_critic.params, 'safe_target_critic': agent.safe_target_critic.params, 'safe_value': agent.safe_value.params, 'safe_target_value': agent.safe_target_value.params})
    if hasattr(agent, 'beta_net'):
        params['beta_net'] = jax.device_get(agent.beta_net.params)
    data = {'params': params, 'policy': policy_name, 'env': 'Quad3D', 'rng': jax.device_get(agent.rng), 'obs_mean': env._obs_mean.copy(), 'obs_var': env._obs_var.copy(), 'obs_count': env._obs_count, 'config': build_run_config()}
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'  Checkpoint saved: {path} ({os.path.getsize(path) / 1000000.0:.1f} MB)')

class SimpleReplayBuffer:
    """Simple numpy replay buffer with optional Quad3D boundary oversampling."""

    def __init__(self, obs_dim, act_dim, max_size=1000000, boundary_replay=False, boundary_mode='ground_near_unsafe', boundary_eps=0.05, boundary_frac=0.25, boundary_min_count=32):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.boundary_replay = boundary_replay
        self.boundary_mode = boundary_mode
        self.boundary_eps = boundary_eps
        self.boundary_frac = boundary_frac
        self.boundary_min_count = boundary_min_count
        self.observations = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.actions = np.zeros((max_size, act_dim), dtype=np.float32)
        self.rewards = np.zeros(max_size, dtype=np.float32)
        self.costs = np.zeros(max_size, dtype=np.float32)
        self.masks = np.zeros(max_size, dtype=np.float32)
        self.next_observations = np.zeros((max_size, obs_dim), dtype=np.float32)
        self._pz_margins = np.zeros(max_size, dtype=np.float32)
        self._radius_margins = np.zeros(max_size, dtype=np.float32)
        self._boundary_slots = np.empty(max_size, dtype=np.int32)
        self._boundary_pos = np.full(max_size, -1, dtype=np.int32)
        self._boundary_size = 0
        self._unsafe_slots = np.empty(max_size, dtype=np.int32)
        self._unsafe_pos = np.full(max_size, -1, dtype=np.int32)
        self._unsafe_size = 0
        self._rng = np.random.default_rng(0)
        self.last_sample_info = {'boundary_active': 0.0, 'boundary_draw_frac': 0.0, 'boundary_frac_in_buffer': 0.0, 'unsafe_frac_in_buffer': 0.0, 'boundary_frac_in_batch': 0.0, 'unsafe_frac_in_batch': 0.0, 'costs_mean_in_batch': 0.0}

    def seed(self, s):
        self._rng = np.random.default_rng(s)

    def _boundary_mask(self, pz_margins, radius_margins):
        pz_margins = np.asarray(pz_margins)
        radius_margins = np.asarray(radius_margins)
        h_raw = np.maximum(pz_margins, radius_margins)
        if self.boundary_mode == 'abs':
            return np.abs(h_raw) <= self.boundary_eps
        if self.boundary_mode == 'near_unsafe':
            return h_raw >= -self.boundary_eps
        ground_dominant = pz_margins >= radius_margins
        if self.boundary_mode == 'ground_abs':
            return ground_dominant & (np.abs(pz_margins) <= self.boundary_eps)
        if self.boundary_mode == 'ground_near_unsafe':
            return ground_dominant & (pz_margins >= -self.boundary_eps)
        raise ValueError(f'Unknown boundary_mode: {self.boundary_mode}')

    def _update_membership(self, idx, is_member, slots, positions, count):
        pos = positions[idx]
        if is_member:
            if pos < 0:
                slots[count] = idx
                positions[idx] = count
                count += 1
            return count
        if pos < 0:
            return count
        last_idx = slots[count - 1]
        slots[pos] = last_idx
        positions[last_idx] = pos
        positions[idx] = -1
        return count - 1

    def insert(self, data):
        i = self.ptr
        self.observations[i] = data['observations']
        self.actions[i] = data['actions']
        self.rewards[i] = data['rewards']
        self.costs[i] = data['costs']
        self.masks[i] = data['masks']
        self.next_observations[i] = data['next_observations']
        self._pz_margins[i] = data['pz_margin']
        self._radius_margins[i] = data['radius_margin']
        cost = float(self.costs[i])
        self._boundary_size = self._update_membership(i, bool(self._boundary_mask(self._pz_margins[i], self._radius_margins[i])), self._boundary_slots, self._boundary_pos, self._boundary_size)
        self._unsafe_size = self._update_membership(i, bool(cost > 0.0), self._unsafe_slots, self._unsafe_pos, self._unsafe_size)
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        target_boundary = int(round(batch_size * self.boundary_frac))
        boundary_active = self.boundary_replay and self.boundary_eps > 0.0 and (target_boundary > 0) and (self._boundary_size >= self.boundary_min_count)
        if boundary_active:
            num_boundary = min(target_boundary, self._boundary_size, batch_size)
            boundary_pos = self._rng.choice(self._boundary_size, size=num_boundary, replace=False)
            boundary_idxs = self._boundary_slots[boundary_pos]
            uniform_idxs = self._rng.integers(0, self.size, size=batch_size - num_boundary)
            idxs = np.concatenate([boundary_idxs, uniform_idxs], axis=0)
            self._rng.shuffle(idxs)
        else:
            num_boundary = 0
            idxs = self._rng.integers(0, self.size, size=batch_size)
        batch_costs = self.costs[idxs]
        batch_boundary = self._boundary_mask(self._pz_margins[idxs], self._radius_margins[idxs])
        self.last_sample_info = {'boundary_active': float(boundary_active), 'boundary_draw_frac': num_boundary / batch_size, 'boundary_frac_in_buffer': self._boundary_size / max(self.size, 1), 'unsafe_frac_in_buffer': self._unsafe_size / max(self.size, 1), 'boundary_frac_in_batch': float(np.mean(batch_boundary)), 'unsafe_frac_in_batch': float(np.mean(batch_costs > 0.0)), 'costs_mean_in_batch': float(np.mean(batch_costs))}
        return {'observations': jnp.array(self.observations[idxs]), 'actions': jnp.array(self.actions[idxs]), 'rewards': jnp.array(self.rewards[idxs]), 'costs': jnp.array(self.costs[idxs]), 'masks': jnp.array(self.masks[idxs]), 'next_observations': jnp.array(self.next_observations[idxs])}

def main(_):
    split_kwargs = get_axis_split_kwargs_from_flags()
    effective_psi = effective_q_psi()
    run_config = build_run_config()
    from ssm.agents.ssm_agent import SSMOnlineAgent
    if not FLAGS.run_name:
        parts = ['quad3d_ssm', f'{FLAGS.qh_stage_mode}_k{FLAGS.qh_min_k_samples}s{FLAGS.qh_min_k_sigma:g}']
        if FLAGS.legacy_reward:
            parts.append('legacy')
        else:
            if split_kwargs:
                parts.append(f"qpvsplit{split_kwargs['q_pos_xy']:g}_{split_kwargs['q_pos_z']:g}_{split_kwargs['q_vel_xy']:g}_{split_kwargs['q_vel_z']:g}")
                parts.append(f'qpsi{effective_psi:g}')
                parts.append(f'tpz{FLAGS.target_pz:g}')
            else:
                parts.append(f'qpva{FLAGS.q_pos:g}_{FLAGS.q_vel:g}_{FLAGS.q_ang:g}')
                if FLAGS.target_pz != 0.0:
                    parts.append(f'tpz{FLAGS.target_pz:g}')
            if FLAGS.reward_norm != 'l2':
                parts.append(FLAGS.reward_norm.upper())
            if not split_kwargs and FLAGS.q_psi >= 0.0 and (FLAGS.q_psi != FLAGS.q_ang):
                parts.append(f'qpsi{FLAGS.q_psi:g}')
        parts.append(f'ap{FLAGS.action_penalty:g}')
        if FLAGS.boundary_replay:
            parts.append(f'br{FLAGS.boundary_mode}_e{FLAGS.boundary_eps:g}_f{FLAGS.boundary_frac:g}')
        if FLAGS.obs_feature_mode != 'state':
            parts.append(f'obs{FLAGS.obs_feature_mode}')
            if FLAGS.obs_h_scale != 1.0:
                parts.append(f'hscale{FLAGS.obs_h_scale:g}')
            if FLAGS.obs_hdot_scale != 1.0:
                parts.append(f'hdscale{FLAGS.obs_hdot_scale:g}')
            if FLAGS.obs_reach_scale != 1.0:
                parts.append(f'rscale{FLAGS.obs_reach_scale:g}')
            if FLAGS.obs_goal_pos_radius != 0.3:
                parts.append(f'gpr{FLAGS.obs_goal_pos_radius:g}')
            if FLAGS.obs_goal_vel_radius != 0.3:
                parts.append(f'gvr{FLAGS.obs_goal_vel_radius:g}')
            if FLAGS.obs_goal_angle_radius != 0.2:
                parts.append(f'gar{FLAGS.obs_goal_angle_radius:g}')
        if FLAGS.init_curriculum:
            cur_tag = f'cur{FLAGS.init_curriculum_mode}_f{FLAGS.init_curriculum_frac:g}_r{FLAGS.init_curriculum_pos_xy_min:g}'
            if FLAGS.init_curriculum_schedule != 'constant':
                cur_tag += f'_{FLAGS.init_curriculum_schedule}_a{FLAGS.init_curriculum_anneal_start}to{FLAGS.init_curriculum_anneal_end}_fend{FLAGS.init_curriculum_anneal_end_frac:g}'
            parts.append(cur_tag)
        if FLAGS.fuse_clip_min != -10000.0:
            parts.append(f'fclip{FLAGS.fuse_clip_min:g}')
        if FLAGS.huber_delta != 10.0:
            parts.append(f'huber{FLAGS.huber_delta:g}')
        parts.append(f'mq{FLAGS.M_q:g}')
        parts.append(f's{FLAGS.seed}')
        run_name = '_'.join(parts)
    else:
        run_name = FLAGS.run_name
    if FLAGS.wandb:
        import wandb
        wandb.init(project=FLAGS.project_name, entity=FLAGS.entity, name=run_name, config=run_config)
    env = make_env(seed=FLAGS.seed)
    env.action_space.seed(FLAGS.seed)
    print(f'Env: Quad3D, obs={env.observation_space.shape}, act={env.action_space.shape}')
    print(f'  dt={env.dt}, max_steps={env.max_episode_steps}, m={env.m}, mg={env.m * GRAV:.4f}')
    print(f"  reward_type={('legacy' if FLAGS.legacy_reward else 'dense_quadratic')}, q_pos={FLAGS.q_pos}, q_vel={FLAGS.q_vel}, q_ang={FLAGS.q_ang}, q_psi={effective_psi}, reward_norm={FLAGS.reward_norm}, action_penalty={FLAGS.action_penalty}, target_pz={FLAGS.target_pz}")
    print(f'  obs_feature_mode={FLAGS.obs_feature_mode}, obs_h_scale={FLAGS.obs_h_scale}, obs_hdot_scale={FLAGS.obs_hdot_scale}, obs_reach_scale={FLAGS.obs_reach_scale}, obs_goal_pos_radius={FLAGS.obs_goal_pos_radius}, obs_goal_vel_radius={FLAGS.obs_goal_vel_radius}, obs_goal_angle_radius={FLAGS.obs_goal_angle_radius}')
    if split_kwargs:
        print(f"  axis_split_q=[q_pos_xy={split_kwargs['q_pos_xy']}, q_pos_z={split_kwargs['q_pos_z']}, q_vel_xy={split_kwargs['q_vel_xy']}, q_vel_z={split_kwargs['q_vel_z']}]")
    if FLAGS.sdf_mode != 'raw':
        print(f'  sdf_shaping=[mode={FLAGS.sdf_mode}, unsafe_const={FLAGS.sdf_unsafe_const}, unsafe_slope={FLAGS.sdf_unsafe_slope}, global_scale={FLAGS.sdf_global_scale}]')
    if FLAGS.boundary_replay:
        print(f'  boundary_replay=[mode={FLAGS.boundary_mode}, eps={FLAGS.boundary_eps}, frac={FLAGS.boundary_frac}, min_count={FLAGS.boundary_min_count}]')
    if FLAGS.init_curriculum:
        print(f'  init_curriculum=[mode={FLAGS.init_curriculum_mode}, frac={FLAGS.init_curriculum_frac}, pos_xy_min={FLAGS.init_curriculum_pos_xy_min}, schedule={FLAGS.init_curriculum_schedule}, anneal_start={FLAGS.init_curriculum_anneal_start}, anneal_end={FLAGS.init_curriculum_anneal_end}, anneal_end_frac={FLAGS.init_curriculum_anneal_end_frac}]')
    actor_hidden = tuple((int(x) for x in FLAGS.actor_hidden.split(',')))
    critic_hidden = tuple((int(x) for x in FLAGS.critic_hidden.split(',')))
    create_kwargs = dict(seed=FLAGS.seed, observation_space=env.observation_space, action_space=env.action_space, actor_hidden_dims=actor_hidden, critic_hidden_dims=critic_hidden, M_q=FLAGS.M_q, alpha_r=FLAGS.alpha_r, beta_safety=FLAGS.beta_safety, delta=FLAGS.delta, discount=FLAGS.discount, discount_h=FLAGS.discount_h, tau=FLAGS.tau, T=FLAGS.T, beta_schedule=FLAGS.beta_schedule, cost_critic_hyperparam=FLAGS.cost_critic_hyperparam, safety_estimator='sampled_qh', pure_qsm=FLAGS.pure_qsm, ddpm_temperature=FLAGS.ddpm_temperature, fuse_clip_min=FLAGS.fuse_clip_min, fuse_clip_max=FLAGS.fuse_clip_max, huber_delta=FLAGS.huber_delta)
    create_kwargs.update(stage_mode=FLAGS.qh_stage_mode, qh_candidate_source=FLAGS.qh_candidate_source, qh_min_k_samples=FLAGS.qh_min_k_samples, qh_min_k_sigma=FLAGS.qh_min_k_sigma)
    agent = SSMOnlineAgent.create(**create_kwargs)
    print(f'SSM policy: M_q={FLAGS.M_q}, beta={FLAGS.beta_safety}')
    print(f'  sampled_Qh=[mode={FLAGS.qh_stage_mode}, source={FLAGS.qh_candidate_source}, K={FLAGS.qh_min_k_samples}, sigma={FLAGS.qh_min_k_sigma}]')
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    replay_buffer = SimpleReplayBuffer(obs_dim, act_dim, max_size=FLAGS.max_steps + 10000, boundary_replay=FLAGS.boundary_replay, boundary_mode=FLAGS.boundary_mode, boundary_eps=FLAGS.boundary_eps, boundary_frac=FLAGS.boundary_frac, boundary_min_count=FLAGS.boundary_min_count)
    replay_buffer.seed(FLAGS.seed)
    env.set_train_step(0)
    obs = env.reset()
    done = False
    episode_count = 0
    start_time = time.time()
    for step in tqdm(range(1, FLAGS.max_steps + 1), smoothing=0.1, disable=not FLAGS.tqdm_bar):
        env.set_train_step(step)
        if step < FLAGS.start_training:
            action = env.action_space.sample()
        else:
            action, agent = agent.sample_actions(obs)
            action = np.asarray(action)
        next_obs, reward, h_val, binary_cost, done, info = env.step(action)
        next_state = env.state.copy()
        pz_margin = float(next_state[env.PZ] - env.safe_pz)
        radius_margin = float(np.linalg.norm(next_state) - env.safe_radius)
        mask = 0.0 if info.get('terminated', False) else 1.0
        replay_buffer.insert(dict(observations=obs, actions=action, rewards=reward, costs=h_val, masks=mask, next_observations=next_obs, pz_margin=pz_margin, radius_margin=radius_margin))
        obs = next_obs
        if done:
            episode_count += 1
            ep_info = env.episode_info
            if FLAGS.wandb:
                import wandb
                wandb.log(compact_episode_log(ep_info, episode_count=episode_count, terminated=info.get('terminated', False)), step=step)
            obs = env.reset()
            done = False
        if step >= FLAGS.start_training:
            batch = replay_buffer.sample(FLAGS.batch_size)
            grad_step = step - FLAGS.start_training
            if FLAGS.actor_delay <= 1 or grad_step % FLAGS.actor_delay == 0:
                agent, update_info = agent.update(batch)
            else:
                agent, update_info = agent.update_critic_only(batch)
            if step % FLAGS.log_interval == 0:
                log_dict = compact_update_log(update_info, sps=step / (time.time() - start_time))
                if FLAGS.wandb:
                    import wandb
                    wandb.log(log_dict, step=step)
                else:
                    print(f"[{step}] actor={float(update_info.get('actor_loss', 0)):.3f} sf={float(update_info.get('safe_frac', 0)):.3f} ind={float(update_info.get('indicator_frac', 0)):.3f} vh={float(update_info.get('vh_mean', 0)):.3f} qc={float(update_info.get('qc_mean', 0)):.3f} br={replay_buffer.last_sample_info['boundary_frac_in_batch']:.3f} uf={replay_buffer.last_sample_info['unsafe_frac_in_batch']:.3f}")
        if step % FLAGS.checkpoint_interval == 0 or step == FLAGS.max_steps:
            save_checkpoint(agent, env, f'checkpoints/{run_name}/step_{step}.pkl', POLICY_NAME)
    print('Training complete.')
    if FLAGS.wandb:
        import wandb
        wandb.finish()
if __name__ == '__main__':
    app.run(main)
