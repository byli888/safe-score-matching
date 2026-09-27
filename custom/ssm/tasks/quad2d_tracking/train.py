"""Train SSM on the Quad2D trajectory-tracking task."""
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
from ssm.tasks.quad2d_tracking.env import Quad2DEnv
FLAGS = flags.FLAGS
flags.DEFINE_integer('checkpoint_interval', 20000, 'Steps between model saves.')
flags.DEFINE_string('project_name', 'safefm-hj', 'wandb project name.')
flags.DEFINE_string('entity', '', 'wandb entity.')
flags.DEFINE_string('run_name', '', 'wandb run name.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_integer('max_steps', 200000, 'Total environment steps.')
flags.DEFINE_integer('start_training', 5000, 'Random warmup steps.')
flags.DEFINE_integer('batch_size', 256, 'Batch size.')
flags.DEFINE_integer('log_interval', 1000, 'Steps between logging.')
flags.DEFINE_boolean('wandb', False, 'Enable wandb logging.')
flags.DEFINE_boolean('tqdm_bar', True, 'Show tqdm progress bar.')
flags.DEFINE_boolean('boundary_replay', False, 'Enable boundary replay oversampling using true SDF h(s).')
flags.DEFINE_enum('boundary_mode', 'abs', ['abs', 'near_unsafe'], 'Boundary criterion on true SDF: |h|<=eps or h>=-eps.')
flags.DEFINE_float('boundary_eps', 0.05, 'Boundary replay threshold eps on true SDF h(s).')
flags.DEFINE_float('boundary_frac', 0.25, 'Target fraction of each batch drawn from boundary set.')
flags.DEFINE_integer('boundary_min_count', 32, 'Minimum distinct boundary samples before oversampling activates.')
POLICY_NAME = 'ssm'
flags.DEFINE_float('M_q', 10.0, 'QSM scaling factor.')
flags.DEFINE_float('alpha_r', 1.0, 'Reward gradient weight.')
flags.DEFINE_float('beta_safety', 5.0, 'Safety gradient weight.')
flags.DEFINE_float('delta', 0.0, 'Safe set threshold.')
flags.DEFINE_float('discount', 0.99, 'Reward discount.')
flags.DEFINE_float('discount_h', 0.999, 'Safety discount.')
flags.DEFINE_float('tau', 0.005, 'EMA rate.')
flags.DEFINE_float('cost_critic_hyperparam', 0.9, 'Reversed expectile for V_h.')
flags.DEFINE_boolean('ref_velocity', True, 'Use moving reference velocity in Quad2D waypoints (RCRL-style).')
flags.DEFINE_float('q_vel', 5.0, 'Velocity tracking weight in Q matrix. RCRL paper uses 1.0.')
flags.DEFINE_float('action_penalty', 0.01, 'Action penalty weight in R matrix. RCRL paper uses 1e-4.')
flags.DEFINE_string('sdf_mode', 'baseline', 'SDF shaping mode for Quad2D cost signal. Options: baseline, hard1, hard10, hard10_safe20. All modes preserve sign(h) and the h=0 boundary.')
flags.DEFINE_string('beta_schedule', 'vp', 'DDPM beta schedule.')
flags.DEFINE_integer('T', 5, 'DDPM diffusion steps.')
flags.DEFINE_integer('actor_delay', 1, 'Actor update every N critic updates.')
flags.DEFINE_string('actor_hidden', '256,256,256', 'Actor hidden dims.')
flags.DEFINE_string('critic_hidden', '256,256', 'Critic hidden dims.')
flags.DEFINE_boolean('pure_qsm', False, 'Pure QSM ablation.')
flags.DEFINE_float('ddpm_temperature', 0.2, 'DDPM sampling temperature (0.2 for Quad2D).')
flags.DEFINE_float('fuse_clip_min', -50.0, 'Bellman target clip lower bound (-5000 for Quad2D).')
flags.DEFINE_float('fuse_clip_max', 100.0, 'Bellman target clip upper bound.')
flags.DEFINE_float('huber_delta', 10.0, 'Huber delta for reward critic. 0=MSE.')

def save_checkpoint(agent, env, path, policy_name):
    """Save agent params + obs normalization stats."""
    params = jax.device_get({'score_model': agent.score_model.params, 'critic_1': agent.critic_1.params, 'critic_2': agent.critic_2.params, 'target_critic_1': agent.target_critic_1.params, 'target_critic_2': agent.target_critic_2.params, 'safe_critic': agent.safe_critic.params, 'safe_target_critic': agent.safe_target_critic.params, 'safe_value': agent.safe_value.params, 'safe_target_value': agent.safe_target_value.params})
    if hasattr(agent, 'beta_net'):
        params['beta_net'] = jax.device_get(agent.beta_net.params)
    data = {'params': params, 'policy': policy_name, 'rng': jax.device_get(agent.rng), 'obs_mean': env._obs_mean.copy(), 'obs_var': env._obs_var.copy(), 'obs_count': env._obs_count, 'config': {k: FLAGS[k].value for k in FLAGS}}
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'  Checkpoint saved: {path} ({os.path.getsize(path) / 1000000.0:.1f} MB)')

class SimpleReplayBuffer:
    """Simple numpy replay buffer for Quad2D with optional boundary oversampling."""

    def __init__(self, obs_dim, act_dim, max_size=1000000, boundary_replay=False, boundary_mode='abs', boundary_eps=0.05, boundary_frac=0.25, boundary_min_count=32):
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

    def _boundary_mask(self, costs):
        if self.boundary_mode == 'near_unsafe':
            return costs >= -self.boundary_eps
        return np.abs(costs) <= self.boundary_eps

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
        cost = float(self.costs[i])
        self._boundary_size = self._update_membership(i, bool(self._boundary_mask(cost)), self._boundary_slots, self._boundary_pos, self._boundary_size)
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
        self.last_sample_info = {'boundary_active': float(boundary_active), 'boundary_draw_frac': num_boundary / batch_size, 'boundary_frac_in_buffer': self._boundary_size / max(self.size, 1), 'unsafe_frac_in_buffer': self._unsafe_size / max(self.size, 1), 'boundary_frac_in_batch': float(np.mean(self._boundary_mask(batch_costs))), 'unsafe_frac_in_batch': float(np.mean(batch_costs > 0.0)), 'costs_mean_in_batch': float(np.mean(batch_costs))}
        return {'observations': jnp.array(self.observations[idxs]), 'actions': jnp.array(self.actions[idxs]), 'rewards': jnp.array(self.rewards[idxs]), 'costs': jnp.array(self.costs[idxs]), 'masks': jnp.array(self.masks[idxs]), 'next_observations': jnp.array(self.next_observations[idxs])}

def main(_):
    from ssm.agents.ssm_agent import SSMOnlineAgent
    if not FLAGS.run_name:
        parts = ['quad2d_tracking_ssm', f'beta{FLAGS.beta_safety:g}', f'mq{FLAGS.M_q:g}']
        if FLAGS.fuse_clip_min != -50.0:
            parts.append(f'fclip{FLAGS.fuse_clip_min:g}')
        if FLAGS.huber_delta != 10.0:
            parts.append(f'huber{FLAGS.huber_delta:g}')
        if FLAGS.boundary_replay:
            parts.append(f'br{FLAGS.boundary_mode}_e{FLAGS.boundary_eps:g}_f{FLAGS.boundary_frac:g}')
        parts.append(f's{FLAGS.seed}')
        run_name = '_'.join(parts)
    else:
        run_name = FLAGS.run_name
    if FLAGS.wandb:
        import wandb
        config = {'seed': FLAGS.seed, 'env': 'Quad2D', 'policy': POLICY_NAME, 'M_q': FLAGS.M_q, 'alpha_r': FLAGS.alpha_r, 'beta_safety': FLAGS.beta_safety, 'delta': FLAGS.delta, 'discount': FLAGS.discount, 'discount_h': FLAGS.discount_h, 'tau': FLAGS.tau, 'T': FLAGS.T, 'beta_schedule': FLAGS.beta_schedule, 'cost_critic_hyperparam': FLAGS.cost_critic_hyperparam, 'safety_estimator': 'expectile', 'batch_size': FLAGS.batch_size, 'actor_delay': FLAGS.actor_delay, 'max_steps': FLAGS.max_steps, 'start_training': FLAGS.start_training, 'actor_hidden': FLAGS.actor_hidden, 'critic_hidden': FLAGS.critic_hidden, 'pure_qsm': FLAGS.pure_qsm, 'ddpm_temperature': FLAGS.ddpm_temperature, 'fuse_clip_min': FLAGS.fuse_clip_min, 'fuse_clip_max': FLAGS.fuse_clip_max, 'huber_delta': FLAGS.huber_delta, 'ref_velocity': FLAGS.ref_velocity, 'q_vel': FLAGS.q_vel, 'action_penalty': FLAGS.action_penalty, 'sdf_mode': FLAGS.sdf_mode, 'boundary_replay': FLAGS.boundary_replay, 'boundary_mode': FLAGS.boundary_mode, 'boundary_eps': FLAGS.boundary_eps, 'boundary_frac': FLAGS.boundary_frac, 'boundary_min_count': FLAGS.boundary_min_count}
        wandb.init(project=FLAGS.project_name, entity=FLAGS.entity, name=run_name, config=config)
    env = Quad2DEnv(seed=FLAGS.seed, ref_velocity=FLAGS.ref_velocity, q_vel=FLAGS.q_vel, action_penalty=FLAGS.action_penalty, sdf_mode=FLAGS.sdf_mode)
    env.action_space.seed(FLAGS.seed)
    print(f'Env: Quad2D, obs={env.observation_space.shape}, act={env.action_space.shape}')
    actor_hidden = tuple((int(x) for x in FLAGS.actor_hidden.split(',')))
    critic_hidden = tuple((int(x) for x in FLAGS.critic_hidden.split(',')))
    create_kwargs = dict(seed=FLAGS.seed, observation_space=env.observation_space, action_space=env.action_space, actor_hidden_dims=actor_hidden, critic_hidden_dims=critic_hidden, M_q=FLAGS.M_q, alpha_r=FLAGS.alpha_r, beta_safety=FLAGS.beta_safety, delta=FLAGS.delta, discount=FLAGS.discount, discount_h=FLAGS.discount_h, tau=FLAGS.tau, T=FLAGS.T, beta_schedule=FLAGS.beta_schedule, cost_critic_hyperparam=FLAGS.cost_critic_hyperparam, safety_estimator='expectile', pure_qsm=FLAGS.pure_qsm, ddpm_temperature=FLAGS.ddpm_temperature, fuse_clip_min=FLAGS.fuse_clip_min, fuse_clip_max=FLAGS.fuse_clip_max, huber_delta=FLAGS.huber_delta)
    agent = SSMOnlineAgent.create(**create_kwargs)
    print(f'SSM policy: M_q={FLAGS.M_q}, beta={FLAGS.beta_safety}')
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    replay_buffer = SimpleReplayBuffer(obs_dim, act_dim, max_size=FLAGS.max_steps + 10000, boundary_replay=FLAGS.boundary_replay, boundary_mode=FLAGS.boundary_mode, boundary_eps=FLAGS.boundary_eps, boundary_frac=FLAGS.boundary_frac, boundary_min_count=FLAGS.boundary_min_count)
    replay_buffer.seed(FLAGS.seed)
    obs = env.reset()
    done = False
    episode_count = 0
    start_time = time.time()
    for step in tqdm(range(1, FLAGS.max_steps + 1), smoothing=0.1, disable=not FLAGS.tqdm_bar):
        if step < FLAGS.start_training:
            action = env.action_space.sample()
        else:
            action, agent = agent.sample_actions(obs)
            action = np.asarray(action)
        next_obs, reward, h_val, binary_cost, done, info = env.step(action)
        mask = 0.0 if info.get('terminated', False) else 1.0
        replay_buffer.insert(dict(observations=obs, actions=action, rewards=reward, costs=h_val, masks=mask, next_observations=next_obs))
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
                    beta_str = ''
                    if 'beta_mean' in update_info:
                        beta_str = f" β_mean={float(update_info['beta_mean']):.2f}"
                    ratio_str = ''
                    if 'safety_reward_ratio' in update_info:
                        ratio_str = f" s/r={float(update_info['safety_reward_ratio']):.2f}"
                    print(f"[{step}] actor={float(update_info.get('actor_loss', 0)):.3f} sf={float(update_info.get('safe_frac', 0)):.3f} ind={float(update_info.get('indicator_frac', 0)):.3f} vh={float(update_info.get('vh_mean', 0)):.3f} qc={float(update_info.get('qc_mean', 0)):.3f}{beta_str}{ratio_str}")
        if step % FLAGS.checkpoint_interval == 0 or step == FLAGS.max_steps:
            save_checkpoint(agent, env, f'checkpoints/{run_name}/step_{step}.pkl', POLICY_NAME)
    print('Training complete.')
    if FLAGS.wandb:
        import wandb
        wandb.finish()
if __name__ == '__main__':
    app.run(main)
