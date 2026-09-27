"""Online training for the F16 stabilize-avoid v6 benchmark with SSM."""
import json
import os
import pickle
import time
from typing import Dict, Tuple
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
import jax
import numpy as np
from absl import app, flags
from flax import serialization as flax_serialization
from tqdm import tqdm
import ssm.tasks.f16.env as env_v6
from ssm.tasks.f16.env import F16StabilizeEnvV6, INIT_CURRICULUM_MODE_AB_BOUNDARY, INIT_CURRICULUM_MODE_BOX, INIT_CURRICULUM_MODE_BOX_MIXTURE, INIT_CURRICULUM_MODE_EASY_TO_HARD, INIT_CURRICULUM_MODE_THETA_TAIL_PROXY, OBS_TASK_FEATS_MODE_PER_AXIS, REGION_PROFILE_EFPPO_PLUS_P, REGION_PROFILE_V6, TASK_SPEC_VERSION_V6, _DEFAULT_H_THETA_DENOM
from ssm.common.logging import compact_episode_log, compact_update_log
FLAGS = flags.FLAGS
flags.DEFINE_string('project_name', 'safefm-hj', 'wandb project name.')
flags.DEFINE_string('entity', '', 'wandb entity.')
flags.DEFINE_string('run_name', '', 'wandb run name.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_integer('max_steps', 500000, 'Total environment steps.')
flags.DEFINE_integer('start_training', 20000, 'Random warmup steps.')
flags.DEFINE_boolean('seed_warmup_actions', False, 'Seed the action-space stream before agent creation and random warmup.')
flags.DEFINE_integer('batch_size', 512, 'Batch size.')
flags.DEFINE_float('termination_penalty', 0.0, 'Positive reward-penalty magnitude applied to true crash termination transitions.')
flags.DEFINE_integer('checkpoint_interval', 20000, 'Steps between model saves.')
flags.DEFINE_integer('log_interval', 1000, 'Steps between logging.')
flags.DEFINE_string('resume_from', '', 'Optional resume checkpoint created by this script.')
flags.DEFINE_boolean('save_resume_checkpoint', True, 'Save resumable training checkpoints.')
flags.DEFINE_integer('resume_checkpoint_interval', 0, 'Steps between resumable checkpoint saves; <=0 uses checkpoint_interval.')
flags.DEFINE_boolean('wandb', False, 'Enable wandb logging.')
flags.DEFINE_boolean('tqdm_bar', True, 'Show tqdm progress bar.')
flags.DEFINE_boolean('fused_update', False, 'Use the already-jitted fused agent update()/update_critic_only() path.')
flags.DEFINE_boolean('fuse_next_action', False, 'Fuse learner updates with next-action sampling between model saves.')
flags.DEFINE_boolean('sync_timing', False, 'Block on selected JAX outputs before stopping timers for more trustworthy profiling.')
POLICY_NAME = 'ssm'
flags.DEFINE_float('M_q', 10.0, 'QSM scaling factor.')
flags.DEFINE_float('alpha_r', 1.0, 'Reward gradient weight.')
flags.DEFINE_float('beta_safety', 5.0, 'Safety gradient weight.')
flags.DEFINE_float('delta', 0.0, 'Safe set threshold.')
flags.DEFINE_float('hard_margin', 0.0, 'Safe routing margin.')
flags.DEFINE_float('discount', 0.995, 'Reward discount.')
flags.DEFINE_float('discount_h', 0.999, 'Safety discount.')
flags.DEFINE_float('tau', 0.005, 'EMA rate.')
flags.DEFINE_float('cost_critic_hyperparam', 0.9, 'Reversed expectile for V_h.')
flags.DEFINE_string('vc_mode', 'buffer_expectile', 'Safety-value regression mode: buffer_expectile, hybrid_expectile, hybrid_min.')
flags.DEFINE_integer('vc_policy_samples', 0, 'Additional current-policy action samples per state for update_vc.')
flags.DEFINE_integer('vc_proposal_samples', 0, 'Additional clipped-Gaussian proposal actions per state for update_vc.')
flags.DEFINE_float('vc_proposal_std', 0.6, 'Std of clipped-Gaussian proposal actions around the current policy action for update_vc.')
flags.DEFINE_enum('qh_stage_mode', 'stage_a', ['stage_a', 'stage_b'], 'Q_h-only safety mode: stage_a uses Q_h(s, pi(s)); stage_b uses sampled-action min_k Q_h.')
flags.DEFINE_enum('qh_candidate_source', 'policy_plus_gaussian', ['policy_only', 'policy_plus_gaussian'], 'Candidate source for stage_b sampled-action Q_h estimators.')
flags.DEFINE_integer('qh_min_k_samples', 8, 'Candidate count K for stage_b min_k Q_h estimators.')
flags.DEFINE_float('qh_min_k_sigma', 0.3, 'Clipped-Gaussian proposal std around the policy action for stage_b.')
flags.DEFINE_string('beta_schedule', 'vp', 'DDPM beta schedule.')
flags.DEFINE_integer('T', 5, 'DDPM diffusion steps.')
flags.DEFINE_integer('actor_delay', 1, 'Actor update every N critic updates.')
flags.DEFINE_string('actor_hidden', '256,256,256', 'Actor hidden dims.')
flags.DEFINE_string('critic_hidden', '256,256', 'Critic hidden dims.')
flags.DEFINE_boolean('pure_qsm', False, 'Pure QSM ablation.')
flags.DEFINE_float('ddpm_temperature', 0.2, 'DDPM sampling temperature.')
flags.DEFINE_float('fuse_clip_min', -2000.0, 'Lower clip for fused reward target.')
flags.DEFINE_float('fuse_clip_max', 100.0, 'Upper clip for fused reward target.')
flags.DEFINE_float('huber_delta', 10.0, 'Huber loss delta for reward critics.')
flags.DEFINE_integer('goal_dwell_steps', 50, 'Debug-only dwell counter in the goal band.')
flags.DEFINE_integer('max_episode_steps', 640, 'Training episode horizon.')
flags.DEFINE_boolean('terminate_on_crash', True, 'End the episode immediately on crash.')
flags.DEFINE_string('safety_gap_mode', 'nogap', 'One of: literal, nogap.')
flags.DEFINE_string('h_form', 'linear', 'Safety margin form for v6.')
flags.DEFINE_float('h_theta_denom', _DEFAULT_H_THETA_DENOM, 'Theta denominator derived from v6 geometry.')
flags.DEFINE_string('obs_task_feats_mode', OBS_TASK_FEATS_MODE_PER_AXIS, 'Observation task feature mode: aggregate, per_axis.')
flags.DEFINE_string('l_form', 'split_log', 'Task cost form: split_linear, split_log, split_atan.')
flags.DEFINE_float('l_q', 0.1, 'Quadratic task-cost scale for l_form=pure_quadratic.')
flags.DEFINE_float('l_q_center', 0.03, 'Band-interior bowl scale for v6 split l.')
flags.DEFINE_float('l_q_out', 0.16, 'Band-exterior tail scale for v6 split l.')
flags.DEFINE_float('l_kappa', 1.0, 'Saturation parameter for split_log/split_atan.')
flags.DEFINE_string('l_stage_form', 'band_bowl', 'Optional stage reward form: none, band_bowl.')
flags.DEFINE_float('l_stage_reward', 0.15, 'Stage reward magnitude for l_stage_form=band_bowl.')
flags.DEFINE_enum('reset_box_mode', 'ours', ['ours', 'efppo_train'], 'Proposal box used for training resets.')
flags.DEFINE_enum('region_profile', REGION_PROFILE_V6, [REGION_PROFILE_V6, REGION_PROFILE_EFPPO_PLUS_P], "Region geometry profile. 'v6' keeps the current geometry; 'efppo_plus_p' uses EFPPO-style H/theta/terminate-H geometry while retaining the P safety channel.")
flags.DEFINE_boolean('reset_requires_safe', True, 'Whether box-based init samplers reject-and-resample until h(x0) < 0. If false, box samplers only reject non-finite states.')
flags.DEFINE_float('safe_alpha_hi', 0.7853981633974483, 'Safe-set upper alpha bound for v6 h.')
flags.DEFINE_float('safe_beta', 0.5235987755982988, 'Safe-set |beta| bound for v6 h.')
flags.DEFINE_float('train_safe_h_min', float('nan'), 'Optional train-only safe H lower bound. NaN keeps region_profile geometry.')
flags.DEFINE_float('train_safe_h_max', float('nan'), 'Optional train-only safe H upper bound. NaN keeps region_profile geometry.')
flags.DEFINE_float('train_safe_alpha_lo', float('nan'), 'Optional train-only safe alpha lower bound. NaN keeps canonical lower bound.')
flags.DEFINE_float('train_safe_alpha_hi', float('nan'), 'Optional train-only safe alpha upper bound. NaN keeps --safe_alpha_hi.')
flags.DEFINE_float('train_safe_beta', float('nan'), 'Optional train-only safe |beta| bound. NaN keeps --safe_beta.')
flags.DEFINE_boolean('alpha_counts_as_invalid_dynamics', True, 'Whether alpha-limit violations immediately count as invalid dynamics in the training env.')
flags.DEFINE_boolean('beta_counts_as_invalid_dynamics', True, 'Whether beta-limit violations immediately count as invalid dynamics in the training env.')
flags.DEFINE_boolean('clip_relaxed_alpha_beta_to_terminate', False, 'If alpha/beta invalid-dynamics are relaxed, clip next-state alpha/beta back to terminate bounds before continuing.')
flags.DEFINE_boolean('init_curriculum', False, 'Enable training-only reset curriculum.')
flags.DEFINE_enum('init_curriculum_mode', INIT_CURRICULUM_MODE_BOX, [INIT_CURRICULUM_MODE_BOX, INIT_CURRICULUM_MODE_BOX_MIXTURE, INIT_CURRICULUM_MODE_AB_BOUNDARY, INIT_CURRICULUM_MODE_THETA_TAIL_PROXY, INIT_CURRICULUM_MODE_EASY_TO_HARD], "Curriculum reset mode. 'box' keeps the existing absolute-box sampler; 'box_mixture' uses init_curriculum_* as box1 and init_curriculum_box2_* as box2, with both frac values interpreted as absolute reset probabilities; 'ab_boundary' mixes in alpha/beta-near-boundary safe resets relative to the current safety geometry; 'theta_tail_proxy' uses a fixed-theta/H proxy sampler; 'easy_to_hard' starts near benign trim/goal-approach states and widens to the active reset box.")
flags.DEFINE_float('init_curriculum_frac', 0.0, 'Probability of using the curriculum reset branch; box1 probability when mode=box_mixture.')
flags.DEFINE_float('init_curriculum_frac_end', 0.0, 'Final curriculum reset probability after annealing.')
flags.DEFINE_float('init_curriculum_box2_frac', 0.0, 'Probability of using curriculum box2 when mode=box_mixture.')
flags.DEFINE_float('init_curriculum_box2_frac_end', 0.0, 'Final probability of curriculum box2 after annealing when mode=box_mixture.')
flags.DEFINE_integer('init_curriculum_anneal_start', 120000, 'Train step where curriculum annealing begins.')
flags.DEFINE_integer('init_curriculum_anneal_end', 400000, 'Train step where curriculum annealing ends.')
flags.DEFINE_bool('fail_on_curriculum_no_anneal_mismatch', True, 'When init curriculum anneal is [0,0], abort if any effective curriculum start/end bound differs.')
flags.DEFINE_float('init_curriculum_h_min', float('nan'), 'Curriculum reset altitude lower bound; NaN inherits base reset box.')
flags.DEFINE_float('init_curriculum_h_max', float('nan'), 'Curriculum reset altitude upper bound; NaN inherits base reset box.')
flags.DEFINE_float('init_curriculum_h_min_end', float('nan'), 'Final curriculum altitude lower bound; NaN inherits start/base.')
flags.DEFINE_float('init_curriculum_h_max_end', float('nan'), 'Final curriculum altitude upper bound; NaN inherits start/base.')
flags.DEFINE_float('init_curriculum_box2_h_min', float('nan'), 'Curriculum box2 altitude lower bound; NaN inherits base reset box.')
flags.DEFINE_float('init_curriculum_box2_h_max', float('nan'), 'Curriculum box2 altitude upper bound; NaN inherits base reset box.')
flags.DEFINE_float('init_curriculum_box2_h_min_end', float('nan'), 'Final curriculum box2 altitude lower bound; NaN inherits start/base.')
flags.DEFINE_float('init_curriculum_box2_h_max_end', float('nan'), 'Final curriculum box2 altitude upper bound; NaN inherits start/base.')
flags.DEFINE_float('init_curriculum_pe_abs_min', float('nan'), 'Curriculum reset |PE| lower bound; NaN inherits base reset box.')
flags.DEFINE_float('init_curriculum_pe_abs_max', float('nan'), 'Curriculum reset |PE| upper bound; NaN inherits base reset box.')
flags.DEFINE_float('init_curriculum_pe_abs_min_end', float('nan'), 'Final curriculum reset |PE| lower bound; NaN inherits start/base.')
flags.DEFINE_float('init_curriculum_pe_abs_max_end', float('nan'), 'Final curriculum reset |PE| upper bound; NaN inherits start/base.')
flags.DEFINE_float('init_curriculum_theta_lo', float('nan'), 'Curriculum reset theta lower bound; NaN inherits base reset box.')
flags.DEFINE_float('init_curriculum_theta_hi', float('nan'), 'Curriculum reset theta upper bound; NaN inherits base reset box.')
flags.DEFINE_float('init_curriculum_theta_lo_end', float('nan'), 'Final curriculum theta lower bound; NaN inherits start/base.')
flags.DEFINE_float('init_curriculum_theta_hi_end', float('nan'), 'Final curriculum theta upper bound; NaN inherits start/base.')
flags.DEFINE_float('init_curriculum_box2_theta_lo', float('nan'), 'Curriculum box2 theta lower bound; NaN inherits base reset box.')
flags.DEFINE_float('init_curriculum_box2_theta_hi', float('nan'), 'Curriculum box2 theta upper bound; NaN inherits base reset box.')
flags.DEFINE_float('init_curriculum_box2_theta_lo_end', float('nan'), 'Final curriculum box2 theta lower bound; NaN inherits start/base.')
flags.DEFINE_float('init_curriculum_box2_theta_hi_end', float('nan'), 'Final curriculum box2 theta upper bound; NaN inherits start/base.')
flags.DEFINE_float('init_curriculum_alpha_lo', float('nan'), 'Curriculum reset alpha lower bound; NaN inherits base reset box.')
flags.DEFINE_float('init_curriculum_alpha_hi', float('nan'), 'Curriculum reset alpha upper bound; NaN inherits base reset box.')
flags.DEFINE_float('init_curriculum_alpha_lo_end', float('nan'), 'Final curriculum alpha lower bound; NaN inherits start/base.')
flags.DEFINE_float('init_curriculum_alpha_hi_end', float('nan'), 'Final curriculum alpha upper bound; NaN inherits start/base.')
flags.DEFINE_float('init_curriculum_beta_abs_max', float('nan'), 'Curriculum reset |beta| upper bound; NaN inherits base reset box.')
flags.DEFINE_float('init_curriculum_beta_abs_max_end', float('nan'), 'Final curriculum |beta| upper bound; NaN inherits start/base.')
flags.DEFINE_enum('init_curriculum_ab_kind', 'mixed', ['alpha', 'beta', 'joint', 'mixed'], "Alpha/beta boundary-reset branch type when init_curriculum_mode=ab_boundary. 'mixed' randomly chooses alpha-only, beta-only, or joint boundary resets.")
flags.DEFINE_float('init_curriculum_ab_joint_frac', 0.0, 'When init_curriculum_ab_kind=mixed, probability of sampling both alpha and beta near their safety boundaries. Remaining mass is split evenly between alpha-only and beta-only.')
flags.DEFINE_float('init_curriculum_ab_alpha_h_lo', -0.6, 'Initial lower bound on normalized alpha safety margin h for boundary reset sampling (0=safety boundary, 1=terminate boundary, negative=safe interior).')
flags.DEFINE_float('init_curriculum_ab_alpha_h_hi', -0.15, 'Initial upper bound on normalized alpha safety margin h for boundary reset sampling.')
flags.DEFINE_float('init_curriculum_ab_alpha_h_lo_end', -0.6, 'Final lower bound on normalized alpha safety margin h for boundary reset sampling.')
flags.DEFINE_float('init_curriculum_ab_alpha_h_hi_end', -0.15, 'Final upper bound on normalized alpha safety margin h for boundary reset sampling.')
flags.DEFINE_float('init_curriculum_ab_beta_h_lo', -0.6, 'Initial lower bound on normalized |beta| safety margin h for boundary reset sampling.')
flags.DEFINE_float('init_curriculum_ab_beta_h_hi', -0.15, 'Initial upper bound on normalized |beta| safety margin h for boundary reset sampling.')
flags.DEFINE_float('init_curriculum_ab_beta_h_lo_end', -0.6, 'Final lower bound on normalized |beta| safety margin h for boundary reset sampling.')
flags.DEFINE_float('init_curriculum_ab_beta_h_hi_end', -0.15, 'Final upper bound on normalized |beta| safety margin h for boundary reset sampling.')
flags.DEFINE_float('init_curriculum_p_abs_max', float('nan'), 'Curriculum reset |P| upper bound; NaN inherits base reset box.')
flags.DEFINE_float('init_curriculum_p_abs_max_end', float('nan'), 'Final curriculum |P| upper bound; NaN inherits start/base.')
flags.DEFINE_float('init_curriculum_q_abs_max', float('nan'), 'Curriculum reset |Q| upper bound; NaN inherits base reset box.')
flags.DEFINE_float('init_curriculum_q_abs_max_end', float('nan'), 'Final curriculum |Q| upper bound; NaN inherits start/base.')
flags.DEFINE_float('init_curriculum_r_abs_max', float('nan'), 'Curriculum reset |R| upper bound; NaN inherits base reset box.')
flags.DEFINE_float('init_curriculum_r_abs_max_end', float('nan'), 'Final curriculum |R| upper bound; NaN inherits start/base.')
flags.DEFINE_boolean('init_curriculum_allow_in_goal', False, 'Allow curriculum resets that already start in goal.')
flags.DEFINE_boolean('init_curriculum_require_train_safe', True, 'Require curriculum resets to satisfy task safety.')
_CONFIG_EXPORT_KEYS = ['fuse_next_action', 'seed_warmup_actions', 'project_name', 'entity', 'run_name', 'seed', 'max_steps', 'start_training', 'batch_size', 'termination_penalty', 'checkpoint_interval', 'log_interval', 'resume_from', 'save_resume_checkpoint', 'resume_checkpoint_interval', 'fused_update', 'sync_timing', 'M_q', 'alpha_r', 'beta_safety', 'delta', 'hard_margin', 'discount', 'discount_h', 'tau', 'cost_critic_hyperparam', 'vc_mode', 'vc_policy_samples', 'vc_proposal_samples', 'vc_proposal_std', 'qh_stage_mode', 'qh_candidate_source', 'qh_min_k_samples', 'qh_min_k_sigma', 'beta_schedule', 'T', 'actor_delay', 'actor_hidden', 'critic_hidden', 'pure_qsm', 'ddpm_temperature', 'fuse_clip_min', 'fuse_clip_max', 'huber_delta', 'goal_dwell_steps', 'max_episode_steps', 'terminate_on_crash', 'safety_gap_mode', 'h_form', 'h_theta_denom', 'obs_task_feats_mode', 'l_form', 'l_q', 'l_q_center', 'l_q_out', 'l_kappa', 'l_stage_form', 'l_stage_reward', 'reset_box_mode', 'region_profile', 'reset_requires_safe', 'safe_alpha_hi', 'safe_beta', 'train_safe_h_min', 'train_safe_h_max', 'train_safe_alpha_lo', 'train_safe_alpha_hi', 'train_safe_beta', 'alpha_counts_as_invalid_dynamics', 'beta_counts_as_invalid_dynamics', 'clip_relaxed_alpha_beta_to_terminate', 'init_curriculum', 'init_curriculum_mode', 'init_curriculum_frac', 'init_curriculum_frac_end', 'init_curriculum_box2_frac', 'init_curriculum_box2_frac_end', 'init_curriculum_anneal_start', 'init_curriculum_anneal_end', 'fail_on_curriculum_no_anneal_mismatch', 'init_curriculum_h_min', 'init_curriculum_h_max', 'init_curriculum_h_min_end', 'init_curriculum_h_max_end', 'init_curriculum_box2_h_min', 'init_curriculum_box2_h_max', 'init_curriculum_box2_h_min_end', 'init_curriculum_box2_h_max_end', 'init_curriculum_pe_abs_min', 'init_curriculum_pe_abs_max', 'init_curriculum_pe_abs_min_end', 'init_curriculum_pe_abs_max_end', 'init_curriculum_theta_lo', 'init_curriculum_theta_hi', 'init_curriculum_theta_lo_end', 'init_curriculum_theta_hi_end', 'init_curriculum_box2_theta_lo', 'init_curriculum_box2_theta_hi', 'init_curriculum_box2_theta_lo_end', 'init_curriculum_box2_theta_hi_end', 'init_curriculum_alpha_lo', 'init_curriculum_alpha_hi', 'init_curriculum_alpha_lo_end', 'init_curriculum_alpha_hi_end', 'init_curriculum_beta_abs_max', 'init_curriculum_beta_abs_max_end', 'init_curriculum_ab_kind', 'init_curriculum_ab_joint_frac', 'init_curriculum_ab_alpha_h_lo', 'init_curriculum_ab_alpha_h_hi', 'init_curriculum_ab_alpha_h_lo_end', 'init_curriculum_ab_alpha_h_hi_end', 'init_curriculum_ab_beta_h_lo', 'init_curriculum_ab_beta_h_hi', 'init_curriculum_ab_beta_h_lo_end', 'init_curriculum_ab_beta_h_hi_end', 'init_curriculum_p_abs_max', 'init_curriculum_p_abs_max_end', 'init_curriculum_q_abs_max', 'init_curriculum_q_abs_max_end', 'init_curriculum_r_abs_max', 'init_curriculum_r_abs_max_end', 'init_curriculum_allow_in_goal', 'init_curriculum_require_train_safe']

def make_env_v6(seed: int) -> F16StabilizeEnvV6:
    return F16StabilizeEnvV6(seed=seed, max_episode_steps=FLAGS.max_episode_steps, goal_dwell_steps=FLAGS.goal_dwell_steps, terminate_on_crash=FLAGS.terminate_on_crash, init_curriculum=FLAGS.init_curriculum, init_curriculum_mode=FLAGS.init_curriculum_mode, init_curriculum_frac=FLAGS.init_curriculum_frac, init_curriculum_frac_end=FLAGS.init_curriculum_frac_end, init_curriculum_box2_frac=FLAGS.init_curriculum_box2_frac, init_curriculum_box2_frac_end=FLAGS.init_curriculum_box2_frac_end, init_curriculum_anneal_start=FLAGS.init_curriculum_anneal_start, init_curriculum_anneal_end=FLAGS.init_curriculum_anneal_end, init_curriculum_h_min=FLAGS.init_curriculum_h_min, init_curriculum_h_max=FLAGS.init_curriculum_h_max, init_curriculum_h_min_end=FLAGS.init_curriculum_h_min_end, init_curriculum_h_max_end=FLAGS.init_curriculum_h_max_end, init_curriculum_box2_h_min=FLAGS.init_curriculum_box2_h_min, init_curriculum_box2_h_max=FLAGS.init_curriculum_box2_h_max, init_curriculum_box2_h_min_end=FLAGS.init_curriculum_box2_h_min_end, init_curriculum_box2_h_max_end=FLAGS.init_curriculum_box2_h_max_end, init_curriculum_pe_abs_min=FLAGS.init_curriculum_pe_abs_min, init_curriculum_pe_abs_max=FLAGS.init_curriculum_pe_abs_max, init_curriculum_pe_abs_min_end=FLAGS.init_curriculum_pe_abs_min_end, init_curriculum_pe_abs_max_end=FLAGS.init_curriculum_pe_abs_max_end, init_curriculum_theta_lo=FLAGS.init_curriculum_theta_lo, init_curriculum_theta_hi=FLAGS.init_curriculum_theta_hi, init_curriculum_theta_lo_end=FLAGS.init_curriculum_theta_lo_end, init_curriculum_theta_hi_end=FLAGS.init_curriculum_theta_hi_end, init_curriculum_box2_theta_lo=FLAGS.init_curriculum_box2_theta_lo, init_curriculum_box2_theta_hi=FLAGS.init_curriculum_box2_theta_hi, init_curriculum_box2_theta_lo_end=FLAGS.init_curriculum_box2_theta_lo_end, init_curriculum_box2_theta_hi_end=FLAGS.init_curriculum_box2_theta_hi_end, init_curriculum_alpha_lo=FLAGS.init_curriculum_alpha_lo, init_curriculum_alpha_hi=FLAGS.init_curriculum_alpha_hi, init_curriculum_alpha_lo_end=FLAGS.init_curriculum_alpha_lo_end, init_curriculum_alpha_hi_end=FLAGS.init_curriculum_alpha_hi_end, init_curriculum_beta_abs_max=FLAGS.init_curriculum_beta_abs_max, init_curriculum_beta_abs_max_end=FLAGS.init_curriculum_beta_abs_max_end, init_curriculum_ab_kind=FLAGS.init_curriculum_ab_kind, init_curriculum_ab_joint_frac=FLAGS.init_curriculum_ab_joint_frac, init_curriculum_ab_alpha_h_lo=FLAGS.init_curriculum_ab_alpha_h_lo, init_curriculum_ab_alpha_h_hi=FLAGS.init_curriculum_ab_alpha_h_hi, init_curriculum_ab_alpha_h_lo_end=FLAGS.init_curriculum_ab_alpha_h_lo_end, init_curriculum_ab_alpha_h_hi_end=FLAGS.init_curriculum_ab_alpha_h_hi_end, init_curriculum_ab_beta_h_lo=FLAGS.init_curriculum_ab_beta_h_lo, init_curriculum_ab_beta_h_hi=FLAGS.init_curriculum_ab_beta_h_hi, init_curriculum_ab_beta_h_lo_end=FLAGS.init_curriculum_ab_beta_h_lo_end, init_curriculum_ab_beta_h_hi_end=FLAGS.init_curriculum_ab_beta_h_hi_end, init_curriculum_p_abs_max=FLAGS.init_curriculum_p_abs_max, init_curriculum_p_abs_max_end=FLAGS.init_curriculum_p_abs_max_end, init_curriculum_q_abs_max=FLAGS.init_curriculum_q_abs_max, init_curriculum_q_abs_max_end=FLAGS.init_curriculum_q_abs_max_end, init_curriculum_r_abs_max=FLAGS.init_curriculum_r_abs_max, init_curriculum_r_abs_max_end=FLAGS.init_curriculum_r_abs_max_end, init_curriculum_allow_in_goal=FLAGS.init_curriculum_allow_in_goal, init_curriculum_require_train_safe=FLAGS.init_curriculum_require_train_safe, safety_gap_mode=FLAGS.safety_gap_mode, obs_task_feats_mode=FLAGS.obs_task_feats_mode, h_form=FLAGS.h_form, h_theta_denom=FLAGS.h_theta_denom, l_form=FLAGS.l_form, l_q=FLAGS.l_q, l_q_center=FLAGS.l_q_center, l_q_out=FLAGS.l_q_out, l_kappa=FLAGS.l_kappa, l_stage_form=FLAGS.l_stage_form, l_stage_reward=FLAGS.l_stage_reward, reset_box_mode=FLAGS.reset_box_mode, region_profile=FLAGS.region_profile, reset_requires_safe=FLAGS.reset_requires_safe, safe_alpha_hi=FLAGS.safe_alpha_hi, safe_beta=FLAGS.safe_beta, train_safe_h_min=FLAGS.train_safe_h_min, train_safe_h_max=FLAGS.train_safe_h_max, train_safe_alpha_lo=FLAGS.train_safe_alpha_lo, train_safe_alpha_hi=FLAGS.train_safe_alpha_hi, train_safe_beta=FLAGS.train_safe_beta, alpha_counts_as_invalid_dynamics=FLAGS.alpha_counts_as_invalid_dynamics, beta_counts_as_invalid_dynamics=FLAGS.beta_counts_as_invalid_dynamics, clip_relaxed_alpha_beta_to_terminate=FLAGS.clip_relaxed_alpha_beta_to_terminate)

def current_config_dict():
    config = {k: FLAGS[k].value for k in _CONFIG_EXPORT_KEYS}
    config['policy'] = POLICY_NAME
    config['h_theta_denom'] = float(env_v6.resolve_h_theta_denom(str(FLAGS.region_profile), float(FLAGS.h_theta_denom)))
    config['task_spec_version'] = TASK_SPEC_VERSION_V6
    config['env'] = 'F16-StabilizeAvoid'
    config['env_version'] = 'release'
    config['safety_estimator'] = 'sampled_qh'
    return config

def save_checkpoint(agent, path, policy_name):
    params = jax.device_get({'score_model': agent.score_model.params, 'critic_1': agent.critic_1.params, 'critic_2': agent.critic_2.params, 'target_critic_1': agent.target_critic_1.params, 'target_critic_2': agent.target_critic_2.params, 'safe_critic': agent.safe_critic.params, 'safe_target_critic': agent.safe_target_critic.params})
    if hasattr(agent, 'safe_value'):
        params['safe_value'] = agent.safe_value.params
        params['safe_target_value'] = agent.safe_target_value.params
    data = {'params': params, 'policy': policy_name, 'rng': jax.device_get(agent.rng), 'config': current_config_dict()}
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'  Checkpoint saved: {path} ({os.path.getsize(path) / 1000000.0:.1f} MB)')
_RESUME_ENV_STATE_KEYS = ['_x', '_prev_action', '_step_count', '_train_step', '_eval_mode', '_last_paper_sampler_metadata', '_last_reset_curriculum_active', '_last_reset_curriculum_fallback', '_last_reset_curriculum_reject_rate', '_last_reset_curriculum_target_frac', '_ep_reward', '_ep_task_cost', '_ep_unsafe_steps', '_ep_length', '_ep_goal_steps', '_ep_goal_streak', '_ep_max_goal_streak', '_ep_success', '_ep_success_step', '_ep_terminated', '_ep_crash_step', '_ep_crash_cause', '_ep_terminate_reason', '_ep_final_goal_distance', '_ep_min_altitude', '_ep_peak_h', '_ep_peak_h_components', '_ep_max_abs_beta', '_ep_max_abs_theta', '_ep_safe_steps', '_ep_safe_in_goal_steps', '_ep_init_goal', '_ep_init_safe', '_ep_init_h', '_ep_init_h_alpha', '_ep_init_h_beta', '_ep_init_alpha', '_ep_init_abs_beta', '_ep_init_pe_abs', 'episode_info']

def _serialize_env_state(env):
    state = {key: env.__dict__.get(key) for key in _RESUME_ENV_STATE_KEYS}
    state['_rng_state'] = env._rng.get_state()
    return state

def _restore_env_state(env, state):
    for key in _RESUME_ENV_STATE_KEYS:
        if key in state:
            env.__dict__[key] = state[key]
    rng_state = state.get('_rng_state')
    if rng_state is not None:
        env._rng = np.random.RandomState()
        env._rng.set_state(rng_state)

def save_resume_checkpoint(agent, replay_buffer, env, obs, done, step, episode_count, path, policy_name):
    data = {'format_version': 1, 'policy_name': policy_name, 'train_step': int(step), 'episode_count': int(episode_count), 'agent_state': jax.device_get(flax_serialization.to_state_dict(agent)), 'replay_buffer': replay_buffer.state_dict(), 'env_state': _serialize_env_state(env), 'obs': np.asarray(obs, dtype=np.float32), 'done': bool(done), 'config': current_config_dict()}
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'  Resume checkpoint saved: {path} ({os.path.getsize(path) / 1000000.0:.1f} MB)')

def load_resume_checkpoint(path, agent, replay_buffer, env):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    if 'agent_state' not in data or 'replay_buffer' not in data:
        raise ValueError(f'{path} is not a resumable checkpoint from the F16 trainer (missing agent_state or replay_buffer).')
    agent = flax_serialization.from_state_dict(agent, data['agent_state'])
    replay_buffer.load_state_dict(data['replay_buffer'])
    env_state = data.get('env_state')
    if env_state is not None:
        _restore_env_state(env, env_state)
    obs = np.asarray(data.get('obs', env._get_obs(env._x)), dtype=np.float32)
    done = bool(data.get('done', False))
    step = int(data.get('train_step', 0))
    episode_count = int(data.get('episode_count', 0))
    return (agent, replay_buffer, env, obs, done, step, episode_count)

class SimpleReplayBuffer:

    def __init__(self, obs_dim, act_dim, max_size=1000000):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.observations = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.actions = np.zeros((max_size, act_dim), dtype=np.float32)
        self.rewards = np.zeros(max_size, dtype=np.float32)
        self.costs = np.zeros(max_size, dtype=np.float32)
        self.masks = np.zeros(max_size, dtype=np.float32)
        self.next_observations = np.zeros((max_size, obs_dim), dtype=np.float32)
        self._rng = np.random.RandomState(0)

    def seed(self, seed):
        self._rng = np.random.RandomState(seed)

    def insert(self, data):
        i = self.ptr
        self.observations[i] = data['observations']
        self.actions[i] = data['actions']
        self.rewards[i] = data['rewards']
        self.costs[i] = data['costs']
        self.masks[i] = data['masks']
        self.next_observations[i] = data['next_observations']
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        idxs = self._rng.randint(0, self.size, size=batch_size)
        return {'observations': self.observations[idxs], 'actions': self.actions[idxs], 'rewards': self.rewards[idxs], 'costs': self.costs[idxs], 'masks': self.masks[idxs], 'next_observations': self.next_observations[idxs]}

    def state_dict(self):
        if self.size == 0:
            valid_observations = self.observations[:0].copy()
            valid_actions = self.actions[:0].copy()
            valid_rewards = self.rewards[:0].copy()
            valid_costs = self.costs[:0].copy()
            valid_masks = self.masks[:0].copy()
            valid_next_observations = self.next_observations[:0].copy()
        elif self.size < self.max_size:
            valid_observations = self.observations[:self.size].copy()
            valid_actions = self.actions[:self.size].copy()
            valid_rewards = self.rewards[:self.size].copy()
            valid_costs = self.costs[:self.size].copy()
            valid_masks = self.masks[:self.size].copy()
            valid_next_observations = self.next_observations[:self.size].copy()
        else:
            valid_slice = np.arange(self.size) + self.ptr
            valid_idxs = valid_slice % self.max_size
            valid_observations = self.observations[valid_idxs].copy()
            valid_actions = self.actions[valid_idxs].copy()
            valid_rewards = self.rewards[valid_idxs].copy()
            valid_costs = self.costs[valid_idxs].copy()
            valid_masks = self.masks[valid_idxs].copy()
            valid_next_observations = self.next_observations[valid_idxs].copy()
        return {'max_size': int(self.max_size), 'size': int(self.size), 'ptr': int(self.ptr), 'rng_state': self._rng.get_state(), 'observations': valid_observations, 'actions': valid_actions, 'rewards': valid_rewards, 'costs': valid_costs, 'masks': valid_masks, 'next_observations': valid_next_observations}

    def load_state_dict(self, state):
        size = int(state['size'])
        if size > self.max_size:
            raise ValueError(f'Resume buffer size {size} exceeds current max_size {self.max_size}.')
        self.ptr = 0
        self.size = 0
        self.observations.fill(0.0)
        self.actions.fill(0.0)
        self.rewards.fill(0.0)
        self.costs.fill(0.0)
        self.masks.fill(0.0)
        self.next_observations.fill(0.0)
        if size > 0:
            self.observations[:size] = state['observations']
            self.actions[:size] = state['actions']
            self.rewards[:size] = state['rewards']
            self.costs[:size] = state['costs']
            self.masks[:size] = state['masks']
            self.next_observations[:size] = state['next_observations']
        self.size = size
        self.ptr = size % self.max_size
        self._rng = np.random.RandomState()
        self._rng.set_state(state['rng_state'])

def select_update_metrics(update_info):
    keep_keys = ['actor_loss', 'critic_loss_1', 'critic_loss_2', 'safe_critic_loss', 'q1_mean', 'q1_min', 'q1_max', 'q2_mean', 'q2_min', 'q2_max', 'target_q_mean', 'target_q_std', 'target_q_mag', 'target_q_min', 'target_q_max', 'td_err_abs_mean', 'td_err_abs_p99', 'qc_mean', 'qc_min', 'qc_max', 'reward_coverage', 'indicator_given_safe', 'grad_qr_norm_raw', 'grad_qr_norm_clipped', 'grad_qh_norm', 'grad_cos_qr_qh_deadzone_mean', 'grad_cos_qr_qh_all_mean', 'qh_indicator_mean', 'qh_policy_mean', 'qh_gate_mean', 'qh_gate_candidate_count', 'qh_bootstrap_mean', 'qh_bootstrap_candidate_count', 'eps_target_norm', 'eps_pred_norm', 'eps_target_unsafe_norm_mean', 'eps_target_safe_norm_mean']
    return {f'train_diag/{k}': float(update_info[k]) for k in keep_keys if k in update_info}

def select_episode_metrics(ep_info):
    return {'train/episode_reward': float(ep_info['reward']), 'train/episode_task_cost': float(ep_info['task_cost']), 'train/episode_unsafe_step_count': float(ep_info['unsafe_step_count']), 'train/episode_length': float(ep_info['length']), 'train/terminated': float(ep_info['terminated']), 'train/max_goal_streak': float(ep_info['max_goal_streak']), 'train/final_goal_distance': float(ep_info['final_goal_distance']), 'train/peak_h': float(ep_info['peak_h']), 'train/h_alt_peak': float(ep_info['h_alt_peak']), 'train/h_alpha_peak': float(ep_info['h_alpha_peak']), 'train/h_beta_peak': float(ep_info['h_beta_peak']), 'train/h_theta_peak': float(ep_info['h_theta_peak']), 'train/h_pe_peak': float(ep_info['h_pe_peak']), 'train/h_p_peak': float(ep_info['h_p_peak']), 'train/min_altitude': float(ep_info['min_altitude']), 'train/max_abs_beta': float(ep_info['max_abs_beta']), 'train/max_abs_theta': float(ep_info['max_abs_theta']), 'train/curriculum_reject_rate': float(ep_info['curriculum_reject_rate']), 'train/init_h': float(ep_info['init_h']), 'train/init_h_alpha': float(ep_info['init_h_alpha']), 'train/init_h_beta': float(ep_info['init_h_beta']), 'train/init_alpha': float(ep_info['init_alpha']), 'train/init_abs_beta': float(ep_info['init_abs_beta']), 'train/init_pe_abs_mean': float(ep_info['init_pe_abs'])}

def drop_frac_metrics(metrics: Dict[str, float]) -> Dict[str, float]:
    return {key: value for key, value in metrics.items() if not key.rsplit('/', 1)[-1].endswith('_frac')}


def make_timing_accumulator():
    timing_acc = {'time/sample_action_total': 0.0, 'time/diffusion_collect': 0.0, 'time/env_step': 0.0, 'time/update_total': 0.0, 'time/checkpoint': 0.0}
    if FLAGS.fused_update:
        timing_acc['time/update_fused_actor'] = 0.0
        timing_acc['time/update_fused_critic_only'] = 0.0
        if FLAGS.fuse_next_action:
            timing_acc['time/update_and_next_sample'] = 0.0
    else:
        timing_acc.update({'time/update_qc': 0.0, 'time/update_vc': 0.0, 'time/update_q_reward': 0.0, 'time/update_actor': 0.0})
    return timing_acc

def write_local_metrics_jsonl(path: str, step: int, kind: str, metrics: Dict[str, float]):
    record = {'step': int(step), 'kind': kind}
    record.update(metrics)
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(record, sort_keys=True) + '\n')

def block_until_ready_tree(tree):
    for leaf in jax.tree_util.tree_leaves(tree):
        if hasattr(leaf, 'block_until_ready'):
            leaf.block_until_ready()

@jax.jit
def _update_and_next_action(agent, batch, next_obs):
    updated_agent, update_info = agent.update(batch)
    next_action, sampled_agent = updated_agent.sample_actions(next_obs)
    return (updated_agent, next_action, sampled_agent.rng, update_info)

def format_local_update_summary(step: int, metrics: Dict[str, float]) -> str:
    keys = ['train_diag/sps', 'time/sample_action_total', 'time/diffusion_collect', 'time/update_and_next_sample', 'time/env_step', 'time/update_total']
    if 'time/update_fused_actor' in metrics:
        keys.append('time/update_fused_actor')
    if 'time/update_fused_critic_only' in metrics:
        keys.append('time/update_fused_critic_only')
    else:
        keys.extend(['time/update_qc', 'time/update_vc', 'time/update_q_reward', 'time/update_actor'])
    parts = [f'[train_diag {step}]']
    for key in keys:
        if key in metrics:
            short_key = key.split('/')[-1]
            parts.append(f'{short_key}={float(metrics[key]):.4f}')
    return ' '.join(parts)

def _fmt_curriculum_range(lo0: float, hi0: float, lo1: float, hi1: float, fmt: str='.2f') -> str:
    if np.isnan(float(lo0)) and np.isnan(float(hi0)) and np.isnan(float(lo1)) and np.isnan(float(hi1)):
        return 'base'

    def _fmt(v):
        return 'base' if np.isnan(float(v)) else format(float(v), fmt)
    return f'[{_fmt(lo0)}, {_fmt(hi0)}] -> [{_fmt(lo1)}, {_fmt(hi1)}]'

def _fmt_curriculum_abs(max0: float, max1: float, fmt: str='.2f') -> str:
    if np.isnan(float(max0)) and np.isnan(float(max1)):
        return 'base'

    def _fmt(v):
        return 'base' if np.isnan(float(v)) else format(float(v), fmt)
    return f'{_fmt(max0)} -> {_fmt(max1)}'

def _flag_or_default(value: float, default_value: float) -> float:
    return float(default_value) if np.isnan(float(value)) else float(value)

def _assert_startup_float(name: str, actual: float, expected: float, atol: float=1e-09):
    actual_f = float(actual)
    expected_f = float(expected)
    if np.isnan(actual_f) and np.isnan(expected_f):
        return
    if not np.isclose(actual_f, expected_f, rtol=0.0, atol=atol):
        raise RuntimeError(f'[sanity] startup plumbing mismatch for {name}: expected {expected_f:g}, actual {actual_f:g}')

def _assert_startup_bool(name: str, actual: bool, expected: bool):
    if bool(actual) != bool(expected):
        raise RuntimeError(f'[sanity] startup plumbing mismatch for {name}: expected {bool(expected)}, actual {bool(actual)}')

def _assert_startup_str(name: str, actual: str, expected: str):
    if str(actual) != str(expected):
        raise RuntimeError(f'[sanity] startup plumbing mismatch for {name}: expected {str(expected)!r}, actual {str(actual)!r}')

def _curriculum_scalar_pair(start_value: float, end_value: float, base_value: float) -> Tuple[float, float] | None:
    return env_v6._resolve_optional_curriculum_pair(start_value, end_value, base_value)

def _fmt_effective_curriculum_range(lo_start: float, hi_start: float, lo_end: float, hi_end: float, base_lo: float, base_hi: float, fmt: str='.2f') -> str:
    lo_pair = _curriculum_scalar_pair(lo_start, lo_end, base_lo)
    hi_pair = _curriculum_scalar_pair(hi_start, hi_end, base_hi)
    if lo_pair is None and hi_pair is None:
        return 'base'
    lo_pair = lo_pair if lo_pair is not None else (float(base_lo), float(base_lo))
    hi_pair = hi_pair if hi_pair is not None else (float(base_hi), float(base_hi))

    def _fmt(v: float) -> str:
        return format(float(v), fmt)
    return f'[{_fmt(lo_pair[0])}, {_fmt(hi_pair[0])}] -> [{_fmt(lo_pair[1])}, {_fmt(hi_pair[1])}]'

def _assert_no_anneal_curriculum_pair_locked(name: str, start_value: float, end_value: float, base_value: float):
    pair = _curriculum_scalar_pair(start_value, end_value, base_value)
    if pair is None:
        return
    start_eff, end_eff = pair
    if not np.isclose(float(start_eff), float(end_eff), rtol=0.0, atol=1e-09):
        raise RuntimeError(f'[sanity] curriculum no-anneal mismatch for {name}: effective start={float(start_eff):g}, end={float(end_eff):g}. For anneal=[0,0], set *_END equal to the start value, clear inherited *_END env vars, or pass --fail_on_curriculum_no_anneal_mismatch=false intentionally.')

def _assert_no_anneal_value_locked(name: str, start_value: float, end_value: float):
    if not np.isclose(float(start_value), float(end_value), rtol=0.0, atol=1e-09):
        raise RuntimeError(f'[sanity] curriculum no-anneal mismatch for {name}: start={float(start_value):g}, end={float(end_value):g}. For anneal=[0,0], set *_END equal to the start value, clear inherited *_END env vars, or pass --fail_on_curriculum_no_anneal_mismatch=false intentionally.')

def _assert_startup_plumbing(env: F16StabilizeEnvV6):
    for name in ('region_profile', 'reset_box_mode', 'safety_gap_mode', 'obs_task_feats_mode'):
        _assert_startup_str(name, getattr(env, name), getattr(FLAGS, name))
    _assert_startup_bool('reset_requires_safe', env.reset_requires_safe, FLAGS.reset_requires_safe)
    geometry = env_v6.resolve_region_geometry(str(FLAGS.region_profile))
    expected_safe = {'safe_h_min': _flag_or_default(FLAGS.train_safe_h_min, geometry['safe_h_min']), 'safe_h_max': _flag_or_default(FLAGS.train_safe_h_max, geometry['safe_h_max']), 'safe_alpha_lo': _flag_or_default(FLAGS.train_safe_alpha_lo, env_v6._SAFE_ALPHA_LO), 'safe_alpha_hi': _flag_or_default(FLAGS.train_safe_alpha_hi, FLAGS.safe_alpha_hi), 'safe_beta': _flag_or_default(FLAGS.train_safe_beta, FLAGS.safe_beta)}
    print(f"[sanity] env.safe actual=h=[{env.safe_h_min:g},{env.safe_h_max:g}] alpha=[{env.safe_alpha_lo:g},{env.safe_alpha_hi:g}] beta={env.safe_beta:g}; expected=h=[{expected_safe['safe_h_min']:g},{expected_safe['safe_h_max']:g}] alpha=[{expected_safe['safe_alpha_lo']:g},{expected_safe['safe_alpha_hi']:g}] beta={expected_safe['safe_beta']:g}")
    for attr, expected in expected_safe.items():
        _assert_startup_float(attr, getattr(env, attr), expected)
    _assert_startup_bool('init_curriculum', env.init_curriculum, FLAGS.init_curriculum)
    if FLAGS.init_curriculum:
        _assert_startup_str('init_curriculum_mode', env.init_curriculum_mode, FLAGS.init_curriculum_mode)
        curriculum_checks = ['init_curriculum_frac', 'init_curriculum_frac_end', 'init_curriculum_h_min', 'init_curriculum_h_max', 'init_curriculum_h_min_end', 'init_curriculum_h_max_end', 'init_curriculum_pe_abs_min', 'init_curriculum_pe_abs_max', 'init_curriculum_pe_abs_min_end', 'init_curriculum_pe_abs_max_end', 'init_curriculum_theta_lo', 'init_curriculum_theta_hi', 'init_curriculum_theta_lo_end', 'init_curriculum_theta_hi_end', 'init_curriculum_alpha_lo', 'init_curriculum_alpha_hi', 'init_curriculum_alpha_lo_end', 'init_curriculum_alpha_hi_end', 'init_curriculum_beta_abs_max', 'init_curriculum_beta_abs_max_end', 'init_curriculum_p_abs_max', 'init_curriculum_p_abs_max_end', 'init_curriculum_q_abs_max', 'init_curriculum_q_abs_max_end', 'init_curriculum_r_abs_max', 'init_curriculum_r_abs_max_end', 'init_curriculum_allow_in_goal', 'init_curriculum_require_train_safe']
        if FLAGS.init_curriculum_mode == INIT_CURRICULUM_MODE_BOX_MIXTURE:
            curriculum_checks.extend(['init_curriculum_box2_frac', 'init_curriculum_box2_frac_end', 'init_curriculum_box2_h_min', 'init_curriculum_box2_h_max', 'init_curriculum_box2_h_min_end', 'init_curriculum_box2_h_max_end', 'init_curriculum_box2_theta_lo', 'init_curriculum_box2_theta_hi', 'init_curriculum_box2_theta_lo_end', 'init_curriculum_box2_theta_hi_end'])
        for name in curriculum_checks:
            expected = getattr(FLAGS, name)
            actual = getattr(env, name)
            if isinstance(expected, bool):
                _assert_startup_bool(name, actual, expected)
            else:
                _assert_startup_float(name, actual, expected)
        reset_box = env._task_reset_box()
        if FLAGS.init_curriculum_mode == INIT_CURRICULUM_MODE_BOX_MIXTURE:
            print(f"[sanity] curriculum.effective box1 h={_fmt_effective_curriculum_range(FLAGS.init_curriculum_h_min, FLAGS.init_curriculum_h_max, FLAGS.init_curriculum_h_min_end, FLAGS.init_curriculum_h_max_end, reset_box[env_v6.IDX_H, 0], reset_box[env_v6.IDX_H, 1], '.0f')} theta={_fmt_effective_curriculum_range(FLAGS.init_curriculum_theta_lo, FLAGS.init_curriculum_theta_hi, FLAGS.init_curriculum_theta_lo_end, FLAGS.init_curriculum_theta_hi_end, reset_box[env_v6.IDX_THETA, 0], reset_box[env_v6.IDX_THETA, 1], '.2f')}")
            print(f"[sanity] curriculum.effective box2 h={_fmt_effective_curriculum_range(FLAGS.init_curriculum_box2_h_min, FLAGS.init_curriculum_box2_h_max, FLAGS.init_curriculum_box2_h_min_end, FLAGS.init_curriculum_box2_h_max_end, reset_box[env_v6.IDX_H, 0], reset_box[env_v6.IDX_H, 1], '.0f')} theta={_fmt_effective_curriculum_range(FLAGS.init_curriculum_box2_theta_lo, FLAGS.init_curriculum_box2_theta_hi, FLAGS.init_curriculum_box2_theta_lo_end, FLAGS.init_curriculum_box2_theta_hi_end, reset_box[env_v6.IDX_THETA, 0], reset_box[env_v6.IDX_THETA, 1], '.2f')}")
        elif FLAGS.init_curriculum_mode == INIT_CURRICULUM_MODE_EASY_TO_HARD:
            print('[sanity] curriculum.easy_to_hard alpha=0 samples a benign trim/goal-approach box; alpha=1 widens to the active reset box.')
        else:
            print(f"[sanity] curriculum.effective box h={_fmt_effective_curriculum_range(FLAGS.init_curriculum_h_min, FLAGS.init_curriculum_h_max, FLAGS.init_curriculum_h_min_end, FLAGS.init_curriculum_h_max_end, reset_box[env_v6.IDX_H, 0], reset_box[env_v6.IDX_H, 1], '.0f')} theta={_fmt_effective_curriculum_range(FLAGS.init_curriculum_theta_lo, FLAGS.init_curriculum_theta_hi, FLAGS.init_curriculum_theta_lo_end, FLAGS.init_curriculum_theta_hi_end, reset_box[env_v6.IDX_THETA, 0], reset_box[env_v6.IDX_THETA, 1], '.2f')}")
        if FLAGS.fail_on_curriculum_no_anneal_mismatch and int(FLAGS.init_curriculum_anneal_start) == 0 and (int(FLAGS.init_curriculum_anneal_end) == 0):
            _assert_no_anneal_value_locked('init_curriculum_frac', FLAGS.init_curriculum_frac, FLAGS.init_curriculum_frac_end)
            _assert_no_anneal_curriculum_pair_locked('init_curriculum_h_min', FLAGS.init_curriculum_h_min, FLAGS.init_curriculum_h_min_end, reset_box[env_v6.IDX_H, 0])
            _assert_no_anneal_curriculum_pair_locked('init_curriculum_h_max', FLAGS.init_curriculum_h_max, FLAGS.init_curriculum_h_max_end, reset_box[env_v6.IDX_H, 1])
            _assert_no_anneal_curriculum_pair_locked('init_curriculum_theta_lo', FLAGS.init_curriculum_theta_lo, FLAGS.init_curriculum_theta_lo_end, reset_box[env_v6.IDX_THETA, 0])
            _assert_no_anneal_curriculum_pair_locked('init_curriculum_theta_hi', FLAGS.init_curriculum_theta_hi, FLAGS.init_curriculum_theta_hi_end, reset_box[env_v6.IDX_THETA, 1])
            _assert_no_anneal_curriculum_pair_locked('init_curriculum_pe_abs_min', FLAGS.init_curriculum_pe_abs_min, FLAGS.init_curriculum_pe_abs_min_end, 0.0)
            _assert_no_anneal_curriculum_pair_locked('init_curriculum_pe_abs_max', FLAGS.init_curriculum_pe_abs_max, FLAGS.init_curriculum_pe_abs_max_end, max(abs(float(reset_box[env_v6.IDX_PE, 0])), abs(float(reset_box[env_v6.IDX_PE, 1]))))
            _assert_no_anneal_curriculum_pair_locked('init_curriculum_alpha_lo', FLAGS.init_curriculum_alpha_lo, FLAGS.init_curriculum_alpha_lo_end, reset_box[env_v6.IDX_ALPHA, 0])
            _assert_no_anneal_curriculum_pair_locked('init_curriculum_alpha_hi', FLAGS.init_curriculum_alpha_hi, FLAGS.init_curriculum_alpha_hi_end, reset_box[env_v6.IDX_ALPHA, 1])
            _assert_no_anneal_curriculum_pair_locked('init_curriculum_beta_abs_max', FLAGS.init_curriculum_beta_abs_max, FLAGS.init_curriculum_beta_abs_max_end, max(abs(float(reset_box[env_v6.IDX_BETA, 0])), abs(float(reset_box[env_v6.IDX_BETA, 1]))))
            _assert_no_anneal_curriculum_pair_locked('init_curriculum_p_abs_max', FLAGS.init_curriculum_p_abs_max, FLAGS.init_curriculum_p_abs_max_end, max(abs(float(reset_box[env_v6.IDX_P, 0])), abs(float(reset_box[env_v6.IDX_P, 1]))))
            _assert_no_anneal_curriculum_pair_locked('init_curriculum_q_abs_max', FLAGS.init_curriculum_q_abs_max, FLAGS.init_curriculum_q_abs_max_end, max(abs(float(reset_box[env_v6.IDX_Q, 0])), abs(float(reset_box[env_v6.IDX_Q, 1]))))
            _assert_no_anneal_curriculum_pair_locked('init_curriculum_r_abs_max', FLAGS.init_curriculum_r_abs_max, FLAGS.init_curriculum_r_abs_max_end, max(abs(float(reset_box[env_v6.IDX_R, 0])), abs(float(reset_box[env_v6.IDX_R, 1]))))
            if FLAGS.init_curriculum_mode == INIT_CURRICULUM_MODE_BOX_MIXTURE:
                _assert_no_anneal_value_locked('init_curriculum_box2_frac', FLAGS.init_curriculum_box2_frac, FLAGS.init_curriculum_box2_frac_end)
                _assert_no_anneal_curriculum_pair_locked('init_curriculum_box2_h_min', FLAGS.init_curriculum_box2_h_min, FLAGS.init_curriculum_box2_h_min_end, reset_box[env_v6.IDX_H, 0])
                _assert_no_anneal_curriculum_pair_locked('init_curriculum_box2_h_max', FLAGS.init_curriculum_box2_h_max, FLAGS.init_curriculum_box2_h_max_end, reset_box[env_v6.IDX_H, 1])
                _assert_no_anneal_curriculum_pair_locked('init_curriculum_box2_theta_lo', FLAGS.init_curriculum_box2_theta_lo, FLAGS.init_curriculum_box2_theta_lo_end, reset_box[env_v6.IDX_THETA, 0])
                _assert_no_anneal_curriculum_pair_locked('init_curriculum_box2_theta_hi', FLAGS.init_curriculum_box2_theta_hi, FLAGS.init_curriculum_box2_theta_hi_end, reset_box[env_v6.IDX_THETA, 1])

def main(_):
    from ssm.agents.ssm_agent import SSMOnlineAgent
    if FLAGS.fuse_next_action and (not FLAGS.fused_update):
        raise ValueError('fuse_next_action requires fused_update=True')
    run_name_parts = ['f16_stabilize', 'efppo', FLAGS.safety_gap_mode, f'h{FLAGS.h_form}', f'th{FLAGS.h_theta_denom:g}', f'l{FLAGS.l_form}', f'q{FLAGS.l_q:g}', f'qc{FLAGS.l_q_center:g}', f'qo{FLAGS.l_q_out:g}', f'k{FLAGS.l_kappa:g}', f'ls{FLAGS.l_stage_form}', f'rs{FLAGS.l_stage_reward:g}', f'rb{FLAGS.reset_box_mode}', f'rp{FLAGS.region_profile}', f'rj{int(bool(FLAGS.reset_requires_safe))}', f'sah{FLAGS.safe_alpha_hi:g}', f'sb{FLAGS.safe_beta:g}', f'Mq{FLAGS.M_q}', f'b{FLAGS.beta_safety}', f'hm{FLAGS.hard_margin}', f'ep{FLAGS.max_episode_steps}', f's{FLAGS.seed}']
    run_name = FLAGS.run_name or '_'.join(run_name_parts)
    if FLAGS.wandb:
        import wandb
        wandb_kwargs = dict(project=FLAGS.project_name, entity=FLAGS.entity, name=run_name, config=current_config_dict())
        wandb.init(**wandb_kwargs)
    checkpoint_interval = int(FLAGS.checkpoint_interval)
    if checkpoint_interval <= 0:
        checkpoint_interval = int(FLAGS.checkpoint_interval)
    resume_checkpoint_interval = int(FLAGS.resume_checkpoint_interval)
    if resume_checkpoint_interval <= 0:
        resume_checkpoint_interval = checkpoint_interval
    env = make_env_v6(FLAGS.seed)
    if FLAGS.seed_warmup_actions:
        env.action_space.seed(FLAGS.seed)
    actor_hidden = tuple((int(x) for x in FLAGS.actor_hidden.split(',')))
    critic_hidden = tuple((int(x) for x in FLAGS.critic_hidden.split(',')))
    agent_kwargs = dict(seed=FLAGS.seed, observation_space=env.observation_space, action_space=env.action_space, actor_hidden_dims=actor_hidden, critic_hidden_dims=critic_hidden, M_q=FLAGS.M_q, alpha_r=FLAGS.alpha_r, beta_safety=FLAGS.beta_safety, delta=FLAGS.delta, hard_margin=FLAGS.hard_margin, discount=FLAGS.discount, discount_h=FLAGS.discount_h, tau=FLAGS.tau, T=FLAGS.T, beta_schedule=FLAGS.beta_schedule, cost_critic_hyperparam=FLAGS.cost_critic_hyperparam, safety_estimator='sampled_qh', vc_mode=FLAGS.vc_mode, vc_policy_samples=FLAGS.vc_policy_samples, vc_proposal_samples=FLAGS.vc_proposal_samples, vc_proposal_std=FLAGS.vc_proposal_std, pure_qsm=FLAGS.pure_qsm, ddpm_temperature=FLAGS.ddpm_temperature, fuse_clip_min=FLAGS.fuse_clip_min, fuse_clip_max=FLAGS.fuse_clip_max, huber_delta=FLAGS.huber_delta)
    agent_kwargs.update(stage_mode=FLAGS.qh_stage_mode, qh_candidate_source=FLAGS.qh_candidate_source, qh_min_k_samples=FLAGS.qh_min_k_samples, qh_min_k_sigma=FLAGS.qh_min_k_sigma)
    agent = SSMOnlineAgent.create(**agent_kwargs)
    replay_buffer = SimpleReplayBuffer(env.observation_space.shape[0], env.action_space.shape[0], max_size=FLAGS.max_steps + 10000)
    replay_buffer.seed(FLAGS.seed)
    obs = env.reset()
    done = False
    episode_count = 0
    start_step = 0
    if FLAGS.resume_from:
        agent, replay_buffer, env, obs, done, start_step, episode_count = load_resume_checkpoint(FLAGS.resume_from, agent, replay_buffer, env)
        if done:
            obs = env.reset()
            done = False
        print(f'Resumed from {FLAGS.resume_from}: step={start_step} episodes={episode_count} buffer_size={replay_buffer.size}')
    env.set_train_mode()
    env.set_train_step(start_step)
    _assert_startup_plumbing(env)
    print(f'Env: F16-StabilizeAvoid-v6, obs={env.observation_space.shape}, act={env.action_space.shape}')
    print(f'  dt={env.dt}, max_steps={env.max_episode_steps}, debug_dwell={env.goal_dwell_steps}, goal=[{env.goal_h_min:.0f}, {env.goal_h_max:.0f}], gap_mode={env.safety_gap_mode}, region_profile={env.region_profile}, reset_requires_safe={env.reset_requires_safe}')
    print(f'  termination_penalty={abs(float(FLAGS.termination_penalty)):g} (true crash transitions only)')
    print(f'  train_safety h=[{env.safe_h_min:g}, {env.safe_h_max:g}] alpha=[{env.safe_alpha_lo:g}, {env.safe_alpha_hi:g}] beta_abs={env.safe_beta:g}')
    if FLAGS.init_curriculum and FLAGS.init_curriculum_mode == INIT_CURRICULUM_MODE_THETA_TAIL_PROXY:
        print(f'  curriculum={FLAGS.init_curriculum} mode={FLAGS.init_curriculum_mode} frac={FLAGS.init_curriculum_frac:.2f}->{FLAGS.init_curriculum_frac_end:.2f} anneal=[{FLAGS.init_curriculum_anneal_start}, {FLAGS.init_curriculum_anneal_end}] proxy=nominal-fixed(theta/H tails, alpha in [-0.02, 0.05]) allow_in_goal={FLAGS.init_curriculum_allow_in_goal}')
    elif FLAGS.init_curriculum and FLAGS.init_curriculum_mode == INIT_CURRICULUM_MODE_BOX_MIXTURE:
        base0 = max(0.0, 1.0 - FLAGS.init_curriculum_frac - FLAGS.init_curriculum_box2_frac)
        base1 = max(0.0, 1.0 - FLAGS.init_curriculum_frac_end - FLAGS.init_curriculum_box2_frac_end)
        print(f'  curriculum={FLAGS.init_curriculum} mode={FLAGS.init_curriculum_mode} base_frac={base0:.2f}->{base1:.2f} box1_frac={FLAGS.init_curriculum_frac:.2f}->{FLAGS.init_curriculum_frac_end:.2f} box2_frac={FLAGS.init_curriculum_box2_frac:.2f}->{FLAGS.init_curriculum_box2_frac_end:.2f} anneal=[{FLAGS.init_curriculum_anneal_start}, {FLAGS.init_curriculum_anneal_end}] allow_in_goal={FLAGS.init_curriculum_allow_in_goal}')
        print(f"  box1 h={_fmt_curriculum_range(FLAGS.init_curriculum_h_min, FLAGS.init_curriculum_h_max, FLAGS.init_curriculum_h_min_end, FLAGS.init_curriculum_h_max_end, '.0f')} theta={_fmt_curriculum_range(FLAGS.init_curriculum_theta_lo, FLAGS.init_curriculum_theta_hi, FLAGS.init_curriculum_theta_lo_end, FLAGS.init_curriculum_theta_hi_end, '.2f')}")
        print(f"  box2 h={_fmt_curriculum_range(FLAGS.init_curriculum_box2_h_min, FLAGS.init_curriculum_box2_h_max, FLAGS.init_curriculum_box2_h_min_end, FLAGS.init_curriculum_box2_h_max_end, '.0f')} theta={_fmt_curriculum_range(FLAGS.init_curriculum_box2_theta_lo, FLAGS.init_curriculum_box2_theta_hi, FLAGS.init_curriculum_box2_theta_lo_end, FLAGS.init_curriculum_box2_theta_hi_end, '.2f')}")
    elif FLAGS.init_curriculum and FLAGS.init_curriculum_mode == INIT_CURRICULUM_MODE_EASY_TO_HARD:
        print(f'  curriculum={FLAGS.init_curriculum} mode={FLAGS.init_curriculum_mode} frac={FLAGS.init_curriculum_frac:.2f}->{FLAGS.init_curriculum_frac_end:.2f} anneal=[{FLAGS.init_curriculum_anneal_start}, {FLAGS.init_curriculum_anneal_end}] schedule=benign trim/goal-approach box -> active reset box allow_in_goal={FLAGS.init_curriculum_allow_in_goal}')
    else:
        print(f"  curriculum={FLAGS.init_curriculum} mode={FLAGS.init_curriculum_mode} frac={FLAGS.init_curriculum_frac:.2f}->{FLAGS.init_curriculum_frac_end:.2f} anneal=[{FLAGS.init_curriculum_anneal_start}, {FLAGS.init_curriculum_anneal_end}] h={_fmt_curriculum_range(FLAGS.init_curriculum_h_min, FLAGS.init_curriculum_h_max, FLAGS.init_curriculum_h_min_end, FLAGS.init_curriculum_h_max_end, '.0f')} PE_abs={_fmt_curriculum_range(FLAGS.init_curriculum_pe_abs_min, FLAGS.init_curriculum_pe_abs_max, FLAGS.init_curriculum_pe_abs_min_end, FLAGS.init_curriculum_pe_abs_max_end, '.0f')} theta={_fmt_curriculum_range(FLAGS.init_curriculum_theta_lo, FLAGS.init_curriculum_theta_hi, FLAGS.init_curriculum_theta_lo_end, FLAGS.init_curriculum_theta_hi_end, '.2f')} allow_in_goal={FLAGS.init_curriculum_allow_in_goal}")
    if FLAGS.init_curriculum and FLAGS.init_curriculum_mode == INIT_CURRICULUM_MODE_AB_BOUNDARY:
        print(f'  ab_boundary kind={FLAGS.init_curriculum_ab_kind} joint_frac={FLAGS.init_curriculum_ab_joint_frac:.2f} alpha_h=[{FLAGS.init_curriculum_ab_alpha_h_lo:.2f}, {FLAGS.init_curriculum_ab_alpha_h_hi:.2f}] -> [{FLAGS.init_curriculum_ab_alpha_h_lo_end:.2f}, {FLAGS.init_curriculum_ab_alpha_h_hi_end:.2f}] beta_h=[{FLAGS.init_curriculum_ab_beta_h_lo:.2f}, {FLAGS.init_curriculum_ab_beta_h_hi:.2f}] -> [{FLAGS.init_curriculum_ab_beta_h_lo_end:.2f}, {FLAGS.init_curriculum_ab_beta_h_hi_end:.2f}]')
    if FLAGS.init_curriculum and FLAGS.init_curriculum_mode in (INIT_CURRICULUM_MODE_BOX, INIT_CURRICULUM_MODE_BOX_MIXTURE):
        print(f"  box_overrides alpha={_fmt_curriculum_range(FLAGS.init_curriculum_alpha_lo, FLAGS.init_curriculum_alpha_hi, FLAGS.init_curriculum_alpha_lo_end, FLAGS.init_curriculum_alpha_hi_end, '.3f')} beta_abs={_fmt_curriculum_abs(FLAGS.init_curriculum_beta_abs_max, FLAGS.init_curriculum_beta_abs_max_end, '.3f')} P_abs={_fmt_curriculum_abs(FLAGS.init_curriculum_p_abs_max, FLAGS.init_curriculum_p_abs_max_end, '.3f')} Q_abs={_fmt_curriculum_abs(FLAGS.init_curriculum_q_abs_max, FLAGS.init_curriculum_q_abs_max_end, '.3f')} R_abs={_fmt_curriculum_abs(FLAGS.init_curriculum_r_abs_max, FLAGS.init_curriculum_r_abs_max_end, '.3f')}")
    timing_acc = make_timing_accumulator()
    timing_count = 0
    wall_start = time.time()
    ckpt_dir = f'checkpoints/{run_name}'
    log_dir = f'logs/{run_name}'
    os.makedirs(log_dir, exist_ok=True)
    local_metrics_path = os.path.join(log_dir, 'local_metrics.jsonl')
    if not FLAGS.wandb:
        print(f'  local_metrics={local_metrics_path}')
    if start_step >= FLAGS.max_steps:
        print(f'Resume step {start_step} already reached max_steps={FLAGS.max_steps}; nothing to do.')
        if FLAGS.wandb:
            import wandb
            wandb.finish()
        return
    pending_action = pending_rng = None
    progress = range(start_step + 1, FLAGS.max_steps + 1)
    for step in tqdm(progress, smoothing=0.1, disable=not FLAGS.tqdm_bar, initial=start_step, total=FLAGS.max_steps):
        env.set_train_step(step)
        sample_t0 = time.perf_counter()
        if step < FLAGS.start_training:
            action = env.action_space.sample()
        else:
            diff_collect_t0 = time.perf_counter()
            if pending_action is None:
                action, agent = agent.sample_actions(obs)
            else:
                action = pending_action
                agent = agent.replace(rng=pending_rng)
                pending_action = pending_rng = None
            if FLAGS.sync_timing:
                block_until_ready_tree(action)
            timing_acc['time/diffusion_collect'] += time.perf_counter() - diff_collect_t0
            action = np.asarray(action)
        timing_acc['time/sample_action_total'] += time.perf_counter() - sample_t0
        env_t0 = time.perf_counter()
        next_obs, reward, h_val, _, done, info = env.step(action)
        timing_acc['time/env_step'] += time.perf_counter() - env_t0
        timing_count += 1
        terminated = bool(info.get('terminated', False))
        mask = 0.0 if terminated else 1.0
        reward_for_replay = float(reward)
        if terminated and FLAGS.termination_penalty > 0.0:
            reward_for_replay -= abs(float(FLAGS.termination_penalty))
        replay_buffer.insert({'observations': obs, 'actions': action, 'rewards': reward_for_replay, 'costs': h_val, 'masks': mask, 'next_observations': next_obs})
        obs = next_obs
        if done:
            episode_count += 1
            ep_info = env.episode_info
            ep_log = select_episode_metrics(ep_info)
            ep_log['train/episodes'] = episode_count
            ep_log = drop_frac_metrics(ep_log)
            if FLAGS.wandb:
                import wandb
                wandb.log(compact_episode_log(ep_info, episode_count=episode_count, terminated=ep_info.get('terminated', False)), step=step)
            write_local_metrics_jsonl(local_metrics_path, step, 'episode', ep_log)
            obs = env.reset()
            done = False
        if step >= FLAGS.start_training:
            batch = replay_buffer.sample(FLAGS.batch_size)
            grad_step = step - FLAGS.start_training
            should_update_actor = FLAGS.actor_delay <= 1 or grad_step % FLAGS.actor_delay == 0
            update_t0 = time.perf_counter()
            if FLAGS.fused_update:
                fused_t0 = time.perf_counter()
                if should_update_actor:
                    prefetch_next = FLAGS.fuse_next_action and step < FLAGS.max_steps and (step % FLAGS.checkpoint_interval != 0) and (step % checkpoint_interval != 0) and (not FLAGS.save_resume_checkpoint or step % resume_checkpoint_interval != 0)
                    if prefetch_next:
                        agent, pending_action, pending_rng, update_info = _update_and_next_action(agent, batch, obs)
                        if FLAGS.sync_timing:
                            block_until_ready_tree((agent, pending_action, pending_rng, update_info))
                        timing_acc['time/update_and_next_sample'] += time.perf_counter() - fused_t0
                    else:
                        agent, update_info = agent.update(batch)
                        if FLAGS.sync_timing:
                            block_until_ready_tree(update_info)
                        timing_acc['time/update_fused_actor'] += time.perf_counter() - fused_t0
                else:
                    agent, update_info = agent.update_critic_only(batch)
                    if FLAGS.sync_timing:
                        block_until_ready_tree(update_info)
                    timing_acc['time/update_fused_critic_only'] += time.perf_counter() - fused_t0
            else:
                qc_t0 = time.perf_counter()
                agent, qc_info = agent.update_qc(batch)
                if FLAGS.sync_timing:
                    block_until_ready_tree(qc_info)
                timing_acc['time/update_qc'] += time.perf_counter() - qc_t0
                vc_t0 = time.perf_counter()
                agent, vc_info = agent.update_vc(batch)
                if FLAGS.sync_timing:
                    block_until_ready_tree(vc_info)
                timing_acc['time/update_vc'] += time.perf_counter() - vc_t0
                q_t0 = time.perf_counter()
                agent, q_info = agent.update_q(batch)
                if FLAGS.sync_timing:
                    block_until_ready_tree(q_info)
                timing_acc['time/update_q_reward'] += time.perf_counter() - q_t0
                update_info = {**qc_info, **vc_info, **q_info}
                if should_update_actor:
                    actor_t0 = time.perf_counter()
                    agent, actor_info = agent.update_actor(batch)
                    if FLAGS.sync_timing:
                        block_until_ready_tree(actor_info)
                    timing_acc['time/update_actor'] += time.perf_counter() - actor_t0
                    update_info.update(actor_info)
            timing_acc['time/update_total'] += time.perf_counter() - update_t0
            if step % FLAGS.log_interval == 0:
                log_dict = select_update_metrics(update_info)
                log_dict['train_diag/step'] = step
                log_dict['train_diag/sps'] = step / max(time.time() - wall_start, 1e-06)
                log_dict['train_diag/fused_update'] = float(FLAGS.fused_update)
                log_dict['train_diag/sync_timing'] = float(FLAGS.sync_timing)
                if timing_count > 0:
                    for key, total in timing_acc.items():
                        log_dict[key] = total / timing_count
                log_dict = drop_frac_metrics(log_dict)
                if FLAGS.wandb:
                    import wandb
                    wandb.log(compact_update_log(update_info, sps=step / max(time.time() - wall_start, 1e-06)), step=step)
                write_local_metrics_jsonl(local_metrics_path, step, 'train_diag', log_dict)
                if not FLAGS.wandb:
                    print(format_local_update_summary(step, log_dict))
                timing_acc = make_timing_accumulator()
                timing_count = 0
        if step % checkpoint_interval == 0 or step == FLAGS.max_steps:
            save_checkpoint(agent, f'{ckpt_dir}/step_{step}.pkl', POLICY_NAME)
        if FLAGS.save_resume_checkpoint and (step % resume_checkpoint_interval == 0 or step == FLAGS.max_steps):
            save_resume_checkpoint(agent, replay_buffer, env, obs, done, step, episode_count, f'{ckpt_dir}/step_{step}_resume.pkl', POLICY_NAME)
    print('Training complete.')
    if FLAGS.wandb:
        import wandb
        wandb.finish()
if __name__ == '__main__':
    app.run(main)
