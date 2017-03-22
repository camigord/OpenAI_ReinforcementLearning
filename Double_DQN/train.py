import gym
import numpy as np
import tensorflow as tf
import random
import sys
import cv2
import os
import time
from tqdm import tqdm
import threading
import time
from agent import Agent
import commonOps as cops

conf_parameters = {
    'num_episodes': 100000,       # Training episodes

    'eval_freq': 1000,            # How often do we evaluate the model
    'eval_steps': 20,             # Evaluation steps

    # Input size
    'screen_width': 84,
    'screen_height': 84,
    'history_length': 4,
    'pool_frame_size': 1,

    'memory_size': 1000000,       # Replay memory size
    'batch_size': 32,             # Number of training cases over which SGD update is computed
    'gamma': 0.99,                # Discount factor
    'learning_rate': 0.00025,     # Learning rate

    'random_start': 30,           # Maximum number of 'do nothing' actions at the start of an episode

    # Exploration parameters
    'ep_min': 0.1,                 # Final exploration
    'ep_start': 1.0,               # Initial exploration
    'exploration_steps': 250000,   # Final exploration frame

    'target_q_update_step': 10000, # Target network update frequency
    'log_online_summary_rate': 100,
    'steps_before_training': 12500,
    'save_rate': 1000,
    'update_summary_rate': 50000,
    'sync_rate': 2500,

    # Clip rewards
    'min_reward': -1.0,
    'max_reward': 1.0,

    # How many times should the same action be taken
    'action_repeat': 1,

    'checkpoint_dir': 'Models/',
    'log_dir': 'Logs/',
    'device': '/gpu:0'
}

if not os.path.exists(conf_parameters['checkpoint_dir']):
    os.makedirs(conf_parameters['checkpoint_dir'])

if not os.path.exists(conf_parameters['log_dir']):
    os.makedirs(conf_parameters['log_dir'])

def test_run(agent, env, n):
    agent.testing(True)
    score_list = []
    for episode in range(n):
        screen, r, terminal, score = env.reset(), 0, False, 0
        while not terminal:
            action = agent.step(screen, r)
            screen, r, terminal, _ = env.step(action)
            score += r
        agent.done()
        score_list.append(score)
    agent.testing(False)
    return score_list

# Define environment
env = gym.make('Breakout-v0')
num_actions = env.action_space.n

sess_config = tf.ConfigProto()
sess_config.allow_soft_placement = True
sess_config.gpu_options.allow_growth = True
sess_config.log_device_placement = False

with tf.Session(config=sess_config) as sess:
    agent = Agent(conf_parameters, sess, num_actions)
    saver = tf.train.Saver(max_to_keep=20)

    sess.run(tf.initialize_all_variables())

    for episode in tqdm(range(conf_parameters['num_episodes']),ncols=80,initial=0):
        screen, r, terminal, score = env.reset(), 0, False, 0

        ep_begin_t = time.time()
        ep_begin_step_count = agent.step_count
        while not terminal:
            action = agent.step(screen, r)
            screen, r, terminal, _ = env.step(action)
            score += r
        agent.done()
        ep_duration = time.time() - ep_begin_t

        is_final_episode = conf_parameters['num_episodes'] == episode
        # online Summary
        if episode % conf_parameters['log_online_summary_rate'] == 0 or is_final_episode:
            episode_online_summary = tf.Summary(
                value=[
                    tf.Summary.Value(
                        tag="online/epsilon",
                        simple_value=agent.epsilon()),
                    tf.Summary.Value(
                        tag="score/online",
                        simple_value=score),
                    tf.Summary.Value(
                        tag="online/global_step",
                        simple_value=agent.step_count),
                    tf.Summary.Value(
                        tag="online/step_duration",
                        simple_value=ep_duration/(agent.step_count - ep_begin_step_count)),
                    tf.Summary.Value(
                        tag="online/ep_duration_seconds",
                        simple_value=ep_duration)])
            agent.summary_writter.add_summary(episode_online_summary, episode)

        # save
        if (episode % conf_parameters['save_rate'] == 0 and episode != 0) or is_final_episode:
            saver.save(sess,os.path.join(conf_parameters['checkpoint_dir'],'model'), global_step=episode)

        # performance summary
        if episode % conf_parameters['eval_freq'] == 0 or is_final_episode:
            score_list = test_run(agent, env, conf_parameters['eval_steps'])
            performance_summary = tf.Summary(
                value=[
                    tf.Summary.Value(
                        tag="score/average",simple_value=sum(score_list)/len(score_list)),
                    tf.Summary.Value(
                        tag="score/max",simple_value=max(score_list)),
                    tf.Summary.Value(
                        tag="score/min",simple_value=min(score_list)),
                ])
            agent.summary_writter.add_summary(performance_summary, episode)
