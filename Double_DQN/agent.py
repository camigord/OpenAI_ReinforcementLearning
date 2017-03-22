from memory import ReplayMemory
import tensorflow as tf
import numpy as np
import threading
import random
import cv2
import commonOps as cops

class Agent():

    def __init__(self, config, session, num_actions):
        self.config = config
        self.sess = session
        self.num_actions = num_actions

        self.gamma = config['gamma']
        self.learning_rate = config['learning_rate']

        self.exp_replay = ReplayMemory(self.config)
        self.game_state = np.zeros((1, config['screen_width'], config['screen_height'], config['history_length']), dtype=np.uint8)

        self.update_thread = threading.Thread(target=lambda: 0)
        self.update_thread.start()

        self.step_count = 0
        self.episode = 0
        self.isTesting = False

        self.reset_game()
        self.timeout_option = tf.RunOptions(timeout_in_ms=5000)

        # build the net
        with tf.device(config['device']):
            # Create all variables
            self.state_ph = tf.placeholder(tf.float32, [None, config['screen_width'], config['screen_height'], config['history_length']], name='state_ph')
            self.stateT_ph = tf.placeholder(tf.float32, [None, config['screen_width'], config['screen_height'], config['history_length']], name='stateT_ph')
            self.action_ph = tf.placeholder(tf.int64, [None], name='action_ph')
            self.reward_ph = tf.placeholder(tf.float32, [None], name='reward_ph')
            self.terminal_ph = tf.placeholder(tf.float32, [None], name='terminal_ph')

            # Define training network
            with tf.variable_scope('Q') as scope:
                self.Q = self.Q_network(self.state_ph, config, 'Normal')
                # *** Double Q-Learning ***
                scope.reuse_variables()
                self.DoubleQT = self.Q_network(self.stateT_ph, config, 'DoubleQ')
            # Define Target network
            with tf.variable_scope('QT'):
                self.QT = self.Q_network(self.stateT_ph, config, 'Target')

            # Define training operation
            self.train_op = self.train_op(self.Q, self.QT, self.action_ph, self.reward_ph, self.terminal_ph, config, 'Normal')

            # Define operation to copy parameteres from training to target net.
            with tf.variable_scope('Copy_parameters'):
                self.sync_QT_op = []
                for W_pair in zip(tf.get_collection('Target_weights'),tf.get_collection('Normal_weights')):
                    self.sync_QT_op.append(W_pair[0].assign(W_pair[1]))

            # Define the summary ops
            self.Q_summary_op = tf.merge_summary(tf.get_collection('Normal_summaries'))

        self.summary_writter = tf.train.SummaryWriter(config['log_dir'], self.sess.graph, flush_secs=20)

    def update(self):
        state_batch, action_batch, reward_batch, next_state_batch, terminal_batch, _ = self.exp_replay.sample_transition_batch()

        feed_dict={self.state_ph: state_batch,
                    self.stateT_ph: next_state_batch,
                    self.action_ph: action_batch,
                    self.reward_ph: reward_batch,
                    self.terminal_ph: terminal_batch}
        if self.step_count % self.config['update_summary_rate'] == 0:
            _, Q_summary_str = self.sess.run([self.train_op, self.Q_summary_op], feed_dict, options=self.timeout_option)
            self.summary_writter.add_summary(Q_summary_str, self.step_count)
        else:
            _ = self.sess.run(self.train_op, feed_dict, options=self.timeout_option)

        if self.step_count % self.config['sync_rate'] == 0:
            self.sess.run(self.sync_QT_op)

    def Q_network(self, input_state, config, Collection=None):
        conv_stack_shape=[(32,8,4),
                    (64,4,2),
                    (64,3,1)]

        head = tf.div(input_state,256., name='normalized_input')
        head = cops.conv_stack(head, conv_stack_shape, Collection)
        head = cops.flatten(head)
        head = cops.add_relu_layer(head, size=512, Collection=Collection)
        Q = cops.add_linear_layer(head, self.num_actions, Collection, layer_name="Q")

        return Q

    def train_op(self, Q, QT, action, reward, terminal, config, Collection):
        with tf.name_scope('Loss'):
            action_one_hot = tf.one_hot(action, self.num_actions, 1., 0., name='action_one_hot')
            acted_Q = tf.reduce_sum(Q * action_one_hot, reduction_indices=1, name='DQN_acted')

            # *** Double Q-Learning ***
            target_action = tf.argmax(self.DoubleQT, dimension=1)
            target_action_one_hot = tf.one_hot(target_action, self.num_actions, 1., 0., name='target_action_one_hot')
            DoubleQT_acted = tf.reduce_sum(self.QT * target_action_one_hot, reduction_indices=1, name='DoubleQT')
            Y = reward + self.gamma * DoubleQT_acted * (1 - terminal)
            # *** Double Q-Learning ***
            Y = tf.stop_gradient(Y)

            loss_batch = cops.clipped_l2(Y, acted_Q)
            loss = tf.reduce_sum(loss_batch, name='loss')

            tf.scalar_summary('losses/loss', loss, collections=[Collection + '_summaries'])
            tf.scalar_summary('losses/loss_0', loss_batch[0],collections=[Collection + '_summaries'])
            tf.scalar_summary('losses/loss_max', tf.reduce_max(loss_batch),collections=[Collection + '_summaries'])
            tf.scalar_summary('main/Y_0', Y[0], collections=[Collection + '_summaries'])
            tf.scalar_summary('main/Y_max', tf.reduce_max(Y), collections=[Collection + '_summaries'])
            tf.scalar_summary('main/acted_Q_0', acted_Q[0], collections=[Collection + '_summaries'])
            tf.scalar_summary('main/acted_Q_max', tf.reduce_max(acted_Q), collections=[Collection + '_summaries'])
            tf.scalar_summary('main/reward_max', tf.reduce_max(reward), collections=[Collection + '_summaries'])

        train_op, grads = cops.graves_rmsprop_optimizer(loss, self.learning_rate, 0.95, 0.01, 1)

        return train_op

    def testing(self, t=True):
        self.isTesting = t

    def reset_game(self):
        self.episode_begining = True
        self.game_state.fill(0)

    def epsilon(self):
        if self.step_count < self.config['exploration_steps']:
            return self.config['ep_start'] - ((self.config['ep_start'] - self.config['ep_min']) / self.config['exploration_steps']) * self.step_count
        else:
            return self.config['ep_min']

    def e_greedy_action(self, epsilon):
        if np.random.uniform() < epsilon:
            action = random.randint(0, self.num_actions - 1)
        else:
            action = np.argmax(self.sess.run(self.Q, feed_dict={self.state_ph: self.game_state})[0])
        return action

    def done(self):
        if not self.isTesting:
            self.exp_replay.add(self.game_state[:, :, :, -1],self.game_action, self.game_reward, True)
        self.reset_game()

    def observe(self, x, r):
        self.game_reward = r
        x_ = cv2.resize(x, (self.config['screen_width'], self.config['screen_height']))
        x_ = cv2.cvtColor(x_, cv2.COLOR_RGB2GRAY)
        self.game_state = np.roll(self.game_state, -1, axis=3)
        self.game_state[0, :, :, -1] = x_

    def step(self, x, r):
        r = max(self.config['min_reward'], min(self.config['max_reward'], r))
        if not self.isTesting:
            if not self.episode_begining:
                self.exp_replay.add(self.game_state[:, :, :, -1], self.game_action, self.game_reward, False)
            else:
                for i in range(self.config['history_length'] - 1):
                    # add the resetted buffer
                    self.exp_replay.add(self.game_state[:, :, :, i], 0, 0, False)
                self.episode_begining = False
            self.observe(x, r)
            self.game_action = self.e_greedy_action(self.epsilon())
            if self.step_count > self.config['steps_before_training']:
                self.update_thread.join()
                self.update_thread = threading.Thread(target=self.update)
                self.update_thread.start()
            self.step_count += 1
        else:
            self.observe(x, r)
            self.game_action = self.e_greedy_action(0.01)
        return self.game_action
