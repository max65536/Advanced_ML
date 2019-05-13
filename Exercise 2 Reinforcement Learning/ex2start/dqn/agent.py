from __future__ import print_function
import os
import time
import random
import numpy as np
from tqdm import tqdm
import tensorflow as tf

from .base import BaseModel
from .history import History
from .replay_memory import ReplayMemory
from .utils import get_time, save_pkl, load_pkl

class Agent(BaseModel):
  def __init__(self, config, environment, sess):
    super(Agent, self).__init__(config)
    self.sess = sess
    self.weight_dir = 'weights'

    self.env = environment
    self.history = History(self.config)
    self.memory = ReplayMemory(self.config, self.model_dir)

    with tf.variable_scope('step'):
      self.step_op = tf.Variable(0, trainable=False, name='step')
      self.step_input = tf.placeholder('int32', None, name='step_input')
      self.step_assign_op = self.step_op.assign(self.step_input)

    self.build_dqn()

  def train(self):
    start_step = self.step_op.eval()
    start_time = time.time()

    num_game = 0.
    total_reward = 0.

    # screen, reward, action, terminal = self.env.reset()
    screen, reward, action, terminal = self.env.new_random_game()

    for _ in range(self.history_length):
      self.history.add(screen)

    for self.step in tqdm(range(start_step, self.max_step), ncols=70, initial=start_step):
      if self.step == self.learn_start:
        num_game = 0.
        total_reward = 0.

      # self.env.render()

      action = self.predict(self.history.get())

      screen, reward, terminal = self.env.act(action, is_training=True)

      self.observe(screen, reward, action, terminal)

      if terminal:
        print( "\ngame: %i, return: %f\n"%(num_game, total_reward) )
        screen, reward, action, terminal = self.env.new_random_game()
        num_game += 1
        total_reward = 0

      total_reward += reward


  def predict( self, s_t ):

    action = random.randrange(self.env.action_size)

    return action

  def observe(self, screen, reward, action, terminal):
    reward = max(self.min_reward, min(self.max_reward, reward))

    self.history.add(screen)
    self.memory.add(screen, reward, action, terminal)

    if self.step > self.learn_start:
      if self.step % self.train_frequency == 0:
        self.q_learning_mini_batch()

      if self.step % self.target_q_update_step == self.target_q_update_step - 1:
        self.update_target_q_network()


  def q_learning_mini_batch(self):
    if self.memory.count < self.history_length:
      return
    else:
      s_t, action, reward, s_t_plus_1, terminal = self.memory.sample()



  def build_dqn(self):

    self.w = {}
    self.t_w = {}

    with tf.variable_scope('summary'):
      scalar_summary_tags = ['average.reward', 'episode.num of game']

      self.summary_placeholders = {}
      self.summary_ops = {}

      for tag in scalar_summary_tags:
        self.summary_placeholders[tag] = tf.placeholder('float32', None, name=tag.replace(' ', '_'))
        self.summary_ops[tag] = tf.summary.scalar("%s/%s" % (self.env_name, tag),
                                                  self.summary_placeholders[tag])

      #histogram_summary_tags = ['episode.rewards', 'episode.actions']

      #for tag in histogram_summary_tags:
        #self.summary_placeholders[tag] = tf.placeholder('float32', None, name=tag.replace(' ', '_'))
        #self.summary_ops[tag] = tf.summary.histogram(tag, self.summary_placeholders[tag])

      self.writer = tf.summary.FileWriter('./logs/%s' % 'test1', self.sess.graph)

    tf.initialize_all_variables().run()

    # print('-------------------------')
    # print(self.w.values() , [self.step_op])
    self._saver = tf.train.Saver(list(self.w.values()) + [self.step_op], max_to_keep=30)

    self.load_model()
    self.update_target_q_network()

  def update_target_q_network(self):
    for name in self.w.keys():
      self.t_w_assign_op[name].eval({self.t_w_input[name]: self.w[name].eval()})

  def save_weight_to_pkl(self):
    if not os.path.exists(self.weight_dir):
      os.makedirs(self.weight_dir)

    for name in self.w.keys():
      save_pkl(self.w[name].eval(), os.path.join(self.weight_dir, "%s.pkl" % name))

  def load_weight_from_pkl(self, cpu_mode=False):
    with tf.variable_scope('load_pred_from_pkl'):
      self.w_input = {}
      self.w_assign_op = {}

      for name in self.w.keys():
        self.w_input[name] = tf.placeholder('float32', self.w[name].get_shape().as_list(), name=name)
        self.w_assign_op[name] = self.w[name].assign(self.w_input[name])

    for name in self.w.keys():
      self.w_assign_op[name].eval({self.w_input[name]: load_pkl(os.path.join(self.weight_dir, "%s.pkl" % name))})

    self.update_target_q_network()

  def inject_summary(self, tag_dict, step):
    summary_str_lists = self.sess.run([self.summary_ops[tag] for tag in tag_dict.keys()], {
      self.summary_placeholders[tag]: value for tag, value in tag_dict.items()
    })
    for summary_str in summary_str_lists:
      self.writer.add_summary(summary_str, self.step)
#-----------------get render---------------------#
  def play(self, n_step=10000, n_episode=100, test_ep=None, render=True):
    if test_ep == None:
      test_ep = self.ep_end

    test_history = History(self.config)

    if not self.display:
      gym_dir = '/tmp/%s-%s' % (self.env_name, get_time())
      self.env.env.monitor.start(gym_dir)

    best_reward, best_idx = 0, 0
    for idx in xrange(n_episode):
      screen, reward, action, terminal = self.env.new_random_game()
      current_reward = 0

      for _ in range(self.history_length):
        test_history.add(screen)

      for t in tqdm(range(n_step), ncols=70):
        # 1. predict
        action = self.predict(test_history.get(), test_ep)
        # 2. act
        screen, reward, terminal = self.env.act(action, is_training=False)
        # 3. observe
        test_history.add(screen)

        current_reward += reward
        if terminal:
          break

      if current_reward > best_reward:
        best_reward = current_reward
        best_idx = idx

      print("="*30)
      print(" [%d] Best reward : %d" % (best_idx, best_reward))
      print("="*30)

    if not self.display:
      self.env.env.monitor.close()
