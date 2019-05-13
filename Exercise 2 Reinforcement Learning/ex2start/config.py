class AgentConfig(object):

  display = False

  batch_size = 32

  history_length = 4

  memory_size = 1000000

  max_step = 50000000

  learn_start = 0

  train_frequency = 4
  target_q_update_step = 10

  _test_step = 100


class EnvironmentConfig(object):
  env_name = 'Breakout-v0'

  screen_width  = 84
  screen_height = 84
  max_reward = 1.
  min_reward = -1.

  random_start = 30

class DQNConfig(AgentConfig, EnvironmentConfig):
  model = ''
  pass

class M1(DQNConfig):
  backend = 'tf'

def get_config(FLAGS):
  if FLAGS.model == 'm1':
    config = M1
  # elif FLAGS.model == 'm2':
  #   config = M2

  # for k, v in FLAGS.__dict__['__flags'].items():
  for k, v in FLAGS.__flags.items():
    if k == 'use_gpu':
      if v.value == False:
        config.cnn_format = 'NHWC'
      else:
        config.cnn_format = 'NCHW'

    if hasattr(config, k):
      setattr(config, k, v.value)

  return config
