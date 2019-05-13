#
# cartpole.py, exercise sheet 2, Advanced Machine Learning course, RWTH Aachen University, summer term 2019, Jonathon Luiten
#

import gym
# render a dummy environment before importing tensorflow to circumvent tensorflow/openai-gym integration bug
# g_env = gym.make('CartPole-v0')
# g_env.render()

import tensorflow as tf
import random
import numpy as np

num_training_episodes = 1000
episode_length = 200

env = gym.make('CartPole-v0')
# env.render()
monitor = gym.wrappers.Monitor(env, 'cartpole/', force=True)

sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())

print(env.action_space,'    ',env.observation_space)
print(env.observation_space.high)   # 查看 observation 最高取值
print(env.observation_space.low)    # 查看 observation 最低取值

Q=np.zeros([10**4,env.action_space.n])

alpha=0.2
gamma=0.99

def bins(left,right,num):
    return np.linspace(left,right,num+1)[1:-1]

def get_state_bins(observation):

    cart_pos, cart_v, pole_angle, pole_v = observation
    state_bins=0
    state_bins=np.digitize(cart_pos, bins=bins(-4, 4, 10))
    state_bins=state_bins*10+np.digitize(cart_v, bins=bins(-3.0, 3.0, 10))
    state_bins=state_bins*10+np.digitize(pole_angle, bins=bins(-0.4, 0.4, 10))
    state_bins=state_bins*10+np.digitize(pole_v, bins=bins(-3, 3, 10))

    # print(observation,'=',state_bins)
    return state_bins

def get_action(observation,Q,episode):
    state_next=get_state_bins(observation)
    # epsilon=1/np.sqrt(episode+1)
    epsilon=0.5 * (0.99 ** episode)
    if episode<=np.random.uniform(0,1):
        action_next=np.argmax(Q[state_next])
    else:
        action_next=np.random.choice([0,1])
    return action_next

def run_episode( env, sess ):

    #Initialize S
    observation = env.reset()
    episode_return = 0
    state=get_state_bins(observation)

    for t in range( episode_length ):

        # random policy
        # action = 0 if random.uniform(0,1) < 0.5 else 1
        action = get_action(observation,Q,t)

        observation_next, reward, done, info = env.step(action)
        state_next=get_state_bins(observation_next)

        if done:
            reward=-200

        episode_return += reward

        maxQ=max(Q[state_next,0],Q[state_next,1])

        # print('maxQ=',maxQ)

        Q[state,action]+=alpha*(reward+gamma*maxQ-Q[state,action])
        state=state_next

        # disable rendering for faster training
        # env.render() #每一帧重新渲染环境

        if done:
            print("episode ended early")
            break


    print("episode return: %f time:%s"%(episode_return,t))

    return episode_return



for i in range( num_training_episodes ):

    print('episode=',i,end=' ')
    episode_return = run_episode( env, sess )


monitor.close()

