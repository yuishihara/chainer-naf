import chainer
import chainer.functions as F
from chainer import optimizers
from chainer import iterators
from chainer.datasets import tuple_dataset
from chainer.dataset import concat_examples
from chainer.distributions import Normal

from collections import deque

from models.naf_q_function import NafQFunction

import gym

import cupy as cp
import numpy as np


class NAF(object):
    def __init__(self, state_dim, action_num, lr=1.0 * 1e-3, batch_size=100, device=-1):
        super(NAF, self).__init__()
        self._q_optimizer = optimizers.Adam(alpha=lr)

        self._batch_size = batch_size
        self._q = NafQFunction(state_dim=state_dim, action_num=action_num)

        self._target_q = NafQFunction(
            state_dim=state_dim, action_num=action_num)

        if not device < 0:
            self._q.to_gpu()
            self._target_q.to_gpu()

        self._q_optimizer.setup(self._q)

        xp = np if device < 0 else cp

        mean = xp.zeros(shape=(action_num), dtype=xp.float32)
        sigma = xp.ones(shape=(action_num), dtype=xp.float32)
        self._exploration_noise = Normal(loc=mean, scale=sigma * 0.1)

        self._device = device
        self._initialized = False

        self._action_num = action_num

    def act_with_policy(self, env, s):
        s = np.float32(s)
        state = chainer.Variable(np.reshape(s, newshape=(1, ) + s.shape))
        if not self._device < 0:
            state.to_gpu()

        a = self._q.pi(state)
        if not self._device < 0:
            a.to_cpu()
        noise = self._sample_exploration_noise(shape=(1))

        # print('a shape:', a.shape, ' noise shape: ', noise.shape)
        assert a.shape == noise.shape

        a = np.squeeze((a + noise).data, axis=0)
        s_next, r, done, _ = env.step(a)

        s_next = np.float32(s_next)
        a = np.float32(a)
        r = np.float32(r)
        return s, a, r, s_next, done

    def evaluate_policy(self, env, render=False, save_video=False):
        if save_video:
            from OpenGL import GL
            env = gym.wrappers.Monitor(env, directory='video',
                                       write_upon_reset=True, force=True, resume=True, mode='evaluation')
            render = False

        rewards = []
        training_setting = chainer.config.train
        if training_setting:
            chainer.config.train = False
        for _ in range(10):
            s = env.reset()
            episode_reward = 0
            while True:
                if render:
                    env.render()
                s = np.float32(s)
                s = chainer.Variable(np.reshape(s, newshape=(1, ) + s.shape))
                if not self._device < 0:
                    s.to_gpu()

                a = self._q.pi(s)
                if not self._device < 0:
                    a.to_cpu()
                a = np.squeeze(a.data, axis=0)
                s, r, done, _ = env.step(a)
                episode_reward += r
                if done:
                    rewards.append(episode_reward)
                    break
        chainer.config.train = training_setting
        return rewards

    def train(self, replay_buffer, iterations, gamma, tau):
        if not self._initialized:
            self._initialize_target_networks()
        iterator = self._prepare_iterator(replay_buffer)
        for _ in range(iterations):
            batch = iterator.next()
            s_current, action, r, s_next, non_terminal = concat_examples(
                batch, device=self._device)

            r = F.reshape(r, shape=(*r.shape, 1))
            non_terminal = F.reshape(
                non_terminal, shape=(*non_terminal.shape, 1))
            value = self._target_q.value(s_next) * non_terminal
            y = value * gamma + r
            y.unchain()

            q = self._q(s_current, action)
            q_loss = F.mean_squared_error(y, q)

            self._q_optimizer.target.cleargrads()
            q_loss.backward()
            q_loss.unchain_backward()
            self._q_optimizer.update()
            self._update_target_network(self._target_q, self._q, tau)

    def _initialize_target_networks(self):
        self._update_target_network(self._target_q, self._q, 1.0)
        self._initialized = True

    def _update_target_network(self, target, origin, tau):
        for target_param, origin_param in zip(target.params(), origin.params()):
            target_param.data = tau * origin_param.data + \
                (1.0 - tau) * target_param.data

    def _sample_exploration_noise(self, shape):
        return self._exploration_noise.sample(shape)

    def _prepare_iterator(self, buffer):
        return iterators.SerialIterator(buffer, self._batch_size)
