import chainer
import chainer.functions as F
from chainer import optimizers
from chainer import iterators
from chainer.datasets import tuple_dataset
from chainer.dataset import concat_examples
from chainer.distributions import Normal

from collections import deque

from models.naf_q_function import NafQFunction
from models.naf_shared_q_function import NafSharedQFunction


import gym

import cupy as cp
import numpy as np


class NAF(object):
    def __init__(self, state_dim, action_num, lr=1.0 * 1e-3, batch_size=100, device=-1, shared_model=False):
        super(NAF, self).__init__()
        self._q_optimizer = optimizers.Adam(alpha=lr)

        self._batch_size = batch_size

        if shared_model:
            self._q = NafSharedQFunction(state_dim=state_dim, action_num=action_num)
            self._target_q = NafSharedQFunction(
                state_dim=state_dim, action_num=action_num)
        else:
            self._q = NafQFunction(state_dim=state_dim, action_num=action_num)
            self._target_q = NafQFunction(
                state_dim=state_dim, action_num=action_num)

        if not device < 0:
            self._q.to_gpu()
            self._target_q.to_gpu()

        self._q_optimizer.setup(self._q)

        mean = np.zeros(shape=(action_num), dtype=np.float32)
        sigma = np.ones(shape=(action_num), dtype=np.float32)
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
        min_q = []
        max_q = []
        min_v = []
        max_v = []
        for _ in range(10):
            s = env.reset()
            episode_reward = 0
            min_q = np.finfo(np.float32).max
            max_q = np.finfo(np.float32).min
            min_v = np.finfo(np.float32).max
            max_v = np.finfo(np.float32).min
            while True:
                if render:
                    env.render()
                s = np.float32(s)
                s = chainer.Variable(np.reshape(s, newshape=(1, ) + s.shape))
                if not self._device < 0:
                    s.to_gpu()

                a = self._q.pi(s)
                q = self._q(s, a)
                v = self._q.value(s)
                if not self._device < 0:
                    a.to_cpu()
                    q.to_cpu()
                    v.to_cpu()
                a = np.squeeze(a.data, axis=0)
                s, r, done, _ = env.step(a)
                episode_reward += r
                q = np.squeeze(q.data, axis=0)
                v = np.squeeze(v.data, axis=0)
                min_q = min((min_q, q))
                max_q = max((max_q, q))
                min_v = min((min_v, v))
                max_v = max((max_v, v))
                if done:
                    rewards.append(episode_reward)
                    min_q.append(min_q)
                    max_q.append(max_q)
                    min_v.append(min_v)
                    max_v.append(max_v)
                    break
        print('min_q: {}, max_q: {}, min_v: {}, max_v: {}'.format(min_q, max_q, min_v, max_v))
        return rewards

    def target_q_value(self, state):
        chainer.config.train = False
        q_value = self._target_q.value(state)
        chainer.config.train = True
        return q_value

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
            value = self.target_q_value(s_next) * non_terminal
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


if __name__ == "__main__":
    naf = NAF(state_dim=3, action_num=3)
    naf._initialize_target_networks()
