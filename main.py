import gym

import argparse

import collections
import os

import numpy as np
import chainer

from researchutils import files
from researchutils.chainer import serializers

from naf import NAF


def build_env(args):
    return gym.make(args.env)


def load_params(naf, args):
    print('loading model params')

    naf._q.to_cpu()
    serializers.load_model(args.q_params, naf._q)
    if not args.gpu < 0:
        naf._q.to_gpu()


def save_params(naf, timestep, outdir, args):
    print('saving model params of iter: ', timestep)

    q_filename = 'q_iter-{}'.format(timestep)

    naf._q.to_cpu()
    serializers.save_model(os.path.join(outdir, q_filename), naf._q)

    if not args.gpu < 0:
        naf._q.to_gpu()


def run_training_loop(env, naf, args):
    replay_buffer = collections.deque(maxlen=args.max_buffer_size)
    s_current = env.reset()

    previous_evaluation = 0

    outdir = files.prepare_output_dir(base_dir=args.outdir, args=args)

    result_file = os.path.join(outdir, 'result.txt')
    if not files.file_exists(result_file):
        with open(result_file, "w") as f:
            f.write('timestep\tmean\tmedian\n')

    chainer.config.train = False
    chainer.config.enable_backprop = False
    experienced_episodes = 0
    for timestep in range(args.total_timesteps):
        s_current, a, r, s_next, done = naf.act_with_policy(env, s_current)
        non_terminal = np.float32(0 if done else 1)

        replay_buffer.append((s_current, a, r, s_next, non_terminal))

        s_current = s_next

        if len(replay_buffer) > args.batch_size * 10:
            chainer.config.train = True
            chainer.config.enable_backprop = True
            naf.train(replay_buffer, args.iterations, args.gamma, args.tau)
            chainer.config.train = False
            chainer.config.enable_backprop = False

        if done:
            experienced_episodes += 1
            print('experienced episodes: ', experienced_episodes)
            if args.evaluation_interval < timestep - previous_evaluation:
                print('evaluating policy at timestep: ', timestep)
                rewards = naf.evaluate_policy(env)
                print('rewards: ', rewards)
                mean = np.mean(rewards)
                median = np.median(rewards)

                print('mean: {mean}, median: {median}'.format(
                    mean=mean, median=median))
                with open(result_file, "a") as f:
                    f.write('{timestep}\t{mean}\t{median}\n'.format(
                        timestep=timestep, mean=mean, median=median))

                save_params(naf, timestep, outdir, args)
                previous_evaluation = timestep

            s_current = env.reset()


def start_training(args):
    env = build_env(args)

    naf = NAF(state_dim=env.observation_space.shape[0],
              action_num=env.action_space.shape[0],
              lr=args.learning_rate,
              batch_size=args.batch_size,
              device=args.gpu,
              shared_model=args.parameter_shared_model,
              clip_grads=args.clip_grads)
    load_params(naf, args)

    run_training_loop(env, naf, args)

    env.close()


def start_test_run(args):
    env = build_env(args)

    naf = NAF(state_dim=env.observation_space.shape[0],
              action_num=env.action_space.shape[0],
              lr=args.learning_rate,
              batch_size=args.batch_size,
              device=args.gpu,
              shared_model=args.parameter_shared_model)
    load_params(naf, args)

    chainer.config.train = False
    chainer.config.enable_backprop = False
    rewards = naf.evaluate_policy(env, render=True, save_video=args.save_video)
    print('rewards: ', rewards)
    mean = np.mean(rewards)
    median = np.median(rewards)

    print('mean: {mean}, median: {median}'.format(
        mean=mean, median=median))

    env.close()


def main():
    parser = argparse.ArgumentParser()

    # output
    parser.add_argument('--outdir', type=str, default='results')

    # Environment
    parser.add_argument('--env', type=str, default='Walker2d-v2')

    # Gpu
    parser.add_argument('--gpu', type=int, default=-1)

    # testing
    parser.add_argument('--test-run', action='store_true')
    parser.add_argument('--save-video', action='store_true')

    # params
    parser.add_argument('--q-params', type=str, default="")

    # Training parameters
    parser.add_argument('--total-timesteps', type=float, default=1000000)
    parser.add_argument('--learning-rate', type=float, default=1.0 * 1e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=0.001)
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--start-timesteps', type=int, default=1000)
    parser.add_argument('--evaluation-interval', type=float, default=5000)
    parser.add_argument('--iterations', type=int, default=5)
    parser.add_argument('--max-buffer-size', type=int, default=None)
    parser.add_argument('--parameter-shared-model', action='store_true')
    parser.add_argument('--clip-grads', action='store_true')

    args = parser.parse_args()

    if args.test_run:
        start_test_run(args)
    else:
        start_training(args)


if __name__ == "__main__":
    main()
