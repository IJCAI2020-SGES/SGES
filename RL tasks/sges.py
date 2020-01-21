# Code in this file is adapted from
# https://github.com/modestyachts/ARS/blob/master/code/ars.py

import time, argparse, socket, ray, random
import gym
import sys
sys.path.append('./SGES')
from Utils import logz
from Utils.policies import *
from Utils.optimizers import SGD
from Utils.buffer import GradBuffer
from Utils.tools import get_output_folder, output_data


class SharedNoiseGenerator(object):
    def __init__(self, size, noise_params, seed=1):

        self.size = size

        # Noise hyperparameters
        self.alpha = noise_params['alpha']
        self.k = noise_params['k']
        self.noise_stddev = noise_params['noise_stddev']

        self.rg = np.random.RandomState(seed)
        random.seed(seed)

        # Orthonormal basis of the gradients subspace
        self.U = np.zeros((self.size, self.k))

    # For Humanoid-v2
    # def sample(self):
    #     if random.random() < self.alpha:
    #         epsilon = self.rg.randn(self.k) @ self.U.T
    #         epsilon = np.sqrt(self.size / self.k) * epsilon
    #         noise_type = 1
    #     else:
    #         epsilon = self.rg.randn(self.size)
    #         noise_type = 0
    #     return noise_type, epsilon

    # For HalfCheetah-v2 and Ant-v2
    def sample(self):
        """
        Sample Noise from the hybrid Probabilistic distribution
        """
        if random.random() < self.alpha:
            epsilon = self.rg.randn(self.k) @ self.U.T
            noise_type = 1
        else:
            epsilon = self.rg.randn(self.size)
            noise_type = 0

        epsilon = (np.sqrt(self.size) / np.linalg.norm(epsilon)) * epsilon

        return noise_type, epsilon

    def compute_grads(self, scores, noises):
        grads = np.zeros(self.size)
        for i in range(len(noises)):
            grads += (scores[i][0] - scores[i][1]) * noises[i]

        g_hat = grads / (2 * len(noises) * self.noise_stddev)
        # g_hat = grads / (2 * len(noises))    # For Humanoid-v2
        return g_hat

    def update(self, grads, alpha):
        self.U, _ = np.linalg.qr(grads)
        self.alpha = alpha


@ray.remote
class Worker(object):
    """
    Object class for parallel rollout generation.
    """
    def __init__(self, env_seed,
                 env='HalfCheetah-v2',
                 policy_params=None,
                 noise_params=None,
                 max_ep_len=1000):

        # Initialize OpenAI environment for each worker
        self.env = gym.make(env)
        self.env.seed(env_seed)

        self.policy = LinearPolicy(policy_params)
        self.w_policy = self.policy.get_weights()

        self.noise_generator = SharedNoiseGenerator(self.w_policy.size, noise_params, env_seed + 7)
        self.noise_stddev = noise_params['noise_stddev']

        self.max_ep_len = max_ep_len

    def get_weights_plus_stats(self):
        """
        Get current policy weights and current statistics of past states.
        """
        return self.policy.get_weights_plus_stats()

    def rollout(self):
        state, total_reward, steps = self.env.reset(), 0., 0
        for i in range(self.max_ep_len):
            action = self.policy.act(state)
            state, reward, done, _ = self.env.step(action)
            steps += 1
            total_reward += reward
            if done:
                break

        return total_reward, steps

    def do_rollouts(self, w_policy, num_rollouts=1, eval=False):
        """
        Generate multiple rollouts with a policy parametrized by w_policy.
        """
        grad_noise_rewards, random_noise_rewards, grad_noise, random_noise = [], [], [], []
        steps = 0

        for i in range(num_rollouts):
            if eval:
                self.policy.update_weights(w_policy)
                random_noise.append(-1)

                # Set to false so that evaluation rollouts are not used for updating state statistics
                self.policy.update_filter = False

                reward, _ = self.rollout()
                random_noise_rewards.append(reward)
            else:
                noise_type, noise = self.noise_generator.sample()
                if noise_type == 0:
                    random_noise.append(noise)
                else:
                    grad_noise.append(noise)
                noise = (self.noise_stddev * noise).reshape(w_policy.shape)

                # Set to true so that state statistics are updated
                self.policy.update_filter = True

                # Compute reward and number of timesteps used for positive perturbation rollout
                self.policy.update_weights(w_policy + noise)
                pos_reward, pos_steps = self.rollout()

                # Compute reward and number of timesteps used for negative pertubation rollout
                self.policy.update_weights(w_policy - noise)
                neg_reward, neg_steps = self.rollout()
                steps += pos_steps + neg_steps

                if noise_type == 0:
                    random_noise_rewards.append([pos_reward, neg_reward])
                else:
                    grad_noise_rewards.append([pos_reward, neg_reward])

        return {'grad_noise': grad_noise,
                'grad_noise_rewards': grad_noise_rewards,
                'random_noise': random_noise,
                'random_noise_rewards': random_noise_rewards,
                'steps': steps}

    def stats_increment(self):
        self.policy.state_filter.stats_increment()
        return

    def get_weights(self):
        return self.policy.get_weights()

    def get_filter(self):
        return self.policy.state_filter

    def sync_filter(self, other):
        self.policy.state_filter.sync(other)
        return

    def sync_noise_params(self, U, alpha):
        self.noise_generator.U = U
        self.noise_generator.alpha = alpha
        return


class SGES(object):
    """
    Object class implementing the SGES algorithm.
    """

    def __init__(self):

        self.seed = args.seed
        np.random.seed(self.seed)

        self.env = gym.make(args.env)
        self.env.seed(self.seed)
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]

        # Init parameters
        self._init_parameters()

        # Initialize workers with different random seeds
        print('Initializing workers.')
        self.workers = [Worker.remote(self.seed + 7 * i,
                                      env=args.env,
                                      policy_params=self.policy_params,
                                      noise_params=self.noise_params,
                                      max_ep_len=self.max_ep_len) for i in range(self.num_workers)]

        # Initialize policy
        self.policy = LinearPolicy(self.policy_params)
        self.w_policy = self.policy.get_weights()

        self.noise_generator = SharedNoiseGenerator(self.w_policy.size, self.noise_params, self.seed)

        # Initial a gradient archive which stores the recent k estimated gradients
        self.grad_buffer = GradBuffer(self.k, self.w_policy.size)

        # Initialize optimization algorithm
        self.optimizer = SGD(args.lr)
        print("Initialization of SGES complete.")

    def _init_parameters(self):

        self.timesteps = 0
        self.episodes = 0
        self.log_dir = args.dir_path
        self.max_ep_len = args.max_ep_len
        self.epochs = args.epochs
        self.warmup = args.warmup
        self.save_freq = args.save_freq
        self.pop_size = args.pop_size
        self.elite_size = args.elite_size
        self.num_workers = args.num_workers

        self.alpha = args.alpha
        self.k = args.k
        self.noise_stddev = args.noise_stddev

        self.policy_params = {'filter': args.filter, 'state_dim': self.state_dim, 'action_dim': self.action_dim}
        self.noise_params = {'alpha': self.alpha, 'k': self.k, 'noise_stddev': self.noise_stddev}

    def aggregate_rollouts(self, num_rollouts=None, eval=False):
        """
        Aggregate update step from rollouts generated in parallel.
        """

        num_rollouts = self.pop_size if num_rollouts is None else num_rollouts
        rollouts_per_worker = int(num_rollouts / self.num_workers)

        # Put policy weights in the object store
        policy_id = ray.put(self.w_policy)

        # Parallel generation of rollouts
        rollout_ids_one = [worker.do_rollouts.remote(policy_id, num_rollouts=rollouts_per_worker, eval=eval)
                           for worker in self.workers]

        rollout_ids_two = [worker.do_rollouts.remote(policy_id, num_rollouts=1, eval=eval)
                           for worker in self.workers[:(num_rollouts % self.num_workers)]]

        # Gather results
        results_one = ray.get(rollout_ids_one)
        results_two = ray.get(rollout_ids_two)

        grad_noise_rewards, random_noise_rewards, grad_noise, random_noise = [], [], [], []

        for result in results_one + results_two:
            if not eval:
                self.timesteps += result["steps"]
            grad_noise += result['grad_noise']
            grad_noise_rewards += result['grad_noise_rewards']
            random_noise += result['random_noise']
            random_noise_rewards += result['random_noise_rewards']

        noises = np.array(grad_noise + random_noise, dtype=np.float64)
        rollout_rewards = np.array(grad_noise_rewards + random_noise_rewards, dtype=np.float64)

        if eval:
            return rollout_rewards

        grad_noise_rewards = np.array(grad_noise_rewards, dtype=np.float64)
        random_noise_rewards = np.array(random_noise_rewards, dtype=np.float64)
        mean_grad_noise_reward = None if len(grad_noise_rewards) == 0 else np.mean(np.max(grad_noise_rewards, axis=1))
        mean_random_noise_reward = None if len(random_noise_rewards) == 0 else np.mean(np.max(random_noise_rewards, axis=1))
        print('Mean reward of gradient noise : ', mean_grad_noise_reward)
        print('Mean reward of random noise : ', mean_random_noise_reward)
        # print('Maximum reward of collected rollouts :', max(grad_noise_rewards.max(), random_noise_rewards.max()))

        # Select top performing directions if deltas_used < num_deltas
        max_rewards = np.max(rollout_rewards, axis=1)
        if self.elite_size > self.pop_size:
            self.elite_size = self.pop_size

        idx = np.arange(max_rewards.size)[max_rewards >= np.percentile(max_rewards, 100 * (1 - (self.elite_size / self.pop_size)))]
        noises = noises[idx, :]
        rollout_rewards = rollout_rewards[idx, :]

        # Rewards normalization: normalize rewards by their standard deviation
        rollout_rewards /= np.std(rollout_rewards)

        # Aggregate rollouts to form g_hat, the gradient used to compute SGD step
        g_hat = self.noise_generator.compute_grads(rollout_rewards, noises)

        return g_hat, mean_grad_noise_reward, mean_random_noise_reward

    def train(self):

        start_time = time.time()
        history_alpha, alpha = [], 0.5
        for epoch in range(1, self.epochs + 1):

            if epoch <= self.warmup:
                g_hat, _, _ = self.aggregate_rollouts()
            else:
                self.noise_generator.update(self.grad_buffer.grads.T, alpha=alpha)
                g_hat, mean_grad_noise_reward, mean_random_noise_reward = self.aggregate_rollouts()
                if mean_random_noise_reward and mean_grad_noise_reward:
                    alpha = alpha * 1.05 if mean_grad_noise_reward > mean_random_noise_reward else alpha / 1.05
                    alpha = 0.8 if alpha > 0.8 else alpha
                    alpha = 0.1 if alpha < 0.1 else alpha

            self.w_policy -= self.optimizer.step(g_hat).reshape(self.w_policy.shape)
            self.grad_buffer.add(g_hat)
            history_alpha.append(alpha)
            self.episodes += self.pop_size * 2

            print('Epoch:', epoch, ' done.', "Alpha: ", alpha)

            # Evaluate the policy
            rewards = self.aggregate_rollouts(num_rollouts=20, eval=True)
            print('Average Test Reward: ', np.mean(rewards))

            # Record statistics every 10 iterations
            if epoch % self.save_freq == 0:
                w = ray.get(self.workers[0].get_weights_plus_stats.remote())
                np.savez(self.log_dir + "/lin_policy_plus_" + str(epoch), w)

                logz.log_tabular("Time", time.time() - start_time)
                logz.log_tabular("Epoch", epoch)
                logz.log_tabular("EpisodeNums", self.episodes)
                logz.log_tabular("Timesteps", self.timesteps)
                logz.log_tabular("AverageTestReward", np.mean(rewards))
                logz.log_tabular("StdTestRewards", np.std(rewards))
                logz.log_tabular("MaxTestRewardRollout", np.max(rewards))
                logz.log_tabular("MinTestRewardRollout", np.min(rewards))
                logz.dump_tabular()

            # Get statistics from all workers
            for j in range(self.num_workers):
                self.policy.state_filter.update(ray.get(self.workers[j].get_filter.remote()))
            self.policy.state_filter.stats_increment()

            # Make sure master filter buffer is clear
            self.policy.state_filter.clear_buffer()
            # Sync all workers
            filter_id = ray.put(self.policy.state_filter)
            setting_filters_ids = [worker.sync_filter.remote(filter_id) for worker in self.workers]
            # Waiting for sync of all workers
            ray.get(setting_filters_ids)

            increment_filters_ids = [worker.stats_increment.remote() for worker in self.workers]
            # Waiting for increment of all workers
            ray.get(increment_filters_ids)

            # Sync all workers
            U_id = ray.put(self.noise_generator.U)
            alpha_id = ray.put(self.noise_generator.alpha)
            setting_noise_ids = [worker.sync_noise_params.remote(U_id, alpha_id) for worker in self.workers]
            ray.get(setting_noise_ids)

        output_data(self.log_dir + 'history_alpha.csv', history_alpha)

        return


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')

    parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--noise_stddev', type=float, default=0.05)
    parser.add_argument('--pop_size', type=int, default=16)
    parser.add_argument('--elite_size', type=int, default=12)
    parser.add_argument('--max_ep_len', type=int, default=1000)

    parser.add_argument('--alpha', type=float, default=0.)
    parser.add_argument('--k', type=int, default=12)
    parser.add_argument('--warmup', type=int, default=24)

    parser.add_argument('--filter', type=str, default='MeanStdFilter')
    parser.add_argument('--epochs', type=int, default=600)
    parser.add_argument('--save_freq', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--seed', type=int, default=2016)
    parser.add_argument('--dir_path', type=str, default='Logs/sges/')

    local_ip = socket.gethostbyname(socket.gethostname())
    ray.init()

    args = parser.parse_args()

    args.dir_path = get_output_folder(args.dir_path, args.env, args.seed)
    logz.configure_output_dir(args.dir_path)
    logz.save_params(vars(args))
    sges = SGES()
    sges.train()



