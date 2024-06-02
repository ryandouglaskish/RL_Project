import gym
from gym import spaces
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import sys
import os

class Options:
    def __init__(self) -> None:
        pass

    def make_vars(self, args: dict):
        for key, val in args.items():
            self.__setattr__(key, val)

class BitcoinTradingEnv(gym.Env):
    metadata = {'render.modes': ['human', 'plot']}

    def __init__(self, opt):
        super(BitcoinTradingEnv, self).__init__()

        # Load data
        self.data = opt.data
        self.initial_capital = opt.initial_capital
        self.transaction_fee = opt.transaction_fee
        self.window_size = opt.window_size
        self.unrealized_gains_discount = opt.unrealized_gains_discount
        self.log_reward = opt.log_reward
        self.minimum_transaction = opt.minimum_transaction


        # Define action and observation space
        self.action_space = gym.spaces.Discrete(3)  # 0: hold, 1: buy, 2: sell
        self.observation_space = gym.spaces.Box(low=0, high=np.inf, shape=(self.window_size * len(self.data.columns),), dtype=np.float32)
        
        self.current_step = 0
        self.done = False
        self.bitcoin_holdings = 0
        self.current_capital = opt.initial_capital
        self.last_total_asset_value = opt.initial_capital
        self.performance = []

    def step(self, action):
        if self.done:
            return self._get_obs(), 0, self.done, {}

        current_price = self.data.iloc[self.current_step]['close']
        transaction_cost = self._calculate_transaction_cost(action, current_price)

        if action == 1:  # Buy
            # Check if the remaining capital after transaction cost can buy at least the minimum transaction amount of Bitcoin
            if (self.current_capital - transaction_cost) / current_price >= self.minimum_transaction:
                # Buy Bitcoin with the available capital after transaction cost
                self.bitcoin_holdings = (self.current_capital - transaction_cost) / current_price
                self.current_capital = 0  # All capital is spent

        elif action == 2:  # Sell
              # Check if the Bitcoin holdings are greater than or equal to the minimum transaction amount
            if self.bitcoin_holdings > self.minimum_transaction:
                # Sell all Bitcoin holdings and update the capital after transaction cost
                self.current_capital = self.bitcoin_holdings * current_price - transaction_cost
                self.bitcoin_holdings = 0  # All Bitcoin is sold

        # Calculate the unrealized value of Bitcoin holdings
        unrealized_value = self.bitcoin_holdings * current_price
        # Calculate total asset value with discount on unrealized gains
        total_asset_value = self.current_capital + self.unrealized_gains_discount * unrealized_value
        # Reward is the change in total asset value from the last step
        if self.log_reward:
            reward = np.log(total_asset_value / self.last_total_asset_value)
        else:
            reward = total_asset_value - self.last_total_asset_value
        self.last_total_asset_value = total_asset_value
        self.performance.append((self.current_step, self.current_capital, self.bitcoin_holdings, self.last_total_asset_value))
        
        self.current_step += 1
        if self.current_step >= len(self.data) - self.window_size + 1:
            self.done = True

        return self._get_obs(), reward, self.done, {}

    def _calculate_transaction_cost(self, action, price):
        if action in [1, 2]:  # Buying or selling
            amount = self.current_capital if action == 1 else self.bitcoin_holdings * price
            return amount * self.transaction_fee
        return 0

    def reset(self):
        self.current_step = self.window_size - 1
        self.current_capital = self.initial_capital
        self.bitcoin_holdings = 0
        self.done = False
        self.last_total_asset_value = self.initial_capital
        self.performance = []
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([self.data.iloc[self.current_step - self.window_size + 1: self.current_step + 1].values.flatten()])

    def render(self, mode='human', close=False):
        if mode == 'human':
            print(f'Step: {self.current_step}')
            print(f'Capital: {self.current_capital}')
            print(f'Bitcoin Holdings: {self.bitcoin_holdings}')
            print(f'Net Worth: {self.last_total_asset_value}')
        elif mode == 'plot':
            steps, capitals, holdings, net_worths = zip(*self.performance)
            plt.plot(steps, net_worths, label='Net Worth')
            plt.xlabel('Steps')
            plt.ylabel('Net Worth')
            plt.legend()
            plt.show()

if __name__ == "__main__":


    args = {'experiment_id':1,
            'data_mode': 'standard',
            'data_sample_nrows': 100,
            'initial_capital': 10000,
            'transaction_fee': 0.0025,
            'minimum_transaction': 0.000030,
            'window_size': 10,
            'unrealized_gains_discount': 0.95,
            'log_reward': True}
    opt = Options()
    opt.make_vars(args)


    data_directory = 'data/processed'
    opt.data_path = data_directory + f'/{opt.data_mode}.csv'

    if opt.data_sample_nrows:
        opt.data = pd.read_csv(opt.data_path, nrows=opt.data_sample_nrows)
    else:
        opt.data = pd.read_csv(opt.data_path)

    # Example usage
    env = BitcoinTradingEnv(opt)
    obs = env.reset()
    print(obs)
    done = False
    while not done:
        action = env.action_space.sample()  # Take a random action
        obs, reward, done, _ = env.step(action)
        env.render()
        if done:
            env.render(mode='plot')
            print("Episode finished")

