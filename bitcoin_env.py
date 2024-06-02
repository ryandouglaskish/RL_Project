import gym
from gym import spaces
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import sys
import os
import random
from sklearn.model_selection import train_test_split

class Options:
    def __init__(self) -> None:
        pass

    def make_vars(self, args: dict):
        for key, val in args.items():
            self.__setattr__(key, val)

class BitcoinTradingEnv(gym.Env):
    metadata = {'render_modes': ['human', 'plot']}

    def __init__(self, opt, dataset='train'):
        super(BitcoinTradingEnv, self).__init__()

        self.experiment_id = opt.experiment_id
        # Load data
        if dataset == 'train':
            self.data = opt.train_data
        elif dataset == 'validate':
            self.data = opt.validation_data
        elif dataset == 'test':
            self.data = opt.test_data
        self.dataset = dataset
        
        self.initial_capital = opt.initial_capital
        self.transaction_fee = opt.transaction_fee
        self.window_size = opt.window_size
        self.unrealized_gains_discount = opt.unrealized_gains_discount
        self.log_reward = opt.log_reward
        self.minimum_transaction = opt.minimum_transaction

        self.render_mode

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
        self.performance.append((self.current_step, self.current_capital, self.bitcoin_holdings, unrealized_value, self.last_total_asset_value))        
       
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
        #self.current_step = self.window_size - 1
        self.current_step = random.randint(self.window_size, len(self.data) - self.window_size - 1)
        self.current_capital = self.initial_capital
        self.bitcoin_holdings = 0
        self.done = False
        self.last_total_asset_value = self.initial_capital
        self.performance = []
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([self.data.iloc[self.current_step - self.window_size + 1: self.current_step + 1].values.flatten()])

    def render(self):
        print(f'Step: {self.current_step}')
        print(f'Capital: {self.current_capital}')
        print(f'Bitcoin Holdings: {self.bitcoin_holdings}')
        print(f'Net Worth: {self.last_total_asset_value}')
        

    def render_performance(self):
        steps, capitals, holdings, holdings_usd, net_worths = zip(*self.performance)
        plt.figure(figsize=(14, 7))
        plt.plot(steps, capitals, label='Capital')
        plt.plot(steps, holdings_usd, label='Unrealized Value')
        plt.plot(steps, net_worths, label='Net Worth')
        plt.xlabel('Steps')
        plt.ylabel('Value')
        plt.legend()
        if not os.path.exists(f'viz/experiments/{self.experiment_id}'):
            os.makedirs(f'viz/experiments/{self.experiment_id}')
        plt.savefig(f'viz/experiments/{self.experiment_id}/{self.dataset} Plot.png')
        plt.close()
        df = pd.DataFrame({'Steps': steps, 'Capital': capitals, 'Holdings': holdings, 'Holdings USD': holdings_usd, 'Net Worth': net_worths})
        df.to_csv(f'viz/experiments/{self.experiment_id}/{self.dataset} Performance.csv')


if __name__ == "__main__":

    if not os.path.exists('viz/experiments'):
        os.makedirs('viz/experiments')

    args = {'experiment_id':1,
            'data_mode': 'standard',
            'data_sample_nrows': 1000,
            'initial_capital': 10000,
            'transaction_fee': 0.0025,
            'minimum_transaction': 0.000030,
            'window_size': 10,
            'unrealized_gains_discount': 0.95,
            'log_reward': True}
    opt = Options()
    opt.make_vars(args)

    # if experiment already exists, delete all contents
    if os.path.exists(f'viz/experiments/{opt.experiment_id}'):
        for file in os.listdir(f'viz/experiments/{opt.experiment_id}'):
            os.remove(f'viz/experiments/{opt.experiment_id}/{file}')


    data_directory = 'data/processed'
    opt.data_path = data_directory + f'/{opt.data_mode}.csv'

    if opt.data_sample_nrows:
        full_data = pd.read_csv(opt.data_path, nrows=opt.data_sample_nrows)
    else:
        full_data = pd.read_csv(opt.data_path)
    full_data.drop(columns=['date'], inplace=True)

    # Split the data
    train_data, temp_data = train_test_split(full_data, train_size=0.7, shuffle=False)
    validation_data, test_data = train_test_split(temp_data, test_size=0.5, shuffle=False)
    print(f"Train data shape: {train_data.shape}")
    print(f"Validation data shape: {validation_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    opt.train_data = train_data
    opt.validation_data = validation_data
    opt.test_data = test_data

    # Train data
    train_env = BitcoinTradingEnv(opt, 'train')
    
    obs = train_env.reset()
    done = False
    while not done:
        action = train_env.action_space.sample()  # Take a random action
        obs, reward, done, _ = train_env.step(action)
        train_env.render()
        if done:
            train_env.render_performance()
            print("Episode finished")


    # Validation data
    val_env = BitcoinTradingEnv(opt, 'validate')
    
    obs = val_env.reset()
    done = False
    while not done:
        action = val_env.action_space.sample()  # Take a random action
        obs, reward, done, _ = val_env.step(action)
        val_env.render()
        if done:
            # val_env.render(mode='plot')
            val_env.render_performance()
            print("Episode finished")

    # Test data
    test_env = BitcoinTradingEnv(opt, 'test')
    
    obs = test_env.reset()
    # print(obs)
    done = False
    while not done:
        action = test_env.action_space.sample()  # Take a random action
        obs, reward, done, _ = test_env.step(action)
        test_env.render()
        if done:
            test_env.render_performance()
            print("Episode finished")

