import gym
from gym import spaces
import numpy as np
import pandas as pd
class BitcoinTradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, data_files, mode='standard', initial_capital=10000, transaction_fee=0.0025, window_size=10, unrealized_gains_discount=0.95):
        super(BitcoinTradingEnv, self).__init__()

        # Load data
        self.data = pd.read_csv(data_files[mode])
        self.initial_capital = initial_capital
        self.transaction_fee = transaction_fee
        self.window_size = window_size
        self.unrealized_gains_discount = unrealized_gains_discount

        # Define action and observation space
        self.action_space = gym.spaces.Discrete(3)  # 0: hold, 1: buy, 2: sell
        self.observation_space = gym.spaces.Box(low=0, high=np.inf, shape=(self.window_size * len(self.data.columns),), dtype=np.float32)

        self.current_step = 0
        self.done = False
        self.bitcoin_holdings = 0
        self.current_capital = initial_capital
        self.last_total_asset_value = initial_capital

    def step(self, action):
        if self.done:
            return self._get_obs(), 0, self.done, {}

        current_price = self.data.iloc[self.current_step]['close']
        transaction_cost = self._calculate_transaction_cost(action, current_price)

        if action == 1:  # Buy
            if self.current_capital > transaction_cost:
                self.bitcoin_holdings = (self.current_capital - transaction_cost) / current_price
                self.current_capital = 0

        elif action == 2:  # Sell
            if self.bitcoin_holdings > 0:
                self.current_capital = self.bitcoin_holdings * current_price - transaction_cost
                self.bitcoin_holdings = 0

        # Calculate total asset value with discount on unrealized gains
        unrealized_value = self.bitcoin_holdings * current_price
        total_asset_value = self.current_capital + self.unrealized_gains_discount * unrealized_value
        reward = total_asset_value - self.last_total_asset_value
        self.last_total_asset_value = total_asset_value

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
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([self.data.iloc[self.current_step - self.window_size + 1: self.current_step + 1].values.flatten()])

    def render(self, mode='human', close=False):
        print(f'Step: {self.current_step}')
        print(f'Capital: {self.current_capital}')
        print(f'Bitcoin Holdings: {self.bitcoin_holdings}')



if __name__ == "__main__":

    data_directory = 'data/processed'
    data_files = {
        'standard': f'{data_directory}/standard.csv',
        'deltas': f'{data_directory}/delta.csv',
        'log_deltas': f'{data_directory}/log_delta.csv'
    }

    # Example usage
    env = BitcoinTradingEnv(data_files=data_files, mode='standard')
    obs = env.reset()
    print(obs)
    done = False
    while not done:
        action = env.action_space.sample()  # Take a random action
        obs, reward, done, _ = env.step(action)
        env.render()
