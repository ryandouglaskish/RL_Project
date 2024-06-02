import gym
from gym import spaces
import numpy as np
import pandas as pd

class BitcoinTradingEnv(gym.Env):
    def __init__(self, data_files, window_size=10, mode='standard', initial_balance=10000):
        super(BitcoinTradingEnv, self).__init__()
        
        # Modes of data processing: 'standard', 'deltas', 'log_deltas'
        self.mode = mode
        
        # Load the appropriate data file based on the mode
        self.data = pd.read_csv(data_files[self.mode])
        
        # Window size for state observation
        self.window_size = window_size
        
        # Initial balance
        self.initial_balance = initial_balance
        
        # Transaction fee
        self.transaction_fee = 0.0025  # 0.25%
        
        # Define action and observation space
        self.action_space = spaces.Discrete(3)  # [0: Hold, 1: Buy, 2: Sell]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(window_size, self.data.shape[1] + 3), dtype=np.float32
        )
        
        # Initialize the environment
        self.reset()

    def reset(self):
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.btc_held = 0
        self.current_step = self.window_size
        self.done = False
        return self._get_observation()

    def step(self, action):
        current_price = self.data.iloc[self.current_step]['close']
        reward = 0
        
        # Execute the action
        if action == 1:  # Buy
            reward = self._buy(current_price)
        elif action == 2:  # Sell
            reward = self._sell(current_price)
        
        # Move to the next step
        self.current_step += 1
        if self.current_step >= len(self.data) - 1:
            self.done = True
        
        # Calculate the new net worth
        self.net_worth = self.balance + self.btc_held * current_price
        
        obs = self._get_observation()
        return obs, reward, self.done, {}

    def render(self, mode='human'):
        print(f'Step: {self.current_step}')
        print(f'Balance: {self.balance}')
        print(f'BTC Held: {self.btc_held}')
        print(f'Net Worth: {self.net_worth}')

    def _get_observation(self):
        window_data = self.data.iloc[self.current_step-self.window_size:self.current_step].copy()
        obs = np.concatenate((window_data.values, [[self.balance, self.btc_held, self.net_worth]] * self.window_size), axis=1)
        return obs

    def _buy(self, price):
        if self.balance > 0:
            btc_bought = (self.balance / price) * (1 - self.transaction_fee)
            self.btc_held += btc_bought
            self.balance = 0
            return btc_bought * price  # Replace with actual reward calculation if needed
        return 0

    def _sell(self, price):
        if self.btc_held > 0:
            btc_sold = self.btc_held
            self.balance += btc_sold * price * (1 - self.transaction_fee)
            self.btc_held = 0
            return btc_sold * price  # Replace with actual reward calculation if needed
        return 0


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
