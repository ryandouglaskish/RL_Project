
from sklearn.model_selection import train_test_split
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import time

import torch
from ansi import orange, green, cyan, red

from bitcoin_env import BitcoinTradingEnv

from PyTorchModel import DQNAgent

def train_dqn_agent(opt):
    start_time = time.time()

    agent = opt.agent
    state_size = opt.state_size
    env = opt.env
    episodes = opt.episodes
    batch_size = opt.batch_size

    episode_rewards = []
    episode_net_worths = []
    episode_capitals = []
    episode_holdings = []

    start_episode = 1
    if opt.loadpath:
        print(f"Loading model from {opt.loadpath}")
        agent.load(opt.loadpath)
        # Extract the epoch number from the loadpath
        start_episode = int(opt.loadpath.split('_')[-1].split('.')[0]) + 1
        print(f"Starting from episode {start_episode}")

    for e in tqdm(range(start_episode, start_episode + episodes + 1), desc="Training Episodes"):
        green(e)
        state = env.reset()
        state = np.reshape(state, [1, agent.state_size])

        total_reward = 0

        for time_step in range(500):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            reward = reward if not done else -10
            total_reward += reward
            next_state = np.reshape(next_state, [1, agent.state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state

            if done:
                break

            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

        if env.bitcoin_holdings > 0:
            next_state, reward, done, _ = env.step(2)
            reward = reward if not done else -10
            total_reward += reward
            agent.remember(state, 2, reward, next_state, done)
            env.bitcoin_holdings = 0

        agent.update_target_model()

        # Decay epsilon after each episode
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay

        print(f"Episode: {e + 1}/{episodes}, Score: {total_reward}, Epsilon: {agent.epsilon:.2}")

        episode_rewards.append(total_reward)
        episode_net_worths.append(env.last_total_asset_value)
        episode_capitals.append(env.current_capital)
        episode_holdings.append(env.bitcoin_holdings)

        if (e) % 10 == 0:
            agent.save(f"experiments/{opt.experiment_id}/model_weights_{e}.pth")

    # Save final model weights
    agent.save(f"experiments/{opt.experiment_id}/model_weights_{start_episode + episodes}.pth")
    #agent.save(f"experiments/{opt.experiment_id}/model_weights_{e}.weights.h5")
    # Plot the loss values
    plt.plot(agent.losses)
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.title('Loss Over Time')
    plt.savefig(f"experiments/{opt.experiment_id}/loss_plot_{e}.png")

    # Plot the performance metrics after all episodes
    plt.figure(figsize=(12, 8))

    # Plot the loss values
    plt.subplot(2, 2, 1)
    plt.plot(agent.losses)
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.title('Loss Over Time')

    # Plot the episode rewards
    plt.subplot(2, 2, 2)
    plt.plot(episode_rewards)
    plt.xlabel('Episodes')
    plt.ylabel('Total Reward')
    plt.title('Total Rewards Over Episodes')

    # Plot the episode net worth
    plt.subplot(2, 2, 3)
    plt.plot(episode_net_worths)
    plt.xlabel('Episodes')
    plt.ylabel('Net Worth')
    plt.title('Net Worth Over Episodes')

    # Plot the episode capitals and holdings
    plt.subplot(2, 2, 4)
    plt.plot(episode_capitals, label='Capital')
    plt.plot(episode_holdings, label='Holdings')
    plt.xlabel('Episodes')
    plt.ylabel('Value')
    plt.title('Capital and Holdings Over Episodes')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"experiments/{opt.experiment_id}/performance_plot_{e}.png")
    plt.close()

    # save performance metrics as pd dataframe
    episode_performances_df = pd.DataFrame({'episode_rewards': episode_rewards,
                                   'episode_net_worths': episode_net_worths,
                                   'episode_capitals': episode_capitals,
                                   'episode_holdings': episode_holdings})
    episode_performances_df.to_csv(f"experiments/{opt.experiment_id}/episode_performances_{e}.csv", index=False)

    orange(f"Training took {time.time() - start_time} seconds")


class Options:
    def __init__(self) -> None:
        pass

    def make_vars(self, args: dict):
        for key, val in args.items():
            self.__setattr__(key, val)


def set_options(args):
    opt = Options()
    opt.make_vars(args)

    if not os.path.exists(f'experiments/{opt.experiment_id}'):
        os.makedirs(f'experiments/{opt.experiment_id}')

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
    opt.train_data = train_data
    opt.validation_data = validation_data
    opt.test_data = test_data

    opt.env = BitcoinTradingEnv(opt, 'train')
    opt.state_size = opt.env.observation_space.shape[0]
    opt.action_size = opt.env.action_space.n

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    opt.device = device
    opt.agent = DQNAgent(opt.env, opt.state_size, opt.action_size, opt.device)
    return opt
    
if __name__ == "__main__":
   
    # ===========================================
    # Standard Reward Model
    # ===========================================
    # Environment setup
    args = {'experiment_id': 10,
            'data_mode': 'standard',
            'data_sample_nrows': None,
            'initial_capital': 1000,
            'transaction_fee': 0.0025,
            'minimum_transaction': 0.000030,
            'window_size': 15,
            'unrealized_gains_discount': 0.95,
            'log_reward': False,
            'batch_size': 32,
            'episodes': 1000,
            'time_steps': 45,
            'loadpath': None
            }
    opt = set_options(args)

    # Training
    train_dqn_agent(opt)

    # Validation - on train
    # env = BitcoinTradingEnv(opt, 'train')
    # state_size = env.observation_space.shape[0]
    # action_size = env.action_space.n
    # agent = DQNAgent(opt.env, opt.state_size, opt.action_size)
    # agent.load(path=f"experiments/{opt.experiment_id}/model_weights_{opt.episodes}.weights.h5")
    # validate_agent(env, agent, start_step=100000, episodes=10)


    # ===========================================
    # Log Reward Model
    # ===========================================
    # Environment setup
    args2 = {'experiment_id': 11,
            'data_mode': 'standard',
            'data_sample_nrows': None,
            'initial_capital': 1000,
            'transaction_fee': 0.0025,
            'minimum_transaction': 0.000030,
            'window_size': 15,
            'unrealized_gains_discount': 0.95,
            'log_reward': True,
            'batch_size': 32,
            'episodes': 400,
            'time_steps': 45,
            'loadpath': None
            }
    opt2 = set_options(args2)

    # Training
    train_dqn_agent(opt2)