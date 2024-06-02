from sklearn.model_selection import train_test_split
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import time

from ansi import orange, green, cyan, red

from bitcoin_env import BitcoinTradingEnv
from tf_model1 import DQNAgent

def train(opt):
    start_time = time.time()
    # save all opts that are not data
    with open(f"experiments/{opt.experiment_id}/opts.txt", 'w') as f:
        for key, val in vars(opt).items():
            if key != 'train_data' and key != 'validation_data' and key != 'test_data':
                f.write(f"{key}: {val}\n")


    env = BitcoinTradingEnv(opt, 'train')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(env, state_size, action_size)
    done = False

    episode_rewards = []
    episode_net_worths = []
    episode_capitals = []
    episode_holdings = []

    for e in tqdm(range(1, opt.episodes+1)):
        # Reset environment and get initial state
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        
        total_reward = 0
        for time_step in range(opt.time_steps):
            # Agent takes action
            action = agent.act(state)
            
            # Environment responds to action
            next_state, reward, done, _ = env.step(action)
            
            reward = reward
            # Adjust reward if episode ends
            reward = reward if not done else -10
            total_reward += reward
            
            # Reshape next_state
            next_state = np.reshape(next_state, [1, state_size])
            
            # Store experience in replay memory
            agent.remember(state, action, reward, next_state, done)
            
            # Update state to next_state
            state = next_state
            
            if done:
                break
            
            # Perform experience replay if memory is sufficient
            if len(agent.memory) > opt.batch_size:
                agent.replay(opt.batch_size)

        # Ensure last action is a sell if holding Bitcoin
        if env.bitcoin_holdings > 0:
            next_state, reward, done, _ = env.step(2)  # Force sell action
            reward = reward if not done else -10
            total_reward += reward
            agent.remember(state, 2, reward, next_state, done)
            env.bitcoin_holdings = 0  # Update holdings to zero

        
        # Update target model after each episode
        agent.update_target_model()
        print(f"Episode: {e}/{opt.episodes}, Score: {total_reward}, Epsilon: {agent.epsilon:.2}")
        
        # Log performance metrics
        episode_rewards.append(total_reward)
        episode_net_worths.append(env.last_total_asset_value)
        episode_capitals.append(env.current_capital)
        episode_holdings.append(env.bitcoin_holdings)
        
        # Save model weights periodically
        if (e) % 10 == 0:
            agent.save(f"experiments/{opt.experiment_id}/model_weights_{e}.weights.h5")

    # Save final model weights
    agent.save(f"experiments/{opt.experiment_id}/model_weights_{e}.weights.h5")
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


# TODO: keep track of each episode's performance metrics for each time step, and plot on one graph
def validate_agent(env, agent, start_step, episodes=10):
    total_rewards = []
    total_net_worths = []
    
    for e in tqdm(range(1, episodes+1)):
        state = env.reset(start_step=start_step)
        state = np.reshape(state, [1, env.observation_space.shape[0]])
        
        total_reward = 0
        done = False
        
        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, env.observation_space.shape[0]])
            total_reward += reward
            state = next_state
        
        total_rewards.append(total_reward)
        total_net_worths.append(env.last_total_asset_value)
    
    avg_reward = np.mean(total_rewards)
    avg_net_worth = np.mean(total_net_worths)
    
    print(f"Validation - Avg Reward: {avg_reward}, Avg Net Worth: {avg_net_worth}")
    

    return total_rewards, total_net_worths

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
    return opt
    
if __name__ == "__main__":

    # Environment setup
    args = {'experiment_id':3,
            'data_mode': 'standard',
            'data_sample_nrows': None,
            'initial_capital': 10000,
            'transaction_fee': 0.0025,
            'minimum_transaction': 0.000030,
            'window_size': 10,
            'unrealized_gains_discount': 0.95,
            'log_reward': False,
            'batch_size': 32,
            'episodes': 3,
            'time_steps': 50,
            }
    opt = set_options(args)

    # Training
    # train(opt)

    # Validation - on train
    env = BitcoinTradingEnv(opt, 'train')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(env, state_size, action_size)
    agent.load(path=f"experiments/{opt.experiment_id}/model_weights_{opt.episodes}.weights.h5")
    validate_agent(env, agent, start_step=100000, episodes=10)
