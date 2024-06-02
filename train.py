from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from tqdm import tqdm


from bitcoin_env import BitcoinTradingEnv
from tf_model1 import DQNAgent

def train(opt):
    env = BitcoinTradingEnv(opt, 'train')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(env, state_size, action_size)
    done = False
    batch_size = opt.batch_size
    episodes = opt.episodes

    for e in tqdm(range(episodes)):
        # Reset environment and get initial state
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        for time in tqdm(range(500)):
            # Agent takes action
            action = agent.act(state)
            # Environment responds to action
            next_state, reward, done, _ = env.step(action)
            # Adjust reward if episode ends
            reward = reward if not done else -10
            # Reshape next_state
            next_state = np.reshape(next_state, [1, state_size])
            # Store experience in replay memory
            agent.remember(state, action, reward, next_state, done)
            # Update state to next_state
            state = next_state
            if done:
                # Update target model
                agent.update_target_model()
                print(f"episode: {e}/{episodes}, score: {time}, e: {agent.epsilon:.2}")
                break
            # Perform experience replay if memory is sufficient
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

        if e % 10 == 0:
            agent.save(f"model_weights_{e}.hdf5")

class Options:
    def __init__(self) -> None:
        pass

    def make_vars(self, args: dict):
        for key, val in args.items():
            self.__setattr__(key, val)

if __name__ == "__main__":
    # Environment setup
    args = {'experiment_id':1,
            'data_mode': 'standard',
            'data_sample_nrows': None,
            'initial_capital': 10000,
            'transaction_fee': 0.0025,
            'minimum_transaction': 0.000030,
            'window_size': 10,
            'unrealized_gains_discount': 0.95,
            'log_reward': True,
            'batch_size': 32,
            'episodes': 1000,
            }
    opt = Options()
    opt.make_vars(args)

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

    # Training
    train(opt)
