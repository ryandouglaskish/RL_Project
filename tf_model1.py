import tensorflow as tf
from tensorflow.keras import layers

from collections import deque
import random
import numpy as np

class DQN(tf.keras.Model):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.dense1 = layers.Dense(128, activation='relu')
        self.dense2 = layers.Dense(128, activation='relu')
        self.out = layers.Dense(output_dim)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.out(x)
class DQNAgent:
    def __init__(self, env, state_size, action_size):
        self.env = env
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        self.losses = []

    def _build_model(self):
        model = DQN(self.state_size, self.action_size)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                      loss='mse')
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = np.reshape(state, [1, self.state_size])
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_state = np.reshape(next_state, [1, self.state_size])
                next_state = tf.convert_to_tensor(next_state, dtype=tf.float32)
                target = (reward + self.gamma * np.amax(self.target_model.predict(next_state, verbose=0)[0]))
            state = np.reshape(state, [1, self.state_size])
            state = tf.convert_to_tensor(state, dtype=tf.float32)
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target
            history = self.model.fit(state, target_f, epochs=1, verbose=0)  # Suppress output by setting verbose=0
            self.losses.append(history.history['loss'][0])  # Capture the loss value
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, path):
        self.model.load_weights(path)
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate), loss='mse')  # Recompile the model

    def save(self, path):
        self.model.save_weights(path)

