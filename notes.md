monte-carlo tree search simulation
- e.g., use NN as approximation model to get Q value at certain depth from root, and explore with a fixed max depth


# Set up
https://www.youtube.com/watch?v=YLa_KkehvGw
`$ pip install gym==0.25.2 tensorflow keras-rl2 numpy`


# Env
`_get_observation` It selects a window of historical data from the `self.data` DataFrame. The window size is determined by `self.window_size`.

# Implementation decisions
0.000030 is minimum bitcoin transaction, since coinbase minimum is $2, and current bitcoin conversion is 0.000030

- is it a good idea to randomize the start of each episode?