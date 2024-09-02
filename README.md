# Pacman Q-Learning Agent

## Overview
This project implements a Q-Learning agent for the Pacman AI from the Berkeley AI Project. The agent uses reinforcement learning to optimize the actions Pacman takes in various game states. The code was adapted for use at King's College London (KCL) and has been updated to Python 3 for the 2022 course by Dylan Cope and Lin Li.

The goal of this agent is to learn an optimal policy for Pacman through training episodes, utilizing hyperparameters such as learning rate (`alpha`), discount factor (`gamma`), and exploration rate (`epsilon`). 

## Features
- **Q-Learning Algorithm**: The agent uses the Q-learning algorithm to estimate the optimal policy for Pacman by updating state-action pairs based on the observed rewards and the maximum expected future rewards.
- **Exploration and Exploitation**: The agent balances exploration and exploitation using epsilon-greedy exploration, enabling it to explore the environment while also making use of the learned policy.
- **Count-Based Exploration**: The agent tracks the number of visits to state-action pairs to encourage exploration of less-visited states.

## Key Components
- **`GameStateFeatures`**: A wrapper around the `GameState` class that allows easy extraction of information for the Q-learning algorithm.
- **`QLearnAgent`**: The main reinforcement learning agent that implements Q-learning updates, action selection, and exploration.
- **`computeReward`**: Computes the reward for the transition between game states based on the difference in game scores.
- **`learn`**: Updates the Q-values based on the Q-learning formula, incorporating the learning rate and discount factor.
- **`getAction`**: Chooses the next action for Pacman, either greedily or using exploration based on the epsilon-greedy policy.

## Usage
To use this Q-Learning agent with the Pacman AI framework, follow the instructions provided by the Berkeley AI Pacman project, which can be found at [Berkeley Pacman AI Project](http://ai.berkeley.edu/reinforcement.html).

### Example Command
You can run the Pacman simulation with the Q-learning agent using the following command:
```bash
python pacman.py -p QLearnAgent -x 2000 -n 2010 -l smallGrid
