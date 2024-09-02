# mlLearningAgents.py
# parsons/27-mar-2017
#
# A stub for a reinforcement learning agent to work with the Pacman
# piece of the Berkeley AI project:
#
# http://ai.berkeley.edu/reinforcement.html
#
# As required by the licensing agreement for the PacMan AI we have:
#
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

# This template was originally adapted to KCL by Simon Parsons, but then
# revised and updated to Py3 for the 2022 course by Dylan Cope and Lin Li

from __future__ import absolute_import
from __future__ import print_function

import random
from collections import defaultdict

from pacman import Directions, GameState
from pacman_utils.game import Agent
from pacman_utils import util


class GameStateFeatures:
    """
    Wrapper class around a game state where you can extract
    useful information for your Q-learning algorithm

    WARNING: We will use this class to test your code, but the functionality
    of this class will not be tested itself
    """

    def __init__(self, state: GameState):
        """
        Args:
            state: A given game state object
        """

        self.currentState = state
        self.legal = [action for action in state.getLegalPacmanActions() if
                      action != Directions.STOP]


class QLearnAgent(Agent):

    def __init__(self,
                 alpha: float = 0.2,
                 epsilon: float = 0.05,
                 gamma: float = 0.8,
                 maxAttempts: int = 30,
                 numTraining: int = 10):
        """
        These values are either passed from the command line (using -a alpha=0.5,...)
        or are set to the default values above.

        The given hyperparameters are suggestions and are not necessarily optimal
        so feel free to experiment with them.

        Args:
            alpha: learning rate
            epsilon: exploration rate
            gamma: discount factor
            maxAttempts: How many times to try each action in each state
            numTraining: number of training episodes
        """
        super().__init__()
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.gamma = float(gamma)
        self.maxAttempts = int(maxAttempts)
        self.numTraining = int(numTraining)
        # Count the number of games we have played
        self.episodesSoFar = 0

        self.q_value = defaultdict(int)

        self.count = defaultdict(int)

        self.prev_state = None
        self.prev_action = None

    # Accessor functions for the variable episodesSoFar controlling learning
    def incrementEpisodesSoFar(self):
        self.episodesSoFar += 1

    def getEpisodesSoFar(self):
        return self.episodesSoFar

    def getNumTraining(self):
        return self.numTraining

    # Accessor functions for parameters
    def setEpsilon(self, value: float):
        self.epsilon = value

    def getAlpha(self) -> float:
        return self.alpha

    def setAlpha(self, value: float):
        self.alpha = value

    def getGamma(self) -> float:
        return self.gamma

    def getMaxAttempts(self) -> int:
        return self.maxAttempts

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    @staticmethod
    def computeReward(startState: GameState,
                      endState: GameState) -> float:
        """
        Args:
            startState: A starting state
            endState: A resulting state

        Returns:
            The reward assigned for the given trajectory
        """
        score = endState.getScore() - startState.getScore()
        return score

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def getQValue(self,
                  state: GameStateFeatures,
                  action: Directions) -> float:
        """
        Args:
            state: A given state
            action: Proposed action to take

        Returns:
            Q(state, action)
        """
        return self.q_value.get((state.currentState, action), 0)

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def maxQValue(self, state: GameStateFeatures) -> float:
        """
        Args:
            state: The given state

        Returns:
            q_value: the maximum estimated Q-value attainable from the state
        """
        # Get legal actions pacman can take excluding STOP
        qValues = [self.getQValue(state, action) for action in
                   state.legal]
        return max(qValues, default=0)

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def learn(self,
              state: GameStateFeatures,
              action: Directions,
              reward: float,
              nextState: GameStateFeatures):
        """
        Performs a Q-learning update

        Args:
            state: the initial state
            action: the action that was took
            nextState: the resulting state
            reward: the reward received on this trajectory
        """
        # Get the current Q-value for the given state-action pair.
        currentQValue = self.getQValue(state, action)

        # Get the maximum Q-value for the next state.
        maxNextQValue = self.maxQValue(nextState)

        # Update the Q-value for the current state-action pair using Q-learning formula.
        self.q_value[
            (state.currentState, action)] = currentQValue + self.alpha * (
                reward + self.gamma * maxNextQValue - currentQValue)

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def updateCount(self,
                    state: GameStateFeatures,
                    action: Directions):
        """
        Updates the stored visitation counts.

        Args:
            state: Starting state
            action: Action taken
        """
        self.count[(state, action)] += 1

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def getCount(self,
                 state: GameStateFeatures,
                 action: Directions) -> int:
        """
        Args:
            state: Starting state
            action: Action taken

        Returns:
            Number of times that the action has been taken in a given state
        """
        return self.count[(state, action)]

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def explorationFn(self,
                      utility: float,
                      counts: int) -> float:
        """
        Computes exploration function.
        Return a value based on the counts

        HINT: Do a greed-pick or a least-pick

        Args:
            utility: expected utility for taking some action a in some given state s
            counts: counts for having taken visited

        Returns:
            The exploration value
        """
        exploration_value = utility / (counts + 1)

        return exploration_value

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def getAction(self, state: GameState) -> Directions:
        """
        Choose an action to take to maximise reward while
        balancing gathering data for learning

        If you wish to use epsilon-greedy exploration, implement it in this method.
        HINT: look at pacman_utils.util.flipCoin

        Args:
            state: the current state

        Returns:
            The action to take
        """
        # Extract features from the current state.
        stateFeatures = GameStateFeatures(state)
        legal = stateFeatures.legal

        # If it's the first action, randomly choose an action.
        if self.prev_action is None:
            self.prev_action = random.choice(legal)
            self.prev_state = state
            return self.prev_action

        # Extract features from the previous state.
        old_stateFeatures = GameStateFeatures(self.prev_state)

        # Compute reward for the transition between previous state and current state.
        reward = self.computeReward(self.prev_state, state)

        # Update the Q-values based on the learning algorithm.
        self.learn(old_stateFeatures, self.prev_action, reward, stateFeatures)

        # Determine the maximum Q-value and corresponding actions for the current state.
        max_q = self.maxQValue(stateFeatures)
        max_actions = [action for action in legal if
                       self.getQValue(stateFeatures, action) == max_q]

        # Choose action greedily or explore with a probability.
        greedyAction = random.choice(max_actions)
        greedyUtility = self.getQValue(stateFeatures, greedyAction)
        greedyCount = self.getCount(stateFeatures, greedyAction)
        exploreUtility = self.explorationFn(greedyUtility, greedyCount)

        # Decide whether to explore or exploit based on the exploration utility.
        if exploreUtility > greedyUtility:
            self.prev_action = random.choice(legal)
        else:
            self.prev_action = greedyAction

        # Update the previous state and return the chosen action.
        self.prev_state = state
        return self.prev_action

    def final(self, state: GameState):
        """Handle the end of episodes. This is called by the game after a win or a loss."""

        # Compute the reward for the transition between the previous state and the current state.
        reward = self.computeReward(self.prev_state, state)

        # Update Q-values based on the learning algorithm with the computed reward.
        self.learn(GameStateFeatures(self.prev_state), self.prev_action, reward,
                   GameStateFeatures(state))

        # Reset previous action and state to None.
        self.prev_action = None
        self.prev_state = None
        print(f"Game {self.getEpisodesSoFar()} just ended!")

        # Keep track of the number of games played, and set learning
        # parameters to zero when we are done with the pre-set number
        # of training episodes
        self.incrementEpisodesSoFar()
        if self.getEpisodesSoFar() == self.getNumTraining():
            msg = 'Training Done (turning off epsilon and alpha)'
            print('%s\n%s' % (msg, '-' * len(msg)))
            self.setAlpha(0)
            self.setEpsilon(0)
