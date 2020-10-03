import random
from collections import defaultdict
import numpy as np
from utils import vector_add, orientations, turn_right, turn_left, print_table

class MDP:
    """A Markov Decision Process, defined by an initial state, transition model,
    and reward function. We also keep track of a gamma value, for use by
    algorithms. The transition model is represented somewhat differently from
    the text. Instead of P(s' | s, a) being a probability number for each
    state/state/action triplet, we instead have T(s, a) return a
    list of (p, s') pairs. We also keep track of the possible states,
    terminal states, and actions for each state."""

    def __init__(self, init, actlist, terminals, transitions=None, reward=None, states=None, gamma=0.9):
        if not (0 < gamma <= 1):
            raise ValueError("An MDP must have 0 < gamma <= 1")

        # collect states from transitions table if not passed.
        self.states = states or self.get_states_from_transitions(transitions)

        self.init = init

        if isinstance(actlist, list):
            # if actlist is a list, all states have the same actions
            self.actlist = actlist

        elif isinstance(actlist, dict):
            # if actlist is a dict, different actions for each state
            self.actlist = actlist

        self.terminals = terminals
        self.transitions = transitions or {}
        if not self.transitions:
            print("Warning: Transition table is empty.")

        self.gamma = gamma

        self.reward = reward or {s: 0 for s in self.states}

        # self.check_consistency()

    def R(self, state):
        """Return a numeric reward for this state."""

        return self.reward[state]

    def T(self, state, action):
        """Transition model. From a state and an action, return a list
        of (probability, result-state) pairs."""

        if not self.transitions:
            raise ValueError("Transition model is missing")
        else:
            return self.transitions[state][action]

    def actions(self, state):
        """Return a list of actions that can be performed in this state. By default, a
        fixed list of actions, except for terminal states. Override this
        method if you need to specialize by state."""

        if state in self.terminals:
            return [None]
        else:
            return self.actlist

    def get_states_from_transitions(self, transitions):
        if isinstance(transitions, dict):
            s1 = set(transitions.keys())
            s2 = set(tr[1] for actions in transitions.values()
                     for effects in actions.values()
                     for tr in effects)
            return s1.union(s2)
        else:
            print('Could not retrieve states from transitions')
            return None

    def check_consistency(self):

        # check that all states in transitions are valid
        assert set(self.states) == self.get_states_from_transitions(self.transitions)

        # check that init is a valid state
        assert self.init in self.states

        # check reward for each state
        assert set(self.reward.keys()) == set(self.states)

        # check that all terminals are valid states
        assert all(t in self.states for t in self.terminals)

        # check that probability distributions for all actions sum to 1
        for s1, actions in self.transitions.items():
            for a in actions.keys():
                s = 0
                for o in actions[a]:
                    s += o[0]
                assert abs(s - 1) < 0.001

class MDP2(MDP):
    """
    Inherits from MDP. Handles terminal states, and transitions to and from terminal states better.
    """

    def __init__(self, init, actlist, terminals, transitions, reward=None, gamma=0.9):
        MDP.__init__(self, init, actlist, terminals, transitions, reward, gamma=gamma)

    def T(self, state, action):
        if action is None:
            return [(0.0, state)]
        else:
            return self.transitions[state][action]

# Transition Matrix as nested dict. State -> Actions in state -> List of (Probability, State) tuples
t = {
    "A": {
            "X": [(0.3, "A"), (0.7, "B")],
            "Y": [(1.0, "A")]
         },
    "B": {
            "X": {(0.8, "End"), (0.2, "B")},
            "Y": {(1.0, "A")}
         },
    "End": {}
}

init = "A"

terminals = ["End"]

rewards = {
    "A": 5,
    "B": -10,
    "End": 100
}

class CustomMDP(MDP):
    def __init__(self, init, terminals, transition_matrix, reward = None, gamma=.9):
        # All possible actions.
        actlist = []
        for state in transition_matrix.keys():
            actlist.extend(transition_matrix[state])
        actlist = list(set(actlist))
        MDP.__init__(self, init, actlist, terminals, transition_matrix, reward, gamma=gamma)

    def T(self, state, action):
        if action is None:
            return [(0.0, state)]
        else: 
            return self.t[state][action]

our_mdp = CustomMDP(init, terminals, t, rewards, gamma=.9)

class GridMDP(MDP):
    """A two-dimensional grid MDP. All you have to do is
    specify the grid as a list of lists of rewards; use None for an obstacle
    (unreachable state). Also, you should specify the terminal states.
    An action is an (x, y) unit vector; e.g. (1, 0) means move east."""

    def __init__(self, grid, terminals, init=(0, 0), gamma=.9):
        grid.reverse()  # because we want row 0 on bottom, not on top
        reward = {}
        states = set()
        self.rows = len(grid)
        self.cols = len(grid[0])
        self.grid = grid
        for x in range(self.cols):
            for y in range(self.rows):
                if grid[y][x]:
                    states.add((x, y))
                    reward[(x, y)] = grid[y][x]
        self.states = states
        actlist = orientations
        transitions = {}
        for s in states:
            transitions[s] = {}
            for a in actlist:
                transitions[s][a] = self.calculate_T(s, a)
        MDP.__init__(self, init, actlist=actlist,
                     terminals=terminals, transitions=transitions,
                     reward=reward, states=states, gamma=gamma)

    def calculate_T(self, state, action):
        if action:
            return [(0.8, self.go(state, action)),
                    (0.1, self.go(state, turn_right(action))),
                    (0.1, self.go(state, turn_left(action)))]
        else:
            return [(0.0, state)]

    def T(self, state, action):
        return self.transitions[state][action] if action else [(0.0, state)]

    def go(self, state, direction):
        """Return the state that results from going in this direction."""

        state1 = vector_add(state, direction)
        return state1 if state1 in self.states else state

    def to_grid(self, mapping):
        """Convert a mapping from (x, y) to v into a [[..., v, ...]] grid."""

        return list(reversed([[mapping.get((x, y), None)
                               for x in range(self.cols)]
                              for y in range(self.rows)]))

    def to_arrows(self, policy):
        chars = {(1, 0): '>', (0, 1): '^', (-1, 0): '<', (0, -1): 'v', None: '.'}
        return self.to_grid({s: chars[a] for (s, a) in policy.items()})

def value_iteration(mdp, epsilon=0.001):
    """Solving an MDP by value iteration."""

    U1 = {s: 0 for s in mdp.states}
    R, T, gamma = mdp.R, mdp.T, mdp.gamma
    while True:
        U = U1.copy()
        delta = 0
        for s in mdp.states:
            U1[s] = R(s) + gamma * max(sum(p * U[s1] for (p, s1) in T(s, a))
                                       for a in mdp.actions(s))
            delta = max(delta, abs(U1[s] - U[s]))
        if delta <= epsilon * (1 - gamma) / gamma:
            return U

def best_policy(mdp, U):
    """Given an MDP and a utility function U, determine the best policy,
    as a mapping from state to action."""

    pi = {}
    for s in mdp.states:
        pi[s] = max(mdp.actions(s), key=lambda a: expected_utility(a, s, U, mdp))
    return pi

def expected_utility(a, s, U, mdp):
    """The expected utility of doing a in state s, according to the MDP and U."""

    return sum(p * U[s1] for (p, s1) in mdp.T(s, a))

def policy_iteration(mdp):
    """Solve an MDP by policy iteration """

    U = {s: 0 for s in mdp.states}
    pi = {s: random.choice(mdp.actions(s)) for s in mdp.states}
    while True:
        U = policy_evaluation(pi, U, mdp)
        unchanged = True
        for s in mdp.states:
            a = max(mdp.actions(s), key=lambda a: expected_utility(a, s, U, mdp))
            if a != pi[s]:
                pi[s] = a
                unchanged = False
        if unchanged:
            return pi

def policy_evaluation(pi, U, mdp, k=20):
    """Return an updated utility mapping U from each state in the MDP to its
    utility, using an approximation (modified policy iteration)."""

    R, T, gamma = mdp.R, mdp.T, mdp.gamma
    for i in range(k):
        for s in mdp.states:
            U[s] = R(s) + gamma * sum(p * U[s1] for (p, s1) in T(s, pi[s]))
    return U

def T(self, state, action):
    if action is None:
        return [(0.0, state)]
    else:
        return self.transitions[state][action]

def to_arrows(self, policy):
        chars = {
            (1, 0): '>', (0, 1): '^', (-1, 0): '<', (0, -1): 'v', None: '.'}
        return self.to_grid({s: chars[a] for (s, a) in policy.items()})

def to_grid(self, mapping):
        """Convert a mapping from (x, y) to v into a [[..., v, ...]] grid."""
        return list(reversed([[mapping.get((x, y), None)
                               for x in range(self.cols)]
                              for y in range(self.rows)]))

#--------------------criando os mundos--------------------#

sequential_decision_environment = GridMDP([[-0.04, -0.04, -0.04, +1],
                                           [-0.04, None, -0.04, -1],
                                           [-0.04, -0.04, -0.04, -0.04]],
                                          terminals=[(3, 2), (3, 1)])

sequential_decision_environment2 = GridMDP([[-0.04, -0.04, -0.04, -0.04, +1],
                                           [-0.04, -0.04, -0.04, -0.4, -1],
                                           [-0.04, -0.04, None, -0.04, -0.04],
                                           [-0.04, -0.04, -0.04, -0.04, -0.04],
                                           [-0.04, -0.04, -0.04, -0.04, -0.04]],
                                          terminals=[(4, 4), (4, 3)])

sequential_decision_environment3 = GridMDP([[-0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04, +1],
                                           [-0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -1],
                                           [-0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04],
                                           [-0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04],
                                           [-0.04, -0.04, -0.04, -0.04, None, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04],
                                           [-0.04, -0.04, -0.04, -0.04, None, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04],
                                           [-0.04, -0.04, -0.04, -0.04, None, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04],
                                           [-0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04],
                                           [-0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04],
                                           [-0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04],
                                           [-0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04]],
                                          terminals=[(10, 10), (10, 9)])

#--------------------executando primeiro mundo--------------------#

print("Iteracao de valor do mundo 4x3 ((estado): utilidade do estado):")
print(value_iteration(sequential_decision_environment))

print("Iteracao de politica (mundo 4x3):")
pi = best_policy(sequential_decision_environment, value_iteration(sequential_decision_environment, .001))
print_table(sequential_decision_environment.to_arrows(pi))

valueIteration = policy_iteration(sequential_decision_environment)
print(valueIteration)

#Setando estado inicial#
initialState = 0,2

currentState = initialState

#Recompensa total#
rTotal = []

#Pegando caminho#
while True:
    print("Estado atual:", currentState, "|| Ir para direcao:", valueIteration.get(currentState), "|| Recompensa:", sequential_decision_environment.R(currentState))
    rTotal.append(sequential_decision_environment.R(currentState))
    currentState = sequential_decision_environment.go(currentState, valueIteration.get(currentState))

    #Se chegar no estado terminal, parar#
    if currentState in sequential_decision_environment.terminals or currentState == (3,2):
        break

total = sum(rTotal)
totalReal = total + 1

print("Recompensa total (mundo 4x3):", totalReal)

#--------------------executando segundo mundo--------------------#

print("Iteracao de valor do mundo 5x5 ((estado): utilidade do estado):")
print(value_iteration(sequential_decision_environment2))

print("Iteracao de politica (mundo 5x5):")
pi2 = best_policy(sequential_decision_environment2, value_iteration(sequential_decision_environment2, .001))
print_table(sequential_decision_environment2.to_arrows(pi2))

valueIteration2 = policy_iteration(sequential_decision_environment2)
print(valueIteration2)

#Setando estado inicial#
initialState2 = 0,3
currentState2 = initialState2

#Recompensa total#
rTotal2 = []

#Pegando caminho#
while True:
    print("Estado atual:", currentState2, "|| Ir para direcao:", valueIteration2.get(currentState2), "|| Recompensa:", sequential_decision_environment2.R(currentState2))
    rTotal2.append(sequential_decision_environment2.R(currentState2))
    currentState2 = sequential_decision_environment2.go(currentState2, valueIteration2.get(currentState2))

    #Se chegar no estado terminal, parar#
    if currentState2 in sequential_decision_environment2.terminals or currentState2 == (4,4):
        break

total2 = sum(rTotal2)
totalReal2 = total2 + 1
    
print("Recompensa total (mundo 5x5):", totalReal2)

#--------------------executando terceiro mundo--------------------#

print("Iteracao de valor do mundo 11x11 ((estado): utilidade do estado):")
print(value_iteration(sequential_decision_environment3))

print("Iteracao de politica (mundo 11x11):")
pi3 = best_policy(sequential_decision_environment3, value_iteration(sequential_decision_environment3, .001))
print_table(sequential_decision_environment3.to_arrows(pi3))

valueIteration3 = policy_iteration(sequential_decision_environment3)
print(valueIteration3)

#Setando estado inicial#
initialState3 = 0,9
currentState3 = initialState3

#Recompensa total#
rTotal3 = []

#Pegando caminho#
while True:
    
    print("Estado atual:", currentState3, "|| Ir para direcao:", valueIteration3.get(currentState3), "|| Recompensa:", sequential_decision_environment3.R(currentState3))
    rTotal3.append(sequential_decision_environment3.R(currentState3))
    currentState3 = sequential_decision_environment3.go(currentState3, valueIteration3.get(currentState3))

    #Se chegar no estado terminal, parar#
    if currentState3 in sequential_decision_environment3.terminals or currentState3 == (10,10):
        break

total3 = sum(rTotal3)
totalReal3 = total3 + 1
    
print("Recompensa total (mundo 11x11):", totalReal3)