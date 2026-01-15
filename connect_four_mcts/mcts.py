from copy import deepcopy
import numpy as np
import math 
import random

from connect_four import ConnectFour

class MCTSNode:
    def __init__(self, board: ConnectFour, parent=None, move=None, exploration_param = 1.41):
        self.state = board
        self.parent = parent 
        self.move = move
        self.children = {}
        self.visits = 0
        self.value = 0.0
        self.untried_actions = board.get_legal_moves()
        self.c = exploration_param

    def is_terminal_node(self):
        return self.state.get_winner() != -1 
    
    def expand(self):
        action = self.untried_actions.pop()
        new_board = deepcopy(self.state)
        new_board.make_move(action)
        child = MCTSNode(new_board, parent=self, move=action)
        self.children[action] = child
        return child

    def rollout(self):
        curr = deepcopy(self.state)
        
        for _ in range(42): 
            if curr.get_winner() != -1:
                break
            possible_moves = curr.get_legal_moves()
            if not possible_moves:
                break
            action = random.choice(possible_moves)    
            curr.make_move(action)
        
        return curr.get_winner()
    
    def backpropagate(self, result):
        self.visits += 1
        # assuming mcts is playing as player 1 
        if result == 0: # player 0 wins  
            self.value -= 1
        elif result == 1: # player 1 wins
            self.value += 1
        elif result == 2: # draw 
            self.value += 0.5

        if self.parent:
            self.parent.backpropagate(result)

    def uct_value(self):
        if self.visits == 0:
            return float('inf')
        
        win_rate = self.value / self.visits
        exploration = self.c * math.sqrt(math.log(self.parent.visits) / self.visits)
        return win_rate + exploration
    
class MCTS:
    def __init__(self):
        self.nodes_cache = {}
    
    def search(self, root_node: MCTSNode, iterations=2000, choice_method = "uct"):         
        root_key = root_node.state.get_key()
        if root_key in self.nodes_cache:
            root_node = self.nodes_cache[root_key]
        else:
            root_node = MCTSNode(deepcopy(root_node.state))
            self.nodes_cache[root_key] = root_node

        for _ in range(iterations):
            # SELECTION
            node = root_node
            while not node.untried_actions and node.children:
                node = max(node.children.values(), key=lambda n: n.uct_value())

            # EXPANSION
            if not node.is_terminal_node() and node.untried_actions:
                node = node.expand()
                self.nodes_cache[node.state.get_key()] = node

            # SIMULATION
            reward = node.rollout()

            # BACKPROPAGATION
            node.backpropagate(reward)
        
        if choice_method == "visits":
            best_child = max(root_node.children.values(), key=lambda n: n.visits)
        elif choice_method == "value":
            best_child = max(root_node.children.values(), key=lambda n: n.value)
        elif choice_method == "uct":
            best_child = max(root_node.children.values(), key=lambda n: n.uct_value())
        return best_child.move