import numpy as np
import random
import seaborn as sns
import matplotlib.pyplot as plt
from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

class TicTacToe:
    def __init__(self, size=3):
        self.size = size
        self.board = [[' ' for _ in range(size)] for _ in range(size)]
        self.moves_made = 0

    def is_valid_move(self, x, y):
        return 0 <= x < self.size and 0 <= y < self.size and self.board[x][y] == ' '

    def make_move(self, x, y, player):
        if self.is_valid_move(x, y):
            self.board[x][y] = player
            self.moves_made += 1
            return True
        return False

    def check_win(self, player):
        for row in self.board:
            if all(cell == player for cell in row):
                return True
        for col in range(self.size):
            if all(self.board[row][col] == player for row in range(self.size)):
                return True
        if all(self.board[i][i] == player for i in range(self.size)):
            return True
        if all(self.board[i][self.size - 1 - i] == player for i in range(self.size)):
                return True
        return False

    def play_game(self):
        players = ['LLM1', 'LLM2']
        for turn in range(self.size * self.size):
            player = players[turn % 2]
            empty_cells = [(i, j) for i in range(self.size) for j in range(self.size) if self.board[i][j] == ' ']
            move = random.choice(empty_cells)
            self.board[move[0]][move[1]] = player
            if self.check_win(player):
                return player
        return 'LLM1'  # Draw is considered a win for LLM1

class WumpusWorldAgent:
    def __init__(self, n, ttt_size):
        self.n = n
        self.ttt_size = ttt_size
        self.grid = np.zeros((n, n), dtype=int)
        self.probability_matrix = np.full((n, n), 0.2)
        self.visited = np.zeros((n, n), dtype=bool)
        self.path = []
        self.position = (0, 0)
        self.move_count = 1
        self.generate_world()
        self.create_bayesian_network()
    
    def generate_world(self):
        cells = [(i, j) for i in range(self.n) for j in range(self.n) if (i, j) != (0, 0)]
        random.shuffle(cells)
        self.wumpus_pos = cells.pop()
        self.gold_pos = cells.pop()
        pit_count = max(1, self.n // 4)
        self.pit_positions = [cells.pop() for _ in range(pit_count)]
    
    def create_bayesian_network(self):
        self.model = BayesianModel()
        for i in range(self.n):
            for j in range(self.n):
                self.model.add_node(f'P_{i}_{j}')
        for i in range(self.n):
            for j in range(self.n):
                neighbors = [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]
                valid_neighbors = [(x, y) for x, y in neighbors if 0 <= x < self.n and 0 <= y < self.n]
                for x, y in valid_neighbors:
                    self.model.add_edge(f'P_{x}_{y}', f'B_{i}_{j}')
        cpd_p = TabularCPD(variable='P_0_0', variable_card=2, values=[[0.8], [0.2]])
        self.model.add_cpds(cpd_p)
    
    def update_probabilities(self):
        x, y = self.position
        for nx, ny in [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]:
            if 0 <= nx < self.n and 0 <= ny < self.n and not self.visited[nx, ny]:
                self.probability_matrix[nx, ny] *= 1.2 if (nx, ny) in self.pit_positions else 0.8
    
    def choose_next_move(self, best_move=True):
        x, y = self.position
        neighbors = [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]
        valid_moves = [(nx, ny) for nx, ny in neighbors if 0 <= nx < self.n and 0 <= ny < self.n and not self.visited[nx, ny]]
        
        if not valid_moves:
            return None
        
        if best_move:
            return min(valid_moves, key=lambda pos: self.probability_matrix[pos])
        return random.choice(valid_moves)
    
    def move(self, best_move=True):
        self.visited[self.position] = True
        move_type = 'B' if best_move else 'R'
        self.path.append((*self.position, move_type))
        print(f"{'Best move selected' if best_move else 'Random move selected'}")
        
        self.update_probabilities()  # Ensure probability matrix updates
        
        plt.figure(figsize=(6,6))
        sns.heatmap(self.probability_matrix, annot=True, cmap='coolwarm')
        plt.title(f"Move {self.move_count}")
        plt.savefig(f"best_move_{self.move_count}.png")
        plt.close()
        self.move_count += 1
        
        next_move = self.choose_next_move(best_move)
        if next_move:
            self.position = next_move
        else:
            print("No safe move found. Backtracking...")
            while self.path:
                last_position = self.path.pop()
                self.position = last_position[:2]
                self.path.append((*self.position, 'BCK'))
                
                next_move = self.choose_next_move(best_move)
                if next_move:
                    self.position = next_move
                    break
    
    def find_gold(self):
        while self.position != self.gold_pos:
            ttt = TicTacToe(self.ttt_size)
            winner = ttt.play_game()
            print(f"Tic-Tac-Toe Winner: {winner}")
            best_move = True if winner == 'LLM1' else False
            self.move(best_move)
        print("Gold found at:", self.position)
        print("Path taken:", self.path)

if __name__ == "__main__":
    ttt_size = int(input("Enter Tic-Tac-Toe board size: "))
    n = int(input("Enter grid size: "))
    agent = WumpusWorldAgent(n, ttt_size)
    agent.find_gold()
