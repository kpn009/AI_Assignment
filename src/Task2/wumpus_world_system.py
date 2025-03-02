import numpy as np
import random
import seaborn as sns
import matplotlib.pyplot as plt
from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

class WumpusWorldAgent:
    def __init__(self, n):
        self.n = n
        self.grid = np.zeros((n, n), dtype=int)
        self.probability_matrix = np.full((n, n), 0.2)  # Initial pit probability
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

    def get_neighbors(self, x, y):
        return [(x+dx, y+dy) for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)] if 0 <= x+dx < self.n and 0 <= y+dy < self.n]

    def choose_next_move(self, best_move=True):
        x, y = self.position
        neighbors = self.get_neighbors(x, y)
        safe_moves = [(nx, ny) for nx, ny in neighbors if not self.visited[nx, ny]]
        if not safe_moves:
            return None
        return min(safe_moves, key=lambda pos: self.probability_matrix[pos]) if best_move else random.choice(safe_moves)

    def update_probabilities(self):
        x, y = self.position
        for nx, ny in self.get_neighbors(x, y):
            if not self.visited[nx, ny]:
                self.probability_matrix[nx, ny] *= 1.2 if (nx, ny) in self.pit_positions else 0.8

    def move(self):
        self.visited[self.position] = True
        self.path.append(self.position)
        self.update_probabilities()

        plt.figure(figsize=(6,6))
        sns.heatmap(self.probability_matrix, annot=True, cmap='coolwarm')
        plt.title(f"Move {self.move_count}")
        plt.savefig(f"move_{self.move_count}.png")
        plt.close()
        self.move_count += 1

        while True:
            choice = input("Enter 'B' for best move or 'R' for random move: ").strip().upper()
            if choice in ['B', 'R']:
                best_move = choice == 'B'
                break
            print("Invalid choice. Please enter 'B' or 'R'.")

        next_move = self.choose_next_move(best_move)
        if next_move:
            self.position = next_move
        else:
            print("No safe move found. Backtracking...")
            while self.path:
                last_position = self.path.pop()
                if self.path:
                    self.position = self.path[-1]
                else:
                    self.position = (0, 0)
                if self.choose_next_move():
                    break

    def find_gold(self):
        while self.position != self.gold_pos:
            self.move()
        print("Gold found at:", self.position)
        print("Path taken:", self.path)

if __name__ == "__main__":
    n = int(input("Enter grid size: "))
    agent = WumpusWorldAgent(n)
    agent.find_gold()
