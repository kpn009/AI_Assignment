# CSF407_2025_2022A7PS0057H_2021B5A7PS2970H_2022A7PS0109H_2022A7PS1299H
# TicTacToe Game with LLM Agents(Task1)

This project implements a flexible tic-tac-toe game-playing system that supports both automated LLM vs. LLM simulations and interactive LLM vs. Human play. The system is built on a customizable NxN board and simulates moves using simple heuristics.

## Features

- **Customizable Board Size**: Play on any NxN grid.
- **LLM Agent Simulation**: Two simulated agents,named ChatGPT (LLM-1) and Claude (LLM-2), choose moves based on these heuristics:
  - Select a winning move if available.
  - Block the opponent’s winning move.
  - Otherwise, choose a random move.
- **Simulation Mode (LLM vs. LLM)**:
  - Automatically run 500 game simulations.
  - Record outcomes in a JSON file (`Exercise1.json`).
  - Generate a histogram and a detailed binomial distribution plot (`Exercise1.png` and `Exercise1_alternate.png`).
  - **Special Feature**: In case of a draw, the system overrides the result to count as a win for LLM-1 (ChatGPT).
- **Human Interaction Mode (LLM vs. Human)**:
  - Human players input moves using the format `row,col` (e.g., `0,1`).
  - The current board state is printed after every move.
- **Board Image Saving**:
  - After every move (by either an LLM or a human), the current board is saved as an image file (e.g., `tic_tac_toe_move_3.png`).
  - This allows you to visually track the progress of the game.

 - # Wumpus World Agent Simulation

This project implements a simulation of a Wumpus World agent navigating a grid-based environment. The agent's goal is to find gold while avoiding pits and the Wumpus. It uses a Bayesian network to model the risk (pit probabilities) in each cell and makes movement decisions based on updated risk assessments.

## Features

- **Random World Generation**: 
  - Generates an n×n grid world.
  - Randomly places a single Wumpus, a single gold, and multiple pits.
- **Bayesian Network**: 
  - Constructs a Bayesian network to model the probability of pits in each cell.
- **Agent Navigation**:
  - Uses updated probability estimates to choose the safest move.
  - Implements backtracking if no safe move is found.
- **Visualization**:
  - Saves the heatmap images to files (e.g., `best_move_1.png`, `best_move_2.png`, etc.).

# Merged System: Tic Tac Toe & Wumpus World Integration

This project integrates two simulation exercises into one system:

- **Tic Tac Toe Simulation (Exercise 1):**  
  Uses simulated LLM agents (ChatGPT as LLM-1 and Claude as LLM-2) to play Tic Tac Toe. The outcome of each game is used to determine the next action in the Wumpus World.

- **Wumpus World Simulation (Exercise 2):**  
  An agent navigates a grid-based world filled with hazards (pits and a Wumpus) and seeks to find the gold. The agent uses a Bayesian network to model the risk (pit probabilities) in each cell, updating its beliefs as it moves.

## How It Works

For each move in the Wumpus World:
- A Tic Tac Toe game is simulated between two LLM agents.
- **Outcome-Based Action:**
  - If **LLM-1 (ChatGPT)** wins, the agent makes its best move (using the original `move()` method).
  - If **LLM-2 (Claude)** wins, the agent makes a random move (using the new `agent_random_move()` function).
- The agent's action is visualized by generating a heatmap of the risk probabilities, which is saved as an image file.

## Features

- **Automated Decision Making:**  
  The outcome of a Tic Tac Toe game dictates the agent's move in the Wumpus World.
  
- **Risk Visualization:**  
  Each move in the Wumpus World is accompanied by saving a heatmap showing the current risk probabilities (pit likelihoods). The heatmap is stored with the name move_1, move_2 etc.

