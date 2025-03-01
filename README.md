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

