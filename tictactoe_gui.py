# tictactoe_gui.py

import tkinter as tk
from tictactoe.game import Game
from tictactoe.agent import Qlearner  # Import your agent if needed

class TicTacToeGUI:
    def __init__(self, root):
        # ... (the GUI code)

def main():
    root = tk.Tk()
    app = TicTacToeGUI(root)
    root.mainloop()

if __name__ == "__play__":
    main()
