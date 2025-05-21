import tkinter as tk
from gui import EqualizerGUI

def main():
    root = tk.Tk()
    app = EqualizerGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
