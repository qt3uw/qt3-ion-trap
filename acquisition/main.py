import tkinter as tk
from gui import ParticleTrackingGUI

def main():
    root = tk.Tk()
    app = ParticleTrackingGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()