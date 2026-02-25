import tkinter as tk
from tkinter import scrolledtext, ttk
import difflib

class TextDiffApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Translation Draft Comparator")
        self.root.geometry("1200x700")

        # Set a clean visual theme
        style = ttk.Style()
        if 'clam' in style.theme_names():
            style.theme_use('clam')

        # Toolbar
        toolbar = ttk.Frame(self.root)
        toolbar.pack(fill=tk.X, padx=10, pady=5)

        compare_btn = ttk.Button(toolbar, text="Compare Texts", command=self.compare)
        compare_btn.pack(side=tk.LEFT)

        clear_btn = ttk.Button(toolbar, text="Clear", command=self.clear)
        clear_btn.pack(side=tk.LEFT, padx=10)

        # Legend
        legend = tk.Label(toolbar, text="Legend: Red = Deleted | Green = Inserted | Orange = Modified")
        legend.pack(side=tk.RIGHT, padx=10)

        # Main Layout: Side-by-Side PanedWindow
        paned_window = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        paned_window.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left Text Area (Original)
        frame1 = ttk.LabelFrame(paned_window, text="Text 1 (Original Draft)")
        self.text1 = scrolledtext.ScrolledText(frame1, wrap=tk.WORD, font=("Consolas", 11))
        self.text1.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        paned_window.add(frame1, weight=1)

        # Right Text Area (Revised)
        frame2 = ttk.LabelFrame(paned_window, text="Text 2 (Revised Draft)")
        self.text2 = scrolledtext.ScrolledText(frame2, wrap=tk.WORD, font=("Consolas", 11))
        self.text2.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        paned_window.add(frame2, weight=1)

        # Configure highlighting tags
        self.text1.tag_config("delete", background="#ffcccc", foreground="#990000")
        self.text2.tag_config("insert", background="#ccffcc", foreground="#006600")
        self.text1.tag_config("replace", background="#ffe6cc", foreground="#cc6600")
        self.text2.tag_config("replace", background="#ffe6cc", foreground="#cc6600")

    def clear(self):
        """Clears both text boxes and removes tags."""
        self.text1.delete("1.0", tk.END)
        self.text2.delete("1.0", tk.END)

    def compare(self):
        """Executes the diff algorithm and applies color tags."""
        # Clean up previous tags
        for tag in ["delete", "insert", "replace"]:
            self.text1.tag_remove(tag, "1.0", tk.END)
            self.text2.tag_remove(tag, "1.0", tk.END)

        s1 = self.text1.get("1.0", tk.END)
        s2 = self.text2.get("1.0", tk.END)

        # SequenceMatcher finds the longest contiguous matching subsequences
        matcher = difflib.SequenceMatcher(None, s1, s2)
        
        # Apply tags based on diff opcodes
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "replace":
                self.text1.tag_add("replace", f"1.0+{i1}c", f"1.0+{i2}c")
                self.text2.tag_add("replace", f"1.0+{j1}c", f"1.0+{j2}c")
            elif tag == "delete":
                self.text1.tag_add("delete", f"1.0+{i1}c", f"1.0+{i2}c")
            elif tag == "insert":
                self.text2.tag_add("insert", f"1.0+{j1}c", f"1.0+{j2}c")

if __name__ == "__main__":
    root = tk.Tk()
    app = TextDiffApp(root)
    root.mainloop()