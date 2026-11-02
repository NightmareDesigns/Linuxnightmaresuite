#!/usr/bin/env python3
"""
tools_explorer_13.py — Browse and launch every Nightmare Designs component.
Pure tkinter. No outside code.
"""
import json
import os
import subprocess
import tkinter as tk
from pathlib import Path
from tkinter import ttk

MASTER_DIR = Path(__file__).resolve().parent
CATALOG_FILE = MASTER_DIR / "tools_catalog_13.json"


def load_catalog():
    if CATALOG_FILE.is_file():
        return json.loads(CATALOG_FILE.read_text(encoding="utf-8"))
    return {"categories": {}}


class ToolsExplorer:
    def __init__(self, root):
        self.root = root
        self.root.title("Nightmare Tools Explorer 13")
        self.root.geometry("900x700")
        self.root.configure(bg="#050505")
        self.catalog = load_catalog()
        self.token = self._load_token()

        self.colors = {
            "bg": "#050505",
            "panel": "#0a0a0a",
            "matrix": "#00ff41",
            "matrix_dim": "#008f11",
            "blood": "#8b0000",
            "text": "#e0e0e0",
            "muted": "#888888",
        }

        tk.Label(
            self.root,
            text="NIGHTMARE TOOLS EXPLORER 13",
            font=("Courier New", 18, "bold"),
            fg=self.colors["matrix"],
            bg=self.colors["bg"],
        ).pack(pady=10)

        tk.Label(
            self.root,
            text="Every component we built. One click to launch.",
            font=("Courier New", 10),
            fg=self.colors["muted"],
            bg=self.colors["bg"],
        ).pack()

        self.tree = ttk.Treeview(self.root)
        self.tree.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        self.tree.heading("#0", text="Component", anchor=tk.W)

        style = ttk.Style()
        style.theme_use("default")
        style.configure("Treeview", background="#0a0a0a", foreground="#00ff41",
                        fieldbackground="#0a0a0a")
        style.configure("Treeview.Heading", background="#1a1a1a", foreground="#00ff41")

        self._populate_tree()

        btn_frame = tk.Frame(self.root, bg=self.colors["bg"])
        btn_frame.pack(fill=tk.X, padx=20, pady=10)
        tk.Button(
            btn_frame,
            text="Launch Selected",
            command=self._launch_selected,
            bg=self.colors["panel"],
            fg=self.colors["matrix"],
            activebackground=self.colors["matrix_dim"],
            font=("Courier New", 10, "bold"),
        ).pack(side=tk.LEFT, padx=5)
        tk.Button(
            btn_frame,
            text="Open Folder",
            command=self._open_folder,
            bg=self.colors["panel"],
            fg=self.colors["matrix"],
            activebackground=self.colors["matrix_dim"],
            font=("Courier New", 10),
        ).pack(side=tk.LEFT, padx=5)

        self.detail = tk.Text(self.root, height=6, bg="#000000", fg=self.colors["matrix"],
                              font=("Courier New", 9), wrap=tk.WORD)
        self.detail.pack(fill=tk.X, padx=20, pady=5)

        self.tree.bind("<<TreeviewSelect>>", self._on_select)

    def _load_token(self):
        tf = MASTER_DIR / ".admin_token"
        if tf.is_file():
            return tf.read_text(encoding="utf-8").strip()
        return ""

    def _populate_tree(self):
        for cat_name, items in self.catalog.get("categories", {}).items():
            cat_node = self.tree.insert("", tk.END, text=cat_name, open=False)
            for item in items:
                self.tree.insert(cat_node, tk.END, text=item["name"], values=(json.dumps(item),))

    def _on_select(self, event):
        sel = self.tree.selection()
        if not sel:
            return
        item = self.tree.item(sel[0])
        values = item.get("values")
        if not values:
            self.detail.delete("1.0", tk.END)
            return
        data = json.loads(values[0])
        self.detail.delete("1.0", tk.END)
        self.detail.insert(tk.END, f"{data['name']}\n")
        self.detail.insert(tk.END, f"{data.get('desc', '')}\n")
        self.detail.insert(tk.END, f"Path: {data.get('path', '')}\n")
        self.detail.insert(tk.END, f"Script: {data.get('script', 'N/A')}\n")
        self.detail.insert(tk.END, f"Type: {data.get('type', 'unknown')}\n")

    def _launch_selected(self):
        sel = self.tree.selection()
        if not sel:
            return
        item = self.tree.item(sel[0])
        values = item.get("values")
        if not values:
            return
        data = json.loads(values[0])
        self._launch(data)

    def _launch(self, data):
        path = Path(data["path"])
        script = data.get("script", "")
        args = data.get("args", "")
        typ = data.get("type", "python")

        env = os.environ.copy()
        env["NTMRE_ADMIN_TOKEN"] = self.token
        env["PYTHONUTF8"] = "1"

        if typ == "folder":
            subprocess.Popen(f'explorer "{path}"', shell=True)
            return

        if typ == "node":
            cmd = f"node {script} {args}"
        else:
            cmd = f"python {script} {args}"

        subprocess.Popen(cmd, cwd=str(path), env=env, shell=True,
                        creationflags=subprocess.CREATE_NEW_CONSOLE)

    def _open_folder(self):
        sel = self.tree.selection()
        if not sel:
            return
        item = self.tree.item(sel[0])
        values = item.get("values")
        if not values:
            return
        data = json.loads(values[0])
        path = Path(data["path"])
        subprocess.Popen(f'explorer "{path}"', shell=True)


if __name__ == "__main__":
    root = tk.Tk()
    app = ToolsExplorer(root)
    root.mainloop()
