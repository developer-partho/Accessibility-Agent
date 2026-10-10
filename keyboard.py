import tkinter as tk
from tkinter import ttk
import pyautogui
import win32gui
import win32api
import win32con


class ProfessionalKeyboard:
    def __init__(self, root):
        self.root = root
        self.root.title("On-Screen Keyboard")
        self.root.geometry("900x400")
        self.root.attributes('-topmost', True)
        self.root.configure(bg='#1e1e1e')
        self.root.resizable(False, False)

        # Style
        self.style = ttk.Style()
        self.style.theme_use('clam')

        self.bg_color = '#1e1e1e'
        self.button_bg = '#2d2d2d'
        self.button_fg = '#ffffff'
        self.button_en = '#008000'
        self.special_button_bg = '#007acc'
        self.danger_button_bg = '#d9534f'
        self.hover_color = '#696969'
        self.display_bg = '#252526'
        self.display_fg = '#00ff00'

        self.style.configure(
            'TButton',
            font=('Arial', 14, 'bold'),
            borderwidth=2,
            relief='flat',
            background=self.button_bg,
            foreground=self.button_fg
        )

        self.style.map(
            'TButton',
            background=[('active', self.hover_color)]
        )

        self.style.configure(
            'Display.TEntry',
            font=('Arial', 24),
            fieldbackground=self.display_bg,
            foreground=self.display_fg,
            padding=10,
            borderwidth=3,
            relief='sunken'
        )

        self.last_caret_pos = pyautogui.position()
        self.previous_window = win32gui.GetForegroundWindow()
        self.caps_lock_on = False
        self.is_dragging = False

        self.create_display()
        self.create_keyboard()

        self.root.bind("<Button-1>", self.start_drag)
        self.root.bind("<B1-Motion>", self.drag_window)

        self.root.after(100, self.track_mouse_clicks)

    def track_mouse_clicks(self):
        try:
            hwnd = win32gui.GetForegroundWindow()
            if hwnd != self.get_hwnd():
                self.last_caret_pos = pyautogui.position()
                self.previous_window = hwnd
        finally:
            self.root.after(100, self.track_mouse_clicks)

    def get_hwnd(self):
        return win32gui.FindWindow(None, self.root.title())

    def create_display(self):
        frame = ttk.Frame(self.root)
        frame.pack(fill="x", pady=(20, 10), padx=20)

        self.display_var = tk.StringVar()
        self.display = ttk.Entry(
            frame,
            textvariable=self.display_var,
            style='Display.TEntry',
            justify="center",
            state="readonly"
        )
        self.display.pack(fill="x")

    def create_keyboard(self):
        frame = ttk.Frame(self.root)
        frame.pack(expand=True, fill="both", padx=20, pady=10)

        keys = [
            ['1','2','3','4','5','6','7','8','9','0'],
            ['Q','W','E','R','T','Y','U','I','O','P'],
            ['A','S','D','F','G','H','J','K','L'],
            ['Z','X','C','V','B','N','M',',','.','?']
        ]

        for row in keys:
            row_frame = ttk.Frame(frame)
            row_frame.pack(expand=True, fill="both", pady=2)

            for key in row:
                ttk.Button(
                    row_frame,
                    text=key,
                    width=6,
                    command=lambda k=key: self.handle_key(k)
                ).pack(side="left", expand=True, fill="both", padx=1)

        special = ttk.Frame(frame)
        special.pack(expand=True, fill="both", pady=2)

        ttk.Button(special, text="Enter", width=15,
                   command=lambda: self.handle_key("Enter")).pack(side="left", expand=True, fill="both", padx=1)

        ttk.Button(special, text="Caps", width=10,
                   command=lambda: self.handle_key("Caps")).pack(side="left", expand=True, fill="both", padx=1)

        ttk.Button(special, text="Space", width=15,
                   command=lambda: self.handle_key("Space")).pack(side="left", expand=True, fill="both", padx=1)

        ttk.Button(special, text="Clear", width=10,
                   command=lambda: self.handle_key("Clear")).pack(side="left", expand=True, fill="both", padx=1)

        ttk.Button(special, text="⌫", width=20,
                   command=lambda: self.handle_key("⌫")).pack(side="left", expand=True, fill="both", padx=1)

    def handle_key(self, key):
        win32gui.SetForegroundWindow(self.previous_window)
        current = self.display_var.get()

        if key == "⌫":
            self.display_var.set(current[:-1])
            pyautogui.press("backspace")
        elif key == "Enter":
            pyautogui.press("enter")
            self.display_var.set("")
        elif key == "Caps":
            self.caps_lock_on = not self.caps_lock_on
        elif key == "Space":
            self.display_var.set(current + " ")
            pyautogui.write(" ")
        elif key == "Clear":
            self.display_var.set("")
            pyautogui.hotkey("ctrl", "a")
            pyautogui.press("delete")
        else:
            char = key.upper() if self.caps_lock_on else key.lower()
            self.display_var.set(current + char)
            pyautogui.write(char)

    def start_drag(self, event):
        self.is_dragging = True
        self.drag_x = event.x
        self.drag_y = event.y

    def drag_window(self, event):
        if self.is_dragging:
            x = self.root.winfo_x() - self.drag_x + event.x
            y = self.root.winfo_y() - self.drag_y + event.y
            self.root.geometry(f"+{x}+{y}")


class KeyboardModule:
    def __init__(self):
        self.root = None
        self.app = None

    def start(self):
        if self.root is not None:
            return

        self.root = tk.Tk()
        self.app = ProfessionalKeyboard(self.root)
        self.root.protocol("WM_DELETE_WINDOW", self.stop)
        self.root.mainloop()

    def stop(self):
        if self.root:
            self.root.destroy()
            self.root = None
            self.app = None


if __name__ == "__main__":
    # Launched as a standalone process by BackendController.open_keyboard so the
    # keyboard gets its OWN Tk main loop (Tkinter must run on the main thread, and
    # the companion already owns one — two loops in one process is unstable).
    KeyboardModule().start()