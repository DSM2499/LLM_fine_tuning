import json
from collections import deque

class TaskLogger:
    def __init__(self, max_len = 5):
        self.buffer = deque(maxlen = max_len)
        self.introspection_callback = None

    def register_introspection_callback(self, callback_fn):
        self.introspection_callback = callback_fn
    
    def log_task(self, prompt: str, response: str):
        self.buffer.append({"prompt": prompt, "response": response})
        print(f"[LOG] Added task {len(self.buffer)}/5")

        if len(self.buffer) == self.buffer.maxlen:
            self.introspection_callback(list(self.buffer))
    
    def export_log(self, filepath = "task_history.json"):
        with open(filepath, "w") as f:
            json.dump(list(self.buffer), f, indent = 4)
    
    def clear(self):
        self.buffer.clear()
        print("[LOG] Cleared task history")