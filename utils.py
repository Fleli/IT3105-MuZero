import os
from datetime import datetime


class Logger:
    def __init__(self):
        logs_dir = "logs"
        os.makedirs(logs_dir, exist_ok=True)
        date_str = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
        self.log_filename = os.path.join(logs_dir, f"{date_str}.log")

    def log(self, content: str, force=False):
        if force:
            try:
                with open(self.log_filename, 'a') as f:
                    f.write(f"{content}\n")
            except Exception as e:
                print(f"Failed to write to log file: {e}")
