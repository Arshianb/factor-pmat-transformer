import time
import subprocess
from datetime import datetime

def wait_until(target_hour, target_minute):
    while True:
        now = datetime.now()
        print(now)
        if now.hour >= target_hour and now.minute >= target_minute:
            break
        time.sleep(10)

# Target time
target_hour = 6
target_minute = 30

wait_until(target_hour, target_minute)

# Run the command
cmd = "tr -d '\r' < ./train_football.sh | bash"
subprocess.run(["bash", "-c", cmd])

print("Command executed.")
