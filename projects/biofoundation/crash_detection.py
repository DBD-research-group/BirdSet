import sys
import subprocess
import threading
import time

# Define the timeout in seconds
TIMEOUT = 60

# Get all arguments from the caller script (including the script name itself)
args = sys.argv[1:]

# Function to run the subprocess
def run_script():
    return subprocess.Popen(['python'] + args)

# Function to terminate the process if it exceeds the timeout
def terminate_process(proc):
    proc.terminate()
    print("Process terminated due to timeout.")

# Create the process
process = run_script()

# Create a timer to terminate the process if it exceeds TIMEOUT seconds
timer = threading.Timer(TIMEOUT, terminate_process, args=[process])

# Start the timer
timer.start()

# Poll the process to check if it finishes
while process.poll() is None:
    time.sleep(1)  # Check every second if the process has finished

# Process completed before the timeout, cancel the timer
timer.cancel()

# Wait for the process to clean up and finish
process.wait()

# Optionally, check the process return code if needed
print(f"Process finished with return code {process.returncode}")