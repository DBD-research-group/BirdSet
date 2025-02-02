import sys
import subprocess
import threading
import time

# Define the initial timeout and the increment for retries
INITIAL_TIMEOUT = 8_000  
TIMEOUT_INCREMENT = 600  # 10 minutes

# Define the maximum number of retries
MAX_RETRIES = 3

# Get all arguments from the caller script (including the script name itself)
args = sys.argv[1:]

def run_script_with_timeout(timeout):
    """Run the script with the specified timeout."""
    # Function to run the subprocess
    def run_script():
        return subprocess.Popen(['python'] + args)

    # Function to terminate the process if it exceeds the timeout
    def terminate_process(proc):
        if proc.poll() is None:  # Ensure the process is still running
            proc.terminate()
            print(f"Process terminated due to timeout (timeout was {timeout} seconds).")
            try:
                stdout, stderr = proc.communicate(timeout=10)  # Timeout after 10 seconds

            except subprocess.TimeoutExpired:
                print("Process did not terminate correctly")
                proc.kill()
                stdout, stderr = proc.communicate()
            
            print(stdout.decode())
            print(stderr.decode())


    # Create the process
    process = run_script()

    # Create a timer to terminate the process if it exceeds the timeout
    timer = threading.Timer(timeout, terminate_process, args=[process])

    # Start the timer
    timer.start()

    # Poll the process to check if it finishes
    while process.poll() is None:
        time.sleep(1)  # Check every second if the process has finished

    # Process completed before the timeout, cancel the timer
    timer.cancel()

    return process.returncode

# Main logic to retry the script with increasing timeouts
timeout = INITIAL_TIMEOUT
for attempt in range(1, MAX_RETRIES + 1):
    if attempt > 1:
        print(f"Attempt {attempt} with timeout set to {timeout} seconds.")

    return_code = run_script_with_timeout(timeout)

    if return_code == 0:
        print(f"Process finished successfully with return code {return_code}.")
        sys.exit(0)
    else:
        print(f"Process finished with return code {return_code}.")
        if attempt < MAX_RETRIES:
            print("Retrying with a longer timeout...")
            timeout += TIMEOUT_INCREMENT
        else:
            print("Max retries reached. Exiting with failure.")
            sys.exit(1)

# If the GPU is having problems with memory:
# ps aux | grep python (Check for python processes)
# ps aux | grep 'python birdset/train.py' | grep -v grep | awk '{print $2}' | xargs -r kill -9 (Kill the processes)