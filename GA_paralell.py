# parallel_runner.py
import concurrent.futures
import subprocess

def run_script(script_name):
    subprocess.run(["python", script_name])

if __name__ == "__main__":
    # Specify the number of processes/cores to use
    num_processes = 4

    # Create a list of script names (adjust as needed)
    script_names = ["your_script.py"] * num_processes

    # Run scripts in parallel
    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.map(run_script, script_names)
