"""
Various utilities used throughout the project.
"""

import os
from pathlib import Path
import logging
import subprocess
import time

def run_command_logged(description: str, command: str, log_dir: str, log_file: str):
    """
    Runs a command, logs outputs to a specified directory and file.
    """
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_file)

    print(f"[STARTING] {description}")
    start = time.time()

    try:
        with open(log_path, "a") as f:
            f.write(f"\n{"=" * 60}")
            f.write(f"TIMESTAMP: {time.ctime()}\n")
            f.write(f"DESCRIPTION: {description}\n")
            f.write(f"COMMAND: {command}\n\n")
            f.flush()

            result = subprocess.run(
                command,
                shell=True,
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True
            )

            duration = time.time() - start
            f.write(f"\n[DURATION] {duration:.2f}s | EXIT CODE: {result.returncode}\n")
            f.write(f"\n{"=" * 60}")

        if result.returncode != 0:
            print(f"[FAIL] {description} (code {result.returncode}) in {duration:.2f}s. Check {log_path}")
            return result

        print(f"[FINISHED] {description} in {duration:.2f}s")
        return result

    except Exception as e:
        print(f"[ERROR] {description} failed: {e}")
        raise e

def logged_runner(log_dir: str, log_file: str):
    """
    A factory function that returns a pre-configured command runner.
    """
    def runner(description: str, command: str):
        return run_command_logged(description, command, log_dir, log_file)
    return runner

def run_command(command, error_msg):
    exit_code = os.system(command)
    if exit_code != 0:
        logging.error(f"{error_msg} (exit code {exit_code}). Exiting.")
        exit(exit_code)

def remove_bin_files(path: str):
    for item in Path(path).iterdir():
        if item.is_file() and item.suffix.lower() == '.bin':
            item.unlink()
