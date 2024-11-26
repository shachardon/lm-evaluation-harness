import glob
import logging
import os
import re
import signal
import subprocess
import sys
import time

from tqdm import tqdm


# Constants (unchanged)
TASK = "bluebench"
OUTPUT_DIR_NAME = "bluebench"
MEMORY = "30g"
CORES = "4+0"
QUEUE = "x86_24h"
PYTHON_EXECUTABLE = "/dccstor/eval-research/miniforge3/envs/lmeval/bin/python"
OUTPUT_BASE_PATH = (
    f"/dccstor/eval-research/code/lm-evaluation-harness/outputs/{OUTPUT_DIR_NAME}/"
)
MODELS = [
    "ibm/granite-3-8b-instruct",
    "ibm/granite-3-2b-instruct",
    "google/flan-ul2",
    "ibm/granite-13b-chat-v2",
    "ibm/granite-13b-instruct-v2",
    "ibm/granite-20b-multilingual",
    "meta-llama/llama-3-1-70b-instruct",
    "meta-llama/llama-3-1-8b-instruct",
    "meta-llama/llama-3-70b-instruct",
    "meta-llama/llama-3-8b-instruct",
    "mistralai/mistral-large",
    "mistralai/mixtral-8x7b-instruct-v01",
]

# Logging setup
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def sanitize_model_id(model_id):
    return model_id.replace("/", "_").replace(":", "_")


def run_job(model_id):
    sanitized_model_id = sanitize_model_id(model_id)
    output_path = os.path.join(OUTPUT_BASE_PATH, sanitized_model_id)
    results_file_pattern = os.path.join(output_path, "results_*")  # pattern to check

    # Check if results files already exist
    if glob.glob(results_file_pattern):
        logging.info(
            f"Results already exist for {model_id} in {output_path}. Skipping."
        )
        return None  # Indicate job skipped

    command = [
        "jbsub",
        "-mem",
        MEMORY,
        "-cores",
        CORES,
        "-q",
        QUEUE,
        f"HF_HOME=/dccstor/eval-research/hf_cache_{sanitized_model_id}",
        "&&",
        PYTHON_EXECUTABLE,
        "lm_eval",
        "--model",
        "watsonx_llm",
        "--model_args",
        f"model_id={model_id}",
        "--tasks",
        TASK,
        "--output_path",
        output_path,
        "--cache_requests",
        "true",
        "--log_samples",
    ]

    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        logging.info(f"Submitted job for {model_id}. Output will be in {output_path}")

        match = re.search(r"Job <(\d+)>", result.stdout)
        if match:
            job_id = match.group(1)
            logging.info(f"Job ID: {job_id}")
            return job_id
        else:
            logging.warning(f"Could not parse job ID from output: {result.stdout}")
            return None

    except subprocess.CalledProcessError as e:
        logging.error(f"Error submitting job for {model_id}:")
        logging.error(f"Return code: {e.returncode}")
        logging.error(f"Stdout: {e.stdout}")
        logging.error(f"Stderr: {e.stderr}")
        return None


def get_all_job_statuses():
    """Gets the status of all jobs using bjobs."""
    try:
        result = subprocess.run(
            ["bjobs", "-a", "-o", "jobid stat"],
            capture_output=True,
            text=True,
            check=True,
        )
        output = result.stdout.strip()
        all_job_statuses = {}
        for line in output.splitlines()[1:]:  # Skip header
            job_id_str, status = line.split()
            all_job_statuses[job_id_str] = status
        return all_job_statuses
    except subprocess.CalledProcessError as e:
        logging.error(f"Error getting job statuses: {e}")


def monitor_progress(model_ids, job_ids):
    total_jobs = len(model_ids)
    completed_jobs = 0
    failed_jobs = []

    with tqdm(total=total_jobs, desc="Processing Models") as pbar:
        while completed_jobs < total_jobs - len(failed_jobs):  # Exit early on failure
            time.sleep(60)

            all_job_statuses = get_all_job_statuses()
            if all_job_statuses is None:
                break

            for model_id in list(model_ids):
                job_id = job_ids.get(model_id)
                if job_id:
                    status = all_job_statuses.get(str(job_id))
                    if status == "DONE":
                        completed_jobs += 1
                        pbar.update(1)
                        model_ids.remove(model_id)
                        logging.info(
                            f"Model {model_id} (Job {job_id}) completed successfully."
                        )

                    elif status == "EXIT":
                        failed_jobs.append(job_id)
                        model_ids.remove(model_id)
                        logging.error(
                            f"Model {model_id} (Job {job_id}) failed. Check LSF logs."
                        )
                        pbar.update(1)

                    # Ignore other statuses (RUN, PEND, etc.)

            time.sleep(10)  # Adjust as needed

    if failed_jobs:
        logging.error(f"The following jobs failed: {', '.join(map(str, failed_jobs))}")

    return completed_jobs == total_jobs


def signal_handler(sig, frame):
    logging.warning("Experiment interrupted. Exiting.")
    sys.exit(1)


import logging
import logging.handlers


def setup_logging(output_dir):
    """Configures logging to write to a file and the console."""
    log_file = os.path.join(output_dir, "run_bluebench.log")

    # Create the output directory if it doesn't exist.
    os.makedirs(output_dir, exist_ok=True)

    try:
        handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=10 * 1024 * 1024, backupCount=5
        )  # 10MB max size, 5 backups
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logging.getLogger().addHandler(handler)
        logging.getLogger().setLevel(logging.INFO)

    except Exception as e:  # Catch any potential errors
        print(
            f"Error setting up logging: {e}.  Log messages will only be printed to the console."
        )


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    models_to_run = MODELS.copy()
    job_ids = {}

    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_BASE_PATH, exist_ok=True)

    # Set up logging to write to a file within OUTPUT_BASE_PATH
    setup_logging(OUTPUT_BASE_PATH)

    for model in models_to_run:
        job_id = run_job(model)
        if job_id:
            job_ids[model] = job_id

    if not monitor_progress(models_to_run, job_ids):
        logging.error("Some models failed to complete within timeout")

    logging.info("Experiment finished.")
