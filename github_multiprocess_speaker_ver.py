import os
import shutil
import nemo.collections.asr as nemo_asr
import torch
import numpy as np
import soundfile as sf
import librosa
import logging
from multiprocessing import Pool, Manager, Process
from pathlib import Path

def configure_logging():
    logger = logging.getLogger('NeMoProcessing')
    # Check if handlers are already configured (e.g., in a multiprocessing scenario)
    if not logger.handlers:
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        # Set the basic configuration for the root logger to write to file
        logging.basicConfig(level=logging.DEBUG, format=log_format, filename='/home/mb/gitclones/NeMo/log.txt')

        # Add stdout handler to 'NeMoProcessing' logger
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(log_format)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

# Configure logging at the module level so it's configured regardless of entry point
configure_logging()
logger = logging.getLogger('NeMoProcessing')



# To ensure thread-safe file writing
device = "cuda" if torch.cuda.is_available() else "cpu"

logger = logging.getLogger('NeMoProcessing')

# List of character names
# Function to read lines from file
# Function to read lines from a file
def read_lines_from_file(file_path, start_line=0, end_line=None):
    with open(file_path, "r") as file:
        lines = file.readlines()[start_line:end_line] if end_line else file.readlines()
    return [line.strip() for line in lines]

def refreshlinepath():
    directory_path = "/mnt/8tbwd/star_trek_tng/uvr/uvred/downsampled/"
    test_line_paths = [Path(os.path.join(directory_path, test_line)) for test_line in os.listdir(directory_path)]
    test_lines = test_line_paths
    return test_lines



def process_test_line(test_line, char_line_path, speaker_model, char_folder, results_file_path, lock):
    logger = logging.getLogger('NeMoProcessing')
    try:
        speakers_match = speaker_model.verify_speakers(str(test_line), str(char_line_path))
        if speakers_match:
            destination_path = char_folder / test_line.name
            shutil.move(str(test_line), str(destination_path))
            refreshlinepath()
            logger.info(f"File moved: {test_line} to {destination_path}")
            with lock:
                with open(results_file_path, "a") as results_file:
                    results_file.write(f"Match Found: {test_line} -> {destination_path}\n")
        else:
            logger.debug(f"No match found between: {test_line} and {char_line_path}")
    except Exception as e:
        logger.error(f"Error processing comparison: {e}")

def process_chunk(args):
    char, chunk, test_lines, results_file_path, speaker_model, lock = args
    char_folder = Path(f"/home/mb/gitclones/NeMo/copy/{char}/")
    char_folder.mkdir(parents=True, exist_ok=True)

    test_lines = refreshlinepath()
    for char_line in chunk:
        char_line_path = Path(char_line)
        for test_line in test_lines:
            process_test_line(test_line, char_line_path, speaker_model, char_folder, results_file_path, lock)
            refreshlinepath()
        refreshlinepath()

def process_char_lines(args):
    char, test_lines, results_file_path, lock = args
    test_lines = refreshlinepath()
    speaker_model =  nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained("nvidia/speakerverification_en_titanet_large")
    char_file_path = Path(f"/home/mb/gitclones/NeMo/copy/{char}.txt")
    char_lines = read_lines_from_file(char_file_path)
    chunk_size = 10
    chunks = [char_lines[i:i + chunk_size] for i in range(0, len(char_lines), chunk_size)]
    for chunk in chunks:
        refreshlinepath()
        process_chunk((char, chunk, test_lines, results_file_path, speaker_model, lock))

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    chars = ["computer", "geordi", "obrian", "worf", "picard", "riker", "troi", "crusher"]
    test_lines = refreshlinepath()
    results_file_path = "/home/mb/gitclones/NeMo/copy/results.txt"

    # Initialize logger, speaker_model, and other prerequisites
    # The speaker_model variable and other dependencies would need to be defined and initialized here

    # Example setup for multiprocessing
    with Manager() as manager:
        multiprocessing.set_start_method('spawn', force = True)
        lock = manager.Lock()
        tasks = [(char, test_lines, results_file_path, lock) for char in chars]

        with Pool(os.cpu_count()) as pool:
            pool.map(process_char_lines, tasks)

    logger = logging.getLogger('NeMoProcessing')
    logger.info("Processing completed.")
