import os

WANT_SAMPLES_SEMAPHORE_FILE = 'i_want_samples.semaphore'
WANT_VALIDATION_SEMAPHORE_FILE = 'i_want_validation.semaphore'
INTERRUPT_SAMPLES_SEMAPHORE_FILE = 'no_more_samples.semaphore'
SAVE_FULL_SEMAPHORE_FILE = 'save_full.semaphore'
SAVE_FULL_WITH_OPTIMIZER_SEMAPHORE_FILE = 'save_full_with_optimizer.semaphore'

def check_semaphore_file_and_unlink(sempahore_file) -> bool:
    try:
        os.unlink(sempahore_file)
        return True
    except FileNotFoundError:
        return False

