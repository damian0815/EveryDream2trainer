import os

_WANT_SAMPLES_SEMAPHORE_FILE = 'i_want_samples.semaphore'
_WANT_VALIDATION_SEMAPHORE_FILE = 'i_want_validation.semaphore'
_INTERRUPT_SAMPLES_SEMAPHORE_FILE = 'no_more_samples.semaphore'

def check_semaphore_file_and_unlink(sempahore_file) -> bool:
    try:
        os.unlink(sempahore_file)
        return True
    except FileNotFoundError:
        return False

