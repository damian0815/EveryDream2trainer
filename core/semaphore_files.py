import os

WANT_SAMPLES_SEMAPHORE_FILE = 'i_want_samples.semaphore'
WANT_SAMPLES_OTHEREMA_SEMAPHORE_FILE = 'i_want_samples_otherema.semaphore'
WANT_VALIDATION_SEMAPHORE_FILE = 'i_want_validation.semaphore'
INTERRUPT_SAMPLES_SEMAPHORE_FILE = 'no_more_samples.semaphore'
INTERRUPT_VALIDATION_SEMAPHORE_FILE = 'no_more_validation.semaphore'
STOP_SEMAPHORE_FILE = 'stop.semaphore'
SAVE_FULL_SEMAPHORE_FILE = 'save_full.semaphore'
SAVE_FULL_AND_STOP_SEMAPHORE_FILE = 'save_full_and_stop.semaphore'
SAVE_FULL_WITH_OPTIMIZER_SEMAPHORE_FILE = 'save_full_with_optimizer.semaphore'
SAVE_FULL_WITH_OPTIMIZER_AND_STOP_SEMAPHORE_FILE = 'save_full_with_optimizer_and_stop.semaphore'
PAUSE_TRAINING_SEMAPHORE_FILE = 'pause_training.semaphore'
RESUME_TRAINING_SEMAPHORE_FILE = 'resume_training.semaphore'

def check_semaphore_file_and_unlink(semaphore_file: str) -> bool:
    try:
        os.unlink(semaphore_file)
        return True
    except FileNotFoundError:
        return False

