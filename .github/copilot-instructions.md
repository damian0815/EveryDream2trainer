# Writing Plans

When writing a plan, unless otherwise instructed ensure that the plan can be followed with a high probability of success by a junior developer. This means that the plan should 
* be detailed, 
* be unambiguous,
* not assume any prior knowledge of the problem domain,
* not assume solid understanding of software engineering principles and best practises.

# Python venv

To activate a python environment, use `conda activate everydream2trainer` but note that no CUDA GPU is available (only MPS).

# Code Quality

## Basic principles

Don't repeat yourself. Single responsibility principle.

## Clean Code

Code should be easy to read, and easy to understand. It should always be possible to tell at a glance whether the logic at the current level of abstraction is obviously doing the right thing (or the wrong thing).

### Names

Names should be *meaningful*:
intention revealing
pronouncable
searchable
Additionally:
class names should be *nouns* or *noun phrases*
function names should be *verbs* or *verb phrases*
Pick one term for one abstract concept and stick to it.

### Functions

Functions should be short, have limited or no side effects, and do one thing.

Functions should be arranged in a source file in “step down” order: functions at the top of the file outline the high level process; functions below that break each part into subparts; parts and subparts live near to each other.

### Hungarian notation

Where appropriate use "Apps Hungarian” with suffixes to denote data state - cf coined by Joel Spolsky's 2005 essay "Making Wrong Code Look Wrong”.

## Testing

Write unit tests where logic can get hairy and subtle. Always write integration tests.
