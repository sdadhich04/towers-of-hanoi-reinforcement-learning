# Towers of Hanoi Reinforcement Learning

This repository contains a standalone Python implementation of a Towers of Hanoi Markov decision process (MDP) and small reinforcement-learning utilities.

## What it does

- Represents disk locations as MDP states and enumerates the three-peg action space.
- Checks legal moves and computes deterministic or slip-prone state transitions with rewards.
- Provides value iteration, Q-table initialization, policy extraction, and a one-step Q-learning update.
- Provides epsilon and alpha schedules for exploration and learning-rate decay.

## Hardware, tools, and libraries

No hardware is required. The project runs from the Python command line and uses only the Python standard library, including `dataclasses`, `itertools`, `math`, and `typing`. No external packages or model files are required.

## Run

Use Python 3.10 or newer:

```powershell
python demo.py
```

The demo creates a three-disk MDP, performs value-iteration updates, extracts a policy, applies one Q-table update, and prints the resulting start/goal states and schedule values.

## Credits

Implementation and repository curation: Sparsh Dadhich.
