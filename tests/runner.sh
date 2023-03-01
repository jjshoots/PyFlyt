#!/bin/bash

# run all core examples
for f in ./examples/core/*.py; do
  python3 "$f"
done

# run all gymnasium tests
pytest tests/test_gym_envs.py
