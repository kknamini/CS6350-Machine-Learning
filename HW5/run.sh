#!/bin/bash

echo "Enabling Anaconda commands using: . /c/Anaconda3/etc/profile.d/conda.sh"

. /c/Anaconda3/etc/profile.d/conda.sh

echo "Activating Anaconda Base Environment using: conda activate"

conda activate

echo "Running hw5.py"

python hw5.py

echo "hw5.py complete"