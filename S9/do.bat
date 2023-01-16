@echo off
if "%1" == "lfw" (
    python lfw_dataset.py
)

if "%1" == "extract" (
    python extract.py lfw
)

if "%1" == "visualize" (
    python lfw_dataset.py -visualize_batch
)

if "%1" == "performance" (
    python lfw_dataset.py -get_timing -num_workers %2
)