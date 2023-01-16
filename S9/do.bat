if "%1" = "lfw" (
    python lfw_dataset.py
)

if "%1" = "extract" (
    python extract.py lfw
)