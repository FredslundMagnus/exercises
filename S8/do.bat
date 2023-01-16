@echo off
if "%1" == "start" (
    start /b python -m uvicorn main:app --reload
    exit /b
)

if "%1" == "predict" (
    curl -X "POST" "http://127.0.0.1:8000/iris_v2/?sepal_length=%2&sepal_width=%3&petal_length=%4&petal_width=%5" -H "accept: application/json" -d ""
)

if "%1" == "data" (
    python data_drift.py
)