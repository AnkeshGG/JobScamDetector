# JobScamDetector

A machine learning project to detect fraudulent job postings. The repository has three main parts: ml, frontend, and requirements.txt.

## Project Structure

* **ml**: contains machine learning code for preprocessing, feature extraction, training, and evaluation
* **frontend**: user interface for interacting with the model
* **requirements.txt**: list of Python dependencies for creating the virtual environment

## Setup Instructions

### 1. Create a virtual environment with Conda in the project root

```bash
conda create -p venv python=3.12
conda activate ./venv
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

## ML Component

* Located in the `ml` folder
* Includes scripts for preprocessing datasets, extracting features, training models, and evaluating performance
* Produces a trained model that predicts whether a job posting is real (0) or fake (1)


## Usage

1. Train the model using scripts in `ml`
2. Launch the frontend to interact with the trained model
3. Input job descriptions and receive predictions with confidence scores

## Quick Start

To run the FastAPI backend locally, use the following command:

```bash
uvicorn backend.app.main:app --reload
```

Once it’s running, open your browser and go to:
```bash
http://127.0.0.1:8000/docs (127.0.0.1 in Bing) → Swagger UI (interactive API interface)

http://127.0.0.1:8000/redoc (127.0.0.1 in Bing) → ReDoc interface (alternative documentation view)
```

## Notes

* Dataset is based on publicly available job posting datasets such as Kaggle and EMSCAD
* Labels: 0 = Real, 1 = Fake
* Preprocess datasets before training for best accuracy

## Requirements

* All dependencies are listed in `requirements.txt`
* Install them inside your virtual environment before running any scripts

## Future Improvements

* Add more balanced datasets for better accuracy
* Enhance frontend with visualization of prediction explanations
* Deploy as a web service for broader use