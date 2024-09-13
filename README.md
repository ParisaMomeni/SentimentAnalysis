# SentimentAnalysis
# Semaglutide Twitter Sentiment Analysis

This repository analyzes sentiment on Twitter related to Semaglutide.



## Instructions

1. Clone this repository:

    ```bash
    git clone 
    cd ...
    ```

2. Create the required folder structure and add the data file:

    - In the root directory, create a folder named `data` and place the dataset file `Semaglutide_Twitter_20210601_20240331.pkl` in it.
    - Create a folder `output` and a file `result1` within it.

3. Run the analysis:

    ```bash
    python LR.py data/Semaglutide_Twitter_20210601_20240331.pkl 50 1 output/result1
    ```

    - **data/Semaglutide_Twitter_20210601_20240331.pkl**: The path to the dataset.
    - **50**: Number of iterations for the hypothesis testings.
    - **1**: 1/0 
    - **output/result1**: Folder where the results will be stored.

## Requirements

- Python 3.x
- Required Python packages (can be installed via `requirements.txt`):

    ```bash
    pip install -r requirements.txt
    ```



