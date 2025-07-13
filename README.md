# RockPaperScissors Deep Learning Project

This project trains and evaluates a deep learning model to classify hand gestures for Rock, Paper, and Scissors using PyTorch and MediaPipe.

## Project Structure

```
RockPaperScissors/
├── project/
│   ├── simple_example.py
│   ├── requirements.txt
│   └── simple_example.ipynb
├── modules/
│   ├── dataset.py
│   ├── hand_visualizations.py
├── 2/                   # Dataset directory
│   └── paper/
│   └── rock/
│   └── scissors/
│   └── rps-cs-images/
├── Dockerfile
└── README.md
```

## Running with Docker

1. **Build the Docker image:**
    ```sh
    docker build -t rps-project .
    ```

2. **Run the container:**
    ```sh
    docker run --rm rps-project
    ```

The dataset should be placed in the `2/` directory at the project root (as shown above). The code expects the dataset to be available at `/data` inside the container.

## Notes

- Python dependencies are listed in `project/requirements.txt`.
- The main script is `project/simple_example.py`.
- The Dockerfile sets up all dependencies and copies the dataset into the container.
