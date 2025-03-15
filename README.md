# Multi-Task Learning for BioNLP

A multi-task learning implementation combining sentence classification and named entity recognition (NER) using RoBERTa with LoRA fine-tuning.

## Project Structure

```
.
├── task1_sentence_transformer.py
├── task2_multi_task_learning.py
├── task3_training_considerations.py
├── task4_training_loop.py
├── requirements.txt
├── Dockerfile
└── README.md
```

## Docker Setup

### Prerequisites
- Docker installed on your system
- NVIDIA Docker runtime (for GPU support)

### Building the Docker Image
```bash
# Build the image
docker build -t bionlp-multitask .
```

### Running Different Tasks
You can run different tasks by setting the TASK_FILE environment variable:

```bash
# Run Task 1 (default)
docker run -it --rm bionlp-multitask

# Run Task 1 (explicit)
docker run -it --rm -e TASK_FILE=task1_sentence_transformer.py bionlp-multitask

# Run Task 2
docker run -it --rm -e TASK_FILE=task2_multi_task_learning.py bionlp-multitask

# With GPU support (recommended for LoRA)
docker run -it --rm --gpus all -e TASK_FILE=task2_multi_task_learning.py bionlp-multitask
```

### Development Mode
To develop and test inside the container:
```bash
# Mount current directory and run in interactive mode
docker run -it --rm -v $(pwd):/app bionlp-multitask bash

# Then inside the container you can run any task:
python task1_sentence_transformer.py
python task2_multi_task_learning.py
```

## Manual Setup (without Docker)

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the training:
```bash
python task4_training_loop.py
```

## Model Architecture

- Base model: RoBERTa-large
- Fine-tuning: LoRA (Low-Rank Adaptation)
  - Rank: 8
  - Alpha: 16
  - Dropout: 0.1
- Tasks:
  - Task A: Document section classification
  - Task B: Named Entity Recognition (NER)

## Training Parameters

- Learning rate: 2e-5
- Batch size: 4
- Epochs: 3
- Optimizer: Adam
- Loss weights: Classification (1.0), NER (1.0)
- LoRA Configuration:
  - Target modules: query and value matrices
  - Bias: none
  - Task modules: Classification and NER heads

## Key Features

1. Efficient Fine-tuning
   - Uses LoRA for parameter-efficient training
   - Only fine-tunes small rank decomposition matrices
   - Significantly reduces memory usage and training time

2. Multi-Task Learning
   - Shared transformer backbone
   - Task-specific LoRA adapters
   - Balanced loss weighting

3. Performance Optimization
   - Gradient checkpointing support
   - Mixed precision training
   - Efficient memory management

## Notes

- The model uses LoRA for efficient fine-tuning
- GPU support is highly recommended for training
- See individual task files for detailed implementation notes
- Memory requirements are significantly reduced compared to full fine-tuning


## Task Descriptions

### Task 1: Sentence Transformer
- Basic implementation of a sentence transformer using BERT
- Includes mean pooling for sentence embeddings
- Can be used independently for sentence embedding tasks

### Task 2: Multi-Task Learning
- Extends the base transformer for both classification and NER
- Implements task-specific heads
- Uses the BioNLP dataset for demonstration

### Task 3: Training Considerations
- Documents different training strategies
- Implements various freezing approaches
- Provides transfer learning guidelines

### Task 4: Training Loop
- Complete training implementation
- Includes metrics calculation
- Supports both classification and NER tasks

## Usage

### Running Individual Tasks

1. Sentence Transformer:
```bash
python task1_sentence_transformer.py
```

2. Multi-Task Model:
```bash
python task2_multi_task_learning.py
```

3. Training Considerations:
```bash
python task3_training_considerations.py
```

4. Training Loop:
```bash
python task4_training_loop.py
```

## Model Architecture

The multi-task model consists of:
- A shared transformer backbone (RoBERTa-large)
- A classification head for sentence classification
- An NER head for token classification
- Custom pooling and preprocessing layers

## Training

The training process includes:
- Multi-task learning with weighted losses
- Flexible freezing strategies
- Comprehensive metrics tracking
- Support for both tasks simultaneously

## Metrics

The system tracks:
- Classification metrics (Precision, Recall, F1)
- NER metrics (Entity-level F1)
- Combined loss for both tasks

## License

This project is provided as-is for educational and research purposes. 