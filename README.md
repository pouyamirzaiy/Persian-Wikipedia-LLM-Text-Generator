# Persian Wikipedia Text Generator

## Table of Contents

- [Introduction](#introduction)
- [Dataset Description](#dataset-description)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Conclusion](#conclusion)
- [License](#license)
- [Acknowledgements](#acknowledgements)
- [Contact Information](#contact-information)

## Introduction

This project aims to develop a text generator using a large language model (LLM) on the Persian Wikipedia dataset. The model is implemented from scratch without utilizing pre-trained networks or fine-tuning existing models. The project involves creating a dataset, implementing a neural network for text generation, training the model, and evaluating its performance using various metrics.

## Dataset Description

The dataset consists of articles fetched from Persian Wikipedia. A custom dataset class `PersianWikipediaDataset` is implemented to load and preprocess the articles. The dataset is used for both pre-training and fine-tuning the text generation model.

## Project Structure

```
PersianWikipediaTextGenerator/
│
├── data/
│   └── persian_wikipedia/  # Directory for storing fetched articles
│
├── models/
│   └── text_generator.py  # Implementation of the text generation model
│
├── notebooks/
│   └── LLM.ipynb  # Jupyter notebook
├── scr/
│   ├── pretrain.py  # Script for pre-training the model
│   ├── finetune.py  # Script for fine-tuning the model
│   └── evaluate.py  # Script for evaluating the model
│
├── README.md  # Project README file
└── requirements.txt  # List of dependencies
```

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/PersianWikipediaTextGenerator.git
   cd PersianWikipediaTextGenerator
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Install additional dependencies for evaluation metrics:
   ```bash
   pip install rouge_score datasets
   ```

## Usage

1. **Pre-train the model**:

   ```bash
   python scripts/pretrain.py
   ```

2. **Fine-tune the model**:

   ```bash
   python scripts/finetune.py
   ```

3. **Evaluate the model**:
   ```bash
   python scripts/evaluate.py
   ```

## Results

The model is evaluated using Perplexity, ROUGE, BLEU, and METEOR metrics. The evaluation results are printed at the end of the evaluation script.

Example results:

```
Perplexity: 20.5
ROUGE Results: {'rouge1': 0.45, 'rouge2': 0.25, 'rougeL': 0.40}
BLEU Results: {'bleu': 0.30}
METEOR Results: {'meteor': 0.35}
```

## Conclusion

This project demonstrates the implementation of a text generator using a large language model on the Persian Wikipedia dataset. The model is trained from scratch and evaluated using various metrics. The results indicate the model's ability to generate coherent text in Persian.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- The Persian Wikipedia community for providing the dataset.
- The developers of PyTorch and Hugging Face Transformers for their excellent libraries.
- The creators of the evaluation metrics used in this project.

## Contact Information

For any questions or inquiries, please contact:

- Email:pouya.8226@gmail.com
