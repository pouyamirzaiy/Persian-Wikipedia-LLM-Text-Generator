# Autoencoder Reconstruction of Mixed MNIST and CIFAR-10 Images

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

In this project, we explore the use of autoencoders, a fundamental technique in deep learning, to reconstruct images from two distinct datasets: MNIST and CIFAR-10. The objective is to create an autoencoder model capable of taking the mean of an MNIST and a CIFAR-10 image, feeding it into the model, and generating reconstructions of both MNIST and CIFAR-10 images.

## Dataset Description

- **MNIST**: A dataset of handwritten digits with grayscale images of size 28x28 pixels.
- **CIFAR-10**: A dataset of 60,000 32x32 color images in 10 classes, with 6,000 images per class.

## Project Structure

```
.
├── SRC
│   └── dual_decoder_autoencoder.py
├── Notebook
│   └── autoencoder_training.ipynb
├── requirements.txt
├── README.md
├── Results
│   ├── reconstructed_mnist.png
│   └── reconstructed_cifar.png
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/autoencoder-reconstruction.git
   cd autoencoder-reconstruction
   ```
2. Install the required packages:
   ```bash
   pip install -r Requirements/requirements.txt
   ```

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/medical-appointment-no-shows.git
   ```
2. Navigate to the project directory:
   ```bash
   cd medical-appointment-no-shows
   ```
3. Run the Jupyter notebook or Python script to preprocess the data, train the model, and evaluate its performance.

## Results

### Implementation of MLP Model using PyTorch

The implementation of the Multi-Layer Perceptron (MLP) model was done using PyTorch, a popular open-source machine learning library. The model was designed with an input layer, multiple hidden layers, and an output layer.

#### Model Architecture

The model architecture is as follows:

- **Input Size**: The input size is equal to the number of features in the dataset, which is determined by the number of inputs.
- **Hidden Size**: The hidden size, which is the number of neurons in the hidden layer, was set to 128. This can be adjusted as needed.
- **Output Size**: The output size is 1, corresponding to our binary classification problem.
- **Number of Hidden Layers**: The model was designed with 2 hidden layers.
- **Loss Function and Optimizer**: The model uses Mean Squared Error (MSE) as the loss function and Adam as the optimizer.

### Evaluation Metrics

The quality of the reconstructions was evaluated using the structural similarity index (SSIM) and peak signal-to-noise ratio (PSNR). The SSIM values for the MNIST and CIFAR-10 reconstructions were 0.9380 and 0.2776, respectively, indicating that the model was able to reconstruct the MNIST images with high similarity but had more difficulty with the CIFAR-10 images. The PSNR values for the MNIST and CIFAR-10 reconstructions were 27.45 dB and 20.15 dB, respectively, suggesting that the model was able to reconstruct the MNIST images with less error than the CIFAR-10 images.

- **SSIM (MNIST)**: 0.9380
- **SSIM (CIFAR-10)**: 0.2776
- **PSNR (MNIST)**: 27.45 dB
- **PSNR (CIFAR-10)**: 20.15 dB

## Conclusion

This project demonstrates the capability of autoencoders to reconstruct images from mixed datasets. The results show that the model can effectively learn to reconstruct both MNIST and CIFAR-10 images from their mean.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- The PyTorch team for their excellent deep learning library.
- The authors of the MNIST and CIFAR-10 datasets.

## Contact Information

For any questions or inquiries, please contact pouya.8226@gmail.com.
