
#  Digitgen - Handwritten Digit Generation using DCGAN on MNIST Dataset

## Project Description

This project implements a Deep Convolutional Generative Adversarial Network (DCGAN) to generate realistic handwritten digits using the MNIST dataset. The model learns the distribution of handwritten digits (0-9) and can create novel digit samples that resemble the training data.

## Key Features

- Implementation of DCGAN architecture with generator and discriminator networks
- Training on the standard MNIST dataset (60,000 images)
- Generation of new handwritten digit samples
- TensorFlow 2.x implementation
- Model checkpointing and sample generation during training

## Model Architecture

### Generator
- Input: 200-dimensional random noise vector
- Architecture:
  - Dense layer reshaped to 7x7x256
  - Batch normalization and ReLU activation
  - Two transposed convolution layers with upsampling
  - Final layer with tanh activation producing 28x28x1 output

### Discriminator
- Input: 28x28x1 grayscale image
- Architecture:
  - Two convolution layers with LeakyReLU activation
  - Dropout for regularization
  - Final dense layer with sigmoid activation for binary classification

## Training Details

- Optimizer: Adam (learning rate 1e-4)
- Loss: Binary cross-entropy
- Batch size: 256
- Epochs: 100
- Normalization: Images scaled to [-1, 1]

## Requirements

- Python 3.x
- TensorFlow 2.x
- Matplotlib
- NumPy
- Jupyter Notebook (optional)

## Usage

1. Clone the repository:
```bash
git clone https://github.com/yourusername/mnist-dcgan.git
cd mnist-dcgan
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the training script:
```python
python train_dcgan.py
```

4. To generate new digits after training:
```python
python generate_digits.py --num_samples 10
```

## Results

After training for 100 epochs, the generator can produce realistic handwritten digits. Sample outputs are saved periodically during training to visualize progress.

## References

- Original DCGAN paper: [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434)
- MNIST dataset: [http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/)

