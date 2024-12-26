# MultiLayer Perceptron

This repository contains an implementation of a MultiLayer Perceptron (MLP) using Python and NumPy. An MLP is a class of feedforward artificial neural network that consists of at least three layers of nodes: an input layer, one or more hidden layers, and an output layer. MLPs are capable of modeling complex, non-linear relationships and are widely used for tasks such as classification, regression, and pattern recognition.

## Features

- Implementation of a MultiLayer Perceptron from scratch using Python and NumPy.
- Customizable architecture with adjustable numbers of hidden layers and neurons.
- Training using backpropagation and gradient descent algorithms.
- Activation functions include sigmoid, tanh, and ReLU.
- Example usage on standard datasets.

## Installation

1. **Clone the repository:**

   ```
   git clone https://github.com/NithinVS2k4/MultiLayerPerceptron.git
   cd MultiLayerPerceptron
  
2. **Set up a virtual environment (optional but recommended):**
   ```
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
  
3. **Install the required dependencies:**
   ```
    pip install -r requirements.txt
    Note: Ensure that you have Python 3.x installed on your system.

## Usage
1. **Prepare your dataset:**

     Place your dataset in the appropriate directory or modify the data loading section in the script to point to your dataset.
  Configure the MLP:

2. **Open the mlp.py file**
  Adjust the network architecture by modifying the layers parameter.
  Set hyperparameters such as learning rate, number of epochs, and batch size.

3. **Train the model:**
   ```
    python train.py
    Monitor the training process through the console output.

4. **Evaluate the model:**

  After training, run the evaluation script to assess performance:

    python evaluate.py
    
## Contributing
  Contributions are welcome! If you have suggestions for improvements or new features, feel free to open an issue or submit a pull request.

## License
  This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments
  Inspired by the foundational concepts of neural networks and multilayer perceptrons. 
  Thanks to the open-source community for providing valuable resources and tools.
