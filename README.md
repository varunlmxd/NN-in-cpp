# Neural Network From Scratch in C++ 

This repository contains a pure C++ implementation of neural networks built entirely from scratch, without any external libraries. It includes support for multiple layer types, activation functions, loss functions, and optimization algorithms.

## Setup

### 1. Dataset Preparation

First, navigate to the datasets folder and set up the necessary Python environment:

```bash
cd dataset
pip install -r requirements.txt
```

Run the following Python scripts to download and prepare the MNIST datasets:

```bash
# Download and prepare MNIST digits dataset
python mnist-numbers.py

# Download and prepare MNIST fashion dataset
python mnist-fashion.py
```

These scripts will:
- Download the respective datasets
- Preprocess the data (normalize and flatten)
- Save the datasets as CSV files in the dataset directory

### 2. Building the Project

Compile the project with a C++17 or newer compiler:

```bash
# Compile main program
g++ -std=c++17 -O3 main.cpp -o main
```

## Usage

Run the compiled program:

```bash
./main
```

The default configuration in main.cpp will:
1. Load the MNIST digits dataset
2. Create a neural network with 3 layers (input → 128 → 128 → 10)
3. Train the model for 10 epochs
4. Save the trained model to model.bin
5. Evaluate on the test set

## Key Components

- **Layers**: Dense layers with weights and biases (`layers.h`)
- **Activation Functions**: ReLU, Sigmoid, Softmax, Linear (`activation.h`)
- **Loss Functions**: Categorical/Binary Cross-entropy, MSE, MAE (`loss.h`)
- **Optimizers**: SGD with momentum, Adagrad, RMSprop, Adam (`optimizer.h`)
- **Utilities**: Matrix operations, CSV loading (`utils.h`)
- **Model**: High-level model class for building and training networks (`model.h`)
- **Accuracy Metrics**: Categorical and regression accuracy metrics (`accuracy.h`)

## Sample Code

```cpp
// Example: Creating and training a custom model

#include <bits/stdc++.h>
#include "utils.h"
#include "model.h"
using namespace std;

int main() {
    cout << fixed << setprecision(10);
    cout<<"Hello NN\n";
    // data reading
    cout<<"Data Loading\n";
    
    // Hand Written Digit MNIST
    // Load training data
    vector<vector<long double>> X = read_csv_2d<long double>("dataset/X_mnist_train.csv");
    variant< vector<vector<long double>>,vector<int>> y = read_csv_single<int>("dataset/y_mnist_train.csv");

    // Load testing data
    vector<vector<long double>> X_test = read_csv_2d<long double>("dataset/X_mnist_test.csv");
    variant< vector<vector<long double>>,vector<int>> y_test = read_csv_single<int>("dataset/y_mnist_test.csv");

    cout<<"Data Loaded\n";

    // Add layers
    Model model;
    model.add(Layer_Dense(X[0].size(), 128, 0, 0, 5e-4, 5e-4));
    model.add(Activation_ReLU());
    model.add(Layer_Dense(128, 128));
    model.add(Activation_ReLU());
    model.add(Layer_Dense(128, 10));
    model.add(Activation_Softmax());

    // Set loss, optimizer, and accuracy metric
    model.set( Loss_CategoricalCrossentropy(),Optimizer_Adam(0.005, 1e-4),Accuracy_Categorical());
    
    // Compile and train
    model.train(X,y,10,100,128);
    model.save("./model.bin");

    // Evaluate the trained model
    model.evaluate(X_test,y_test,128);
    
    cout<<"Bye NN\n";
    return 0;
}
```

## Model Saving and Loading

Save a trained model:

```cpp
model.save("./model.bin");
```

Load a previously saved model and evaluate:

```cpp
Model model;
model.load("./model.bin");
model.compile();  // Always compile after loading
model.evaluate(X_test,y_test,128);
```

## Example Results

The implementation achieves over 95% accuracy on the MNIST digits dataset, as shown in the included log file.

Modifications to the training code hyperparameters can lead to improvements beyond the current 95% accuracy benchmark.