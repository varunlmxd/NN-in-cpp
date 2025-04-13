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
    vector<vector<long double>> X = read_csv_2d<long double>("dataset/X_mnist_train.csv");
    variant< vector<vector<long double>>,vector<int>> y = read_csv_single<int>("dataset/y_mnist_train.csv");

    vector<vector<long double>> X_test = read_csv_2d<long double>("dataset/X_mnist_test.csv");
    variant< vector<vector<long double>>,vector<int>> y_test = read_csv_single<int>("dataset/y_mnist_test.csv");

    // Fashion MNIST
    // vector<vector<long double>> X = read_csv_2d<long double>("dataset/X_fashion_mnist_train.csv");
    // variant< vector<vector<long double>>,vector<int>> y = read_csv_single<int>("dataset/y_fashion_mnist_train.csv");

    // vector<vector<long double>> X_test = read_csv_2d<long double>("dataset/X_fashion_mnist_test.csv");
    // variant< vector<vector<long double>>,vector<int>> y_test = read_csv_single<int>("dataset/y_fashion_mnist_test.csv");

    cout<<"Data Loaded\n";

    // Add layers
    Model model;
    model.add(Layer_Dense(X[0].size(), 128, 0, 0, 5e-4, 5e-4));
    model.add(Activation_ReLU());
    model.add(Layer_Dense(128, 128));
    model.add(Activation_ReLU());
    model.add(Layer_Dense(128, 10));
    model.add(Activation_Softmax());
    model.set( Loss_CategoricalCrossentropy(),Optimizer_Adam(0.005, 1e-4),Accuracy_Categorical());
    model.compile();

    model.train(X,y,10,100,128);
    model.save("./model.bin");
    // model.load("./model.bin");
    // model.compile();
    model.evaluate(X_test,y_test,128);
    
    cout<<"Bye NN\n";
    return 0;
}