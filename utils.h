#ifndef UTILS_H
#define UTILS_H

#include <bits/stdc++.h>
using namespace std;

// Split the string using , as delim
vector<string> split(const string& str, char delim) {
    vector<string> tokens;
    string token;
    istringstream tokenStream(str);
    while (getline(tokenStream, token, delim)) {
        tokens.push_back(token);
    }
    return tokens;
}

// Convert string to specific datatype
template<typename T>
T convert_string(const string& str) {
    stringstream ss(str);
    T val;
    ss >> val;
    return val;
}

template<typename T>
vector<T> read_csv_single(const string& filename) {
    vector<T> data;
    ifstream file(filename);

    if (!file.is_open()) {
        cerr << "ERROR: Failed to open file '" << filename << "'" << endl;
        throw runtime_error("Could not open file: " + filename);
    }

    string line;

    while (getline(file, line)) {
        auto values = split(line, ',');
        for (const auto& val : values) {
            data.push_back(convert_string<T>(val));
        }
    }
    return data;
}

template<typename T>
vector<vector<T>> read_csv_2d(const string& filename) {
    vector<vector<T>> data;
    ifstream file(filename);

    if (!file.is_open()) {
        cerr << "ERROR: Failed to open file '" << filename << "'" << endl;
        throw runtime_error("Could not open file: " + filename);
    }
    
    string line;

    while (getline(file, line)) {
        vector<T> row;
        auto values = split(line, ',');
        for (const auto& val : values) {
            row.push_back(convert_string<T>(val));
        }
        data.push_back(row);
    }
    return data;
}

// Matrix Functions
vector<vector<long double>> dot(vector<vector<long double>> inputs, vector<vector<long double>> weights) {
    int n_inputs = inputs.size();
    int n_neurons = weights[0].size();

    vector<vector<long double>> result(n_inputs, vector<long double>(n_neurons, 0));

    for(int i=0; i<n_inputs; i++) {//select input
        for(int j=0; j<n_neurons; j++) {//select weight column
            for(int k=0; k<inputs[0].size(); k++) {//select input as well as weight row 
                result[i][j] += inputs[i][k] * weights[k][j];
            }
        }
    }
    return result;
}

void add(vector<vector<long double>> &results, vector<long double> &biases) {
    if(results[0].size() != biases.size()) {
        cout<<"Error"<<endl;
    }
    for(int i=0; i<results.size(); i++) {
        for(int j=0; j<results[0].size(); j++) {
            results[i][j] += biases[j];
        }
    }
}

vector<vector<long double>> transpose(vector<vector<long double>> arr) {
    vector<vector<long double>> transposed_arr(arr[0].size(), vector<long double>(arr.size(), 0));
    for(int i=0; i<arr.size(); i++) {
        for(int j=0; j<arr[0].size(); j++) {
            transposed_arr[j][i] = arr[i][j];
        }
    }
    return transposed_arr;
}

long double calc_accuracy(vector<vector<long double>> &y_pred, vector<int>& y_true) {
    int correct = 0;
    for(int i=0; i<y_pred.size(); i++) {
        int y_pred_class = distance(y_pred[i].begin(), max_element(y_pred[i].begin(), y_pred[i].end()));
        if(y_pred_class == y_true[i]) correct++;
    }
    return (long double)correct/y_pred.size();
}

long double calc_binary_accuracy(const vector<vector<long double>>& predictions, const vector<vector<int>>& y_true) {
    int correct = 0;
    int total = predictions.size();
    
    for (size_t i = 0; i < predictions.size(); i++) {
        for (size_t j = 0; j < predictions[i].size(); j++) {
            // Convert to binary prediction (1 if > 0.5, 0 otherwise)
            int predicted_class = (predictions[i][j] > 0.5) ? 1 : 0;
            
            // Check if prediction matches ground truth
            if (predicted_class == y_true[i][j]) {
                correct++;
            }
        }
    }
    
    // Return accuracy as percentage
    return (long double)correct / total;
}

#endif // UTILS_H