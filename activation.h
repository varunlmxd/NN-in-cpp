#ifndef ACTIVATION_H
#define ACTIVATION_H

#include <bits/stdc++.h>
using namespace std;

class Activation {
    public:
        Activation() = default;
        vector<vector<long double>> output, inputs, dinputs;
    
        virtual void forward(vector<vector<long double>> &inputs) = 0;
        virtual void backward(vector<vector<long double>> &dvalues) = 0;
        // Template method for predictions
        // template<typename T>
        // T predictions(const vector<vector<long double>>& outputs) {
        //     return T(); // Default empty return
        // }
        virtual void print() {
            for (auto &row : output) {
                for (auto &val : row)
                    cout << val << " ";
                cout << endl;
            }
        }
};

class Activation_ReLU : public Activation {
    public:
        
        void forward(vector<vector<long double>> &inputs) override {
            this->inputs = inputs;
            this->output = inputs;
            for(auto &a : output) {
                for(auto &val : a) {
                    val = max((long double)0, val);
                }
            }
        }

        void backward(vector<vector<long double>> & dvalues) override {
            this->dinputs = dvalues;
            for(int i=0; i<this->dinputs.size(); i++) {
                for(int j=0; j<this->dinputs[0].size(); j++) {
                    if(this->inputs[i][j] <= 0)
                        this->dinputs[i][j] = 0;
                }
            }
        }

        // vector<vector<long double>> predictions(const vector<vector<long double>>& outputs) override {
        //     return outputs;
        // }
};

class Activation_Softmax : public Activation {
    public:

        void forward(vector<vector<long double>> &inputs) override {
            this->inputs = inputs;
            this->output = inputs;
            for(auto &a : this->output) {
                long double max_val = *max_element(a.begin(), a.end());
                long double denominator = 0;

                for(auto &val: a) {
                    val = exp(val - max_val);
                    denominator += val;
                }
                
                for(auto &val: a) {
                    val = val/denominator;
                }
            }
        }
        
        void backward(vector<vector<long double>> &dvalues) override {
            // Create uninitialized array
            this->dinputs = vector<vector<long double>>(dvalues.size(), 
                                vector<long double>(dvalues[0].size(), 0));
            
            // Enumerate outputs and gradients
            for (size_t i = 0; i < output.size(); i++) {
                // Flatten output array - easier to work with
                const auto& single_output = output[i];
                
                // Calculate Jacobian matrix of the output
                for (size_t j = 0; j < single_output.size(); j++) {
                    for (size_t k = 0; k < single_output.size(); k++) {
                        // Calculate element of Jacobian matrix
                        // Derivative of softmax output[j] with respect to input[k]
                        long double jacobian_value;
                        if (j == k) {
                            jacobian_value = single_output[j] * (1 - single_output[j]);
                        } else {
                            jacobian_value = -single_output[j] * single_output[k];
                        }
                        
                        // Multiply by the gradient values
                        this->dinputs[i][j] += jacobian_value * dvalues[i][k];
                    }
                }
            }
        }

        // Calculate predictions from softmax outputs
        // vector<int> predictions(vector<vector<long double>>& outputs) override {
        //     vector<int> predictions;
        //     predictions.reserve(outputs.size());
            
        //     // For each sample, find the index of maximum value (argmax)
        //     for (const auto& output : outputs) {
        //         // Find the index of the maximum element in this sample
        //         auto max_it = max_element(output.begin(), output.end());
        //         int class_idx = distance(output.begin(), max_it);
        //         predictions.push_back(class_idx);
        //     }
            
        //     return predictions;
        // }

        // vector<vector<long double>> predictions(const vector<vector<long double>>& outputs) {
        //     return outputs;
        // }
};

class Activation_Sigmoid : public Activation{
    public:
    
    void forward(vector<vector<long double>> &inputs) override {
        this->inputs = inputs;
        this->output = inputs;
        for(auto &a : output) {
            for(auto &val : a) {
                val = (long double)1.0/((long double)1.0+exp(-val));
            }
        }
    }

    void backward(vector<vector<long double>> & dvalues) override {
        // for initialization we assign which will be overidded in the next loop
        this->dinputs = dvalues;
        for(int i=0; i<this->dinputs.size(); i++) {
            for(int j=0; j<this->dinputs[0].size(); j++) {
                this->dinputs[i][j] = dvalues[i][j] * (1-this->output[i][j]) * (this->output[i][j]);
            }
        }
    }

    // vector<vector<int>> predictions(const vector<vector<long double>>& outputs) override {
    //     vector<vector<int>> binary_predictions;
    //     binary_predictions.reserve(outputs.size());
        
    //     for (const auto& sample : outputs) {
    //         vector<int> row;
    //         row.reserve(sample.size());
            
    //         for (const auto& value : sample) {
    //             // Convert to binary: 1 if value > 0.5, otherwise 0
    //             row.push_back(value > 0.5 ? 1 : 0);
    //         }
            
    //         binary_predictions.push_back(row);
    //     }
        
    //     return binary_predictions;
    // }
};

class Activation_Linear : public Activation{
    public:
        
        void forward(vector<vector<long double>> &inputs) override {
            this->inputs = inputs;
            this->output = inputs;
        }

        void backward(vector<vector<long double>> & dvalues) override {
            this->dinputs = dvalues;
        }
        
        // vector<vector<long double>> predictions(const vector<vector<long double>>& outputs) override {
        //     return outputs;
        // }
};


#endif // ACTIVATION_H