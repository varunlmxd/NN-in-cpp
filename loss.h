#ifndef LOSS_H
#define LOSS_H

#include <bits/stdc++.h>
#include "activation.h"
#include "layers.h"
using namespace std;

class Loss {
    public:
        vector<vector<long double>> dinputs;
        long double accumulated_sum,accumulated_count;
        void clip_vector(vector<long double>& vec, long double &min_val, long double &max_val) {
            for (auto& v : vec) {
                v = clamp(v, min_val, max_val);
            }
        }

        void new_pass(){
            this->accumulated_sum = 0;
            this->accumulated_count = 0;     
        }
        
        long double calculate(vector<vector<long double>> &output, vector<int> &y) {
            // cout<<"loss "<<output.size()<<" y "<<y.size()<<endl;
            vector<long double> sample_losses = forward(output, y);
            long double sum = accumulate(sample_losses.begin(), sample_losses.end(), 0.0);
            long double data_loss = (long double)sum/sample_losses.size();
            this->accumulated_sum += sum;
            this->accumulated_count += sample_losses.size();
            return data_loss;
        }

        long double calculate(vector<vector<long double>> &output, vector<vector<int>> &y) {
            // cout<<"loss "<<output.size()<<" y "<<y.size()<<endl;
            vector<long double> sample_losses = forward(output, y);
            long double sum = accumulate(sample_losses.begin(), sample_losses.end(), 0.0);
            long double data_loss = (long double)sum/sample_losses.size();
            this->accumulated_sum += sum;
            this->accumulated_count += sample_losses.size();
            return data_loss;
        }

        long double calculate(vector<vector<long double>> &output, vector<vector<long double>> &y) {
            // cout<<"loss "<<output.size()<<" y "<<y.size()<<endl;
            vector<long double> sample_losses = forward(output, y);
            long double sum = accumulate(sample_losses.begin(), sample_losses.end(), 0.0);
            long double data_loss = (long double)sum/sample_losses.size();
            this->accumulated_sum += sum;
            this->accumulated_count += sample_losses.size();
            return data_loss;
        }

        long double calculate_accumulated(){
            return this->accumulated_sum / this->accumulated_count ;
        }
        
        long double regularization_loss(Layer_Dense &layer){
            long double regularization_loss = 0.0;
            // cout<<"Enter regularization_loss\n";
            if(layer.weight_regularizer_l1 > 0){
                long double sum = 0;
                for(int i=0; i<layer.weights.size(); i++) {
                    for(int j=0; j<layer.weights[0].size(); j++) {
                        sum += abs(layer.weights[i][j]);
                    }
                }
                regularization_loss += layer.weight_regularizer_l1 * sum;
            }

            if(layer.weight_regularizer_l2 > 0){
                long double sum = 0;
                for(int i=0; i<layer.weights.size(); i++) {
                    for(int j=0; j<layer.weights[0].size(); j++) {
                        sum += (layer.weights[i][j] * layer.weights[i][j]);
                    }
                }
                regularization_loss += layer.weight_regularizer_l2 * sum;
            }

            if(layer.bias_regularizer_l1 > 0){
                long double sum = 0;
                for(int i=0; i<layer.biases.size(); i++) {
                    sum += abs(layer.biases[i]);
                }
                regularization_loss += layer.bias_regularizer_l1 * sum;
            }

            if(layer.bias_regularizer_l2 > 0){
                long double sum = 0;
                for(int i=0; i<layer.biases.size(); i++) {
                    sum += (layer.biases[i] * layer.biases[i]);
                }
                regularization_loss += layer.bias_regularizer_l2 * sum;
            }

            return regularization_loss;
        }


        virtual vector<long double> forward(vector<vector<long double>> &y_pred, vector<int> &y_true) {
            throw runtime_error("This loss function doesn't support vector<int> labels");
        }
        
        // for Binary Cross Entropy
        virtual vector<long double> forward(vector<vector<long double>> &y_pred, vector<vector<int>> &y_true) {
            throw runtime_error("This loss function doesn't support vector<vector<int>> labels");
        }

        // for MSE
        virtual vector<long double> forward(vector<vector<long double>> &y_pred, vector<vector<long double>> &y_true) {
            throw runtime_error("This loss function doesn't support vector<vector<long double>> labels");
        }

        virtual void backward(vector<vector<long double>> &dvalues, vector<vector<long double>> &y_true){
            throw runtime_error("This base loss function's backward method is not implemented for vector<vector<long double>> labels");
        }

        virtual void backward(vector<vector<long double>> &dvalues, vector<vector<int>> &y_true){
            throw runtime_error("This base loss function's backward method is not implemented for vector<vector<int>> labels");
        }
        
        virtual void backward(vector<vector<long double>> &dvalues, vector<int> &y_true){
            throw runtime_error("This base loss function's backward method is not implemented for vector<int> labels");

        }
};

class Loss_CategoricalCrossentropy : public Loss {
    public:
    vector<long double> forward(vector<vector<long double>> &y_pred, vector<int> &y_true) override {
        const size_t samples = y_pred.size();
        vector<long double> negative_log_likelihoods(samples);
        
        long double min_val = 1e-7L;
        long double max_val = 1.0 - min_val;

        for(size_t i = 0; i < samples; i++) {
            clip_vector(y_pred[i], min_val, max_val);
            negative_log_likelihoods[i] = -log(y_pred[i][y_true[i]]);
        }
        
        return negative_log_likelihoods;            
    }

    void backward(vector<vector<long double>> &dvalues, vector<int> &y_true) override {
        // Number of samples
        // cout<<"Enter\n";
        // cout<<dvalues.size()<<endl;
        // cout<<dvalues[0].size()<<endl;
        const int samples = dvalues.size();
        // Number of labels in every sample
        const int labels = dvalues[0].size();
        // cout<<samples<<" "<<labels<<endl;
        // Initialize gradient array
        this->dinputs = vector<vector<long double>>(samples, vector<long double>(labels, 0.0));
        
        for(int i = 0; i < samples; i++) {
            // For the correct class (y_true[i]), set -1/value in dinputs
            // All other positions remain 0
            this->dinputs[i][y_true[i]] = -1.0 / dvalues[i][y_true[i]];
            
            // Normalize gradient (inlined inside the parallel region)
            for(int j = 0; j < labels; j++) {
                this->dinputs[i][j] /= samples;
            }
        }
    }
};

// Combination of Loss and Softmax which is faster
class Activation_Softmax_Loss_CategoricalCrossentropy {
    public:
        Activation_Softmax activation;
        Loss_CategoricalCrossentropy loss;
        vector<vector<long double>> output,dinputs;

        long double forward(vector<vector<long double>> &inputs, vector<int> &y_true) {
            this->activation.forward(inputs);
            this->output = this->activation.output;

            return this->loss.calculate(output, y_true);
        }
        
        void backward(vector<vector<long double>> &dvalues, vector<int> &y_true) {
            int samples = dvalues.size();

            dinputs = dvalues;

            //Calculate Gradient
            for(int i=0; i<dinputs.size(); i++) {
                dinputs[i][y_true[i]] -= 1.0;
            }
            //Normalize Gradient
            for(int i=0; i<dinputs.size(); i++) {
                for(int j=0; j<dinputs[0].size(); j++) {
                    dinputs[i][j] /= samples;
                }
            }
        }
};

class Loss_BinaryCrossentropy : public Loss {
    public:
    
    long double min_val = 1e-7L;
    long double max_val = 1.0 - 1e-7L;

    vector<long double> forward(vector<vector<long double>> &y_pred, vector<vector<int>> &y_true) override {
        vector<long double> sample_losses;

        //clipping the vector 
        for(auto &y_pred_single : y_pred) {
            clip_vector(y_pred_single, min_val, max_val);
        }

        for(int i=0;i<y_pred.size();i++){
            long double sum = 0;
            for(int j=0;j<y_pred[0].size();j++){
                sum+= -(y_true[i][j] * log(y_pred[i][j]) + (1-y_true[i][j]) * log(1 - y_pred[i][j]));
            }
            sample_losses.push_back(sum/y_pred[0].size());
        }

        return sample_losses;            
    }

    void backward(vector<vector<long double>> &dvalues, vector<vector<int>> &y_true){
        int samples = dvalues.size();

        int outputs = dvalues[0].size();

        // clipping the vector 
        for(auto &dvalues_single : dvalues) {
            clip_vector(dvalues_single, min_val, max_val);
        }

        // for initialization of structure and size
        this->dinputs = dvalues;


        //Calculate Gradient
        for(int i=0;i<dinputs.size();i++){
            for(int j=0;j<dinputs[0].size();j++){
                this->dinputs[i][j] = -( y_true[i][j] / dvalues[i][j] - (1-y_true[i][j]) / (1-dvalues[i][j])) / outputs;
            }
        }

        //Normalize Gradient
        for(int i=0; i<dinputs.size(); i++) {
            for(int j=0; j<dinputs[0].size(); j++) {
                dinputs[i][j] /= samples;
            }
        }
    }
};

class Loss_MeanSquaredError : public Loss {
    public:
        
    
        vector<long double> forward(vector<vector<long double>> &y_pred, vector<vector<long double>> &y_true) override {
            // Initialize sample_losses with size of number of samples
            vector<long double> sample_losses(y_pred.size());
    
            // Calculate loss for each sample
            for(int i = 0; i < y_pred.size(); i++) {
                long double sum = 0;
                // Calculate mean squared error along the outputs axis
                for(int j = 0; j < y_pred[i].size(); j++) {
                    sum += pow(y_true[i][j] - y_pred[i][j], 2);
                }
                // Store mean for this sample
                sample_losses[i] = sum / y_pred[i].size();
            }
            return sample_losses;
        }
    
        void backward(vector<vector<long double>> &dvalues, vector<vector<long double>> &y_true) override {
            // Number of samples and outputs
            int samples = dvalues.size();
            int outputs = dvalues[0].size();
    
            // Initialize gradient array
            this->dinputs = vector<vector<long double>>(samples, vector<long double>(outputs));
    
            // Calculate gradient
            for(int i = 0; i < samples; i++) {
                for(int j = 0; j < outputs; j++) {
                    this->dinputs[i][j] = -2 * (y_true[i][j] - dvalues[i][j]) / outputs;
                }
            }
    
            // Normalize gradient
            for(int i = 0; i < samples; i++) {
                for(int j = 0; j < outputs; j++) {
                    this->dinputs[i][j] /= samples;
                }
            }
        }
};

class Loss_MeanAbsoluteError : public Loss {
    public:
        
    
        vector<long double> forward(vector<vector<long double>> &y_pred, vector<vector<long double>> &y_true) override {
            // Initialize sample_losses with size of number of samples
            vector<long double> sample_losses(y_pred.size());
    
            // Calculate loss for each sample
            for(int i = 0; i < y_pred.size(); i++) {
                long double sum = 0;
                // Calculate mean absolute error along the outputs axis
                for(int j = 0; j < y_pred[i].size(); j++) {
                    sum += abs(y_true[i][j] - y_pred[i][j]);
                }
                // Store mean for this sample
                sample_losses[i] = sum / y_pred[i].size();
            }
            return sample_losses;
        }
    
        void backward(vector<vector<long double>> &dvalues, vector<vector<long double>> &y_true) {
            // Number of samples and outputs
            int samples = dvalues.size();
            int outputs = dvalues[0].size();
    
            // Initialize gradient array
            this->dinputs = vector<vector<long double>>(samples, vector<long double>(outputs));
    
            // Calculate gradient
            for(int i = 0; i < samples; i++) {
                for(int j = 0; j < outputs; j++) {
                    // np.sign() equivalent
                    long double diff = y_true[i][j] - dvalues[i][j];
                    this->dinputs[i][j] = (diff > 0) ? 1 : ((diff < 0) ? -1 : 0);
                    this->dinputs[i][j] /= outputs;
                }
            }
    
            // Normalize gradient
            for(int i = 0; i < samples; i++) {
                for(int j = 0; j < outputs; j++) {
                    this->dinputs[i][j] /= samples;
                }
            }
        }
};

#endif // LOSS_H