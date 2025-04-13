#ifndef LAYERS_H
#define LAYERS_H

#include <bits/stdc++.h>
#include "utils.h"
using namespace std;
  
class Layer_Dense{
    public: 
        vector<vector<long double>> weights, inputs, output, dweights, dinputs, weight_momentums, weight_cache ;
        vector<long double> biases, dbiases, bias_momentums, bias_cache;
        long double weight_regularizer_l1, weight_regularizer_l2, bias_regularizer_l1, bias_regularizer_l2;

        Layer_Dense(int n_inputs, int n_neurons, long double weight_regularizer_l1 = 0, long double bias_regularizer_l1 = 0, long double weight_regularizer_l2 = 0, long double bias_regularizer_l2 = 0){

            const long double scale = sqrt(2.0 / n_inputs);
            this->weights.resize(n_inputs, vector<long double>(n_neurons));
            
            // Initialize random number generator
            mt19937 gen(0);
            normal_distribution<long double> d(0.0, 1.0);

            for(int i=0; i<n_inputs; i++) {
                for(int j=0; j<n_neurons; j++) {
                    weights[i][j] = scale*d(gen);
                }
            }
            this->biases.resize(n_neurons, 0);

            this->weight_regularizer_l1 = weight_regularizer_l1;
            this->weight_regularizer_l2 = weight_regularizer_l2;
            this->bias_regularizer_l1 = bias_regularizer_l1;
            this->bias_regularizer_l2 = bias_regularizer_l2;
        }

        #pragma omp parallel for reduction(+:M2) collapse(2)
        void forward(vector<vector<long double>> &inputs) {
            // cout<<"Layer "<<inputs.size()<<endl;
            this->inputs = inputs;
            this->output = dot(inputs, weights);
            
            add(this->output, this->biases);
        }
        
        #pragma omp parallel for reduction(+:M2) collapse(2)
        void backward(vector<vector<long double>> &dvalues) {
            // Gradient Calculation of params
            this->dweights = dot(transpose(this->inputs), dvalues);
            dbiases.resize(biases.size(), 0);

            for(int i=0; i<dvalues.size(); i++) {
                for(int j=0; j<dvalues[0].size(); j++) {
                    dbiases[j] += dvalues[i][j];
                }
            }

            // Gradients on regularization
            if(this->weight_regularizer_l1 > 0){
                for(int i=0; i<dweights.size(); i++) {
                    for(int j=0; j<dweights[0].size(); j++) {
                        this->dweights[i][j] += this->weight_regularizer_l1 * ( this->weights[i][j]<0?-1:1);
                    }
                }                
            }

            if(this->weight_regularizer_l2 > 0){
                for(int i=0; i<dweights.size(); i++){
                    for(int j=0; j<dweights[0].size(); j++){
                        this->dweights[i][j] += 2 * this->weight_regularizer_l2 * this->weights[i][j];
                    }
                } 
            }

            if(this->bias_regularizer_l1 > 0){
                for(int i=0; i<dbiases.size(); i++){
                    this->dbiases[i] += this->bias_regularizer_l1 * ( this->biases[i]<0?-1:1);
                }                
            }

            if(this->bias_regularizer_l2 > 0){
                for(int i=0; i<dbiases.size(); i++){
                    this->dbiases[i] += 2 * this->bias_regularizer_l2 * this->biases[i];
                } 
            }

            // Gradient on values
            this->dinputs = dot(dvalues, transpose(this->weights));
        }

        void print() {
            for(int i=0; i<this->output.size(); i++) {
                for(int j=0; j<this->output[0].size(); j++) {
                    cout<<output[i][j]<<" ";
                }
                cout<<endl;
            }
        }
};

#endif // LAYERS_H