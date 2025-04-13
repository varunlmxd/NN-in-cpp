#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <bits/stdc++.h>
#include "layers.h"
using namespace std;


class Optimizer {
    public:
        long double learning_rate, current_learning_rate, decay;
        int iterations;
        
        Optimizer(long double learning_rate = 0.001, long double decay = 0.0) 
            : learning_rate(learning_rate), current_learning_rate(learning_rate), 
              decay(decay), iterations(0) {}
        
        virtual ~Optimizer() = default;
        
        virtual void pre_update_params() {
            if(this->decay) {
                this->current_learning_rate = this->learning_rate * 
                    ((long double)1.0 / (1.0 + this->decay * this->iterations));
            }
        }
        
        virtual void update_params(Layer_Dense &layer) = 0;
        
        virtual void post_update_params() {
            this->iterations += 1;
        }
};

class Optimizer_SGD : public Optimizer {
    public:
        long double momentum;
        
        Optimizer_SGD(long double learning_rate=1.0, long double decay=0.0, long double momentum=0.0) 
            : Optimizer(learning_rate, decay), momentum(momentum) {}

        void update_params(Layer_Dense &layer) {
            if(this->momentum > 0.0){
                
                // initialize with zeros
                if(layer.weight_momentums.empty()) {
                    layer.weight_momentums.resize(layer.weights.size(), vector<long double>(layer.weights[0].size(), 0.0));
                    layer.bias_momentums.resize(layer.biases.size(), 0.0);
                }

                // Update weights with momentum
                for(int i=0; i<layer.weights.size(); i++) {
                    for(int j=0; j<layer.weights[0].size(); j++) {
                        // Build up momentum
                        layer.weight_momentums[i][j] = (this->momentum * layer.weight_momentums[i][j]) - (this->current_learning_rate * layer.dweights[i][j]);
                        // Update weights with momentum values
                        layer.weights[i][j] += layer.weight_momentums[i][j];
                    }
                }

                // Update biases with momentum
                for(int i=0; i<layer.biases.size(); i++) {
                    // Build up momentum
                    layer.bias_momentums[i] = (this->momentum * layer.bias_momentums[i]) - (this->current_learning_rate * layer.dbiases[i]);
                    // Update biases with momentum values
                    layer.biases[i] += layer.bias_momentums[i];
                }                
            }
            // Vanilla SGD updates
            else{
                for(int i=0; i<layer.weights.size(); i++) {
                    for(int j=0; j<layer.weights[0].size(); j++) {
                        layer.weights[i][j] -= (this->current_learning_rate*layer.dweights[i][j]);
                    }
                }

                for(int i=0; i<layer.biases.size(); i++) {
                    layer.biases[i] -= (this->current_learning_rate*layer.dbiases[i]);
                }
            }
        }
};


class Optimizer_Adagrad : public Optimizer {
    public:
        long double epsilon;
        
        Optimizer_Adagrad(long double learning_rate=1.0, long double decay=0.0, long double epsilon=1e-7)
            : Optimizer(learning_rate, decay), epsilon(epsilon) {}

        void update_params(Layer_Dense &layer) {
                
            // initialize with zeros
            if(layer.weight_cache.empty()) {
                layer.weight_cache.resize(layer.weights.size(), vector<long double>(layer.weights[0].size(), 0.0));
                layer.bias_cache.resize(layer.biases.size(), 0.0);
            }

            // Update cache with squared current gradients
            for(int i=0; i<layer.weight_cache.size(); i++) {
                for(int j=0; j<layer.weight_cache[0].size(); j++) {
                    layer.weight_cache[i][j] += pow(layer.dweights[i][j],2);
                }
            }

            for(int i=0; i<layer.bias_cache.size(); i++) {
                layer.bias_cache[i] += pow(layer.dbiases[i],2);
            }                  
            
            for(int i=0; i<layer.weights.size(); i++) {
                for(int j=0; j<layer.weights[0].size(); j++) {
                    layer.weights[i][j] -= this->current_learning_rate * layer.dweights[i][j] / (sqrt(layer.weight_cache[i][j]) + this->epsilon) ;
                }
            }

            for(int i=0; i<layer.biases.size(); i++) {
                layer.biases[i] -= this->current_learning_rate * layer.dbiases[i] / (sqrt(layer.bias_cache[i]) + this->epsilon) ;
            }               
        }
};


class Optimizer_RMSprop : public Optimizer {
    public:
        long double epsilon, rho;
        
        Optimizer_RMSprop(long double learning_rate=0.001, long double decay=0.0, long double epsilon=1e-7, long double rho=0.9)
            : Optimizer(learning_rate, decay), epsilon(epsilon), rho(rho) {}

        void update_params(Layer_Dense &layer) {
                
            // initialize with zeros
            if(layer.weight_cache.empty()) {
                layer.weight_cache.resize(layer.weights.size(), vector<long double>(layer.weights[0].size(), 0.0));
                layer.bias_cache.resize(layer.biases.size(), 0.0);
            }

            // Update cache with squared current gradients
            for(int i=0; i<layer.weight_cache.size(); i++) {
                for(int j=0; j<layer.weight_cache[0].size(); j++) {
                    layer.weight_cache[i][j] = this->rho * layer.weight_cache[i][j] + (1 - this->rho) * pow(layer.dweights[i][j],2);
                }
            }

            for(int i=0; i<layer.bias_cache.size(); i++) {
                layer.bias_cache[i] = this->rho * layer.bias_cache[i] + (1 - this->rho) * pow(layer.dbiases[i],2);
            }                  
            
            // Vanilla SGD parameter update + normalization with square rooted cache

            for(int i=0; i<layer.weights.size(); i++) {
                for(int j=0; j<layer.weights[0].size(); j++) {
                    layer.weights[i][j] -= this->current_learning_rate * layer.dweights[i][j] / (sqrt(layer.weight_cache[i][j]) + this->epsilon) ;
                }
            }

            for(int i=0; i<layer.biases.size(); i++) {
                layer.biases[i] -= this->current_learning_rate * layer.dbiases[i] / (sqrt(layer.bias_cache[i]) + this->epsilon) ;
            }               
        }
};


class Optimizer_Adam : public Optimizer {  
    public:
        long double epsilon, beta_1, beta_2;
        
        Optimizer_Adam(long double learning_rate=0.001, long double decay=0.0, long double epsilon=1e-7, long double beta_1=0.9, long double beta_2=0.999)
            : Optimizer(learning_rate, decay), epsilon(epsilon), beta_1(beta_1), beta_2(beta_2) {}

        void update_params(Layer_Dense &layer) {
            // cout<<"Optimizer\n"<<layer.biases.size()<<endl;
            // initialize with zeros
            if(layer.weight_cache.empty()) {
                
                layer.weight_cache.resize(layer.weights.size(), vector<long double>(layer.weights[0].size(), 0.0));
                layer.bias_cache.resize(layer.biases.size(), 0.0);

                layer.weight_momentums.resize(layer.weights.size(), vector<long double>(layer.weights[0].size(), 0.0));
                layer.bias_momentums.resize(layer.biases.size(), 0.0);
            }
            // Update cache with squared current gradients

            for(int i=0; i<layer.weight_momentums.size(); i++) {
                for(int j=0; j<layer.weight_momentums[0].size(); j++) {
                    layer.weight_momentums[i][j] = this->beta_1 * layer.weight_momentums[i][j] + (1 - this->beta_1) * layer.dweights[i][j];
                }
            }
            for(int i=0; i<layer.bias_momentums.size(); i++) {
                layer.bias_momentums[i] = this->beta_1 * layer.bias_momentums[i] + (1 - this->beta_1) * layer.dbiases[i];
            }
            
            vector<vector<long double>> weight_momentums_corrected,weight_cache_corrected;
            weight_momentums_corrected = layer.weight_momentums;
            weight_cache_corrected = layer.weight_cache;

            vector<long double> bias_momentums_corrected,bias_cache_corrected; 
            bias_momentums_corrected = layer.bias_momentums;
            bias_cache_corrected = layer.bias_cache;

            // Get corrected momentum this->iteration is 0 at first pass and we need to start with +1 here
            for(int i=0; i<weight_momentums_corrected.size(); i++){
                for(int j=0; j<weight_momentums_corrected[0].size(); j++) {
                    weight_momentums_corrected[i][j] = layer.weight_momentums[i][j] / (1 - pow(this->beta_1, this->iterations+1));
                }
            }

            for(int i=0; i<bias_momentums_corrected.size(); i++){
                bias_momentums_corrected[i] = layer.bias_momentums[i] / (1 - pow(this->beta_1, this->iterations+1));
            }  
            // Update cache with squared current gradients
            for(int i=0; i<layer.weight_cache.size(); i++){
                for(int j=0; j<layer.weight_cache[0].size(); j++){
                    layer.weight_cache[i][j] = this->beta_2 * layer.weight_cache[i][j] + (1 - this->beta_2) * pow(layer.dweights[i][j],2);
                }
            }

            for(int i=0; i<layer.bias_cache.size(); i++){
                layer.bias_cache[i] = this->beta_2 * layer.bias_cache[i] + (1 - this->beta_2) * pow(layer.dbiases[i],2);
            }                  

            //Get corrected cache
            for(int i=0; i<weight_cache_corrected.size(); i++){
                for(int j=0; j<weight_cache_corrected[0].size(); j++) {
                    weight_cache_corrected[i][j] = layer.weight_cache[i][j] / (1 - pow(this->beta_2, this->iterations+1));
                }
            }

            for(int i=0; i<bias_cache_corrected.size(); i++){
                bias_cache_corrected[i] = layer.bias_cache[i] / (1 - pow(this->beta_2, this->iterations+1));
            }  

            // Vanilla SGD parameter update + normalization with square rooted cache
            for(int i=0; i<layer.weights.size(); i++) {
                for(int j=0; j<layer.weights[0].size(); j++) {
                    layer.weights[i][j] -= this->current_learning_rate * weight_momentums_corrected[i][j] / (sqrt(weight_cache_corrected[i][j]) + this->epsilon);
                }
            }

            for(int i=0; i<layer.biases.size(); i++) {
                layer.biases[i] -= this->current_learning_rate * bias_momentums_corrected[i] / (sqrt(bias_cache_corrected[i]) + this->epsilon);
            }              
        }
};


#endif // OPTIMIZER_H