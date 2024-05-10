#include <bits/stdc++.h>
using namespace std;


class Layer{
    public:
    void assign_random_weights(vector<long double> &w){
        random_device rd;
        mt19937 gen(rd());
        // Create a normal distribution with mean 0 and standard deviation 1
        normal_distribution<double> d(0.0, 1.0);
        for (int i = 0; i < w.size(); i++) {
            w[i] = d(gen);
        }
        b = d(gen);
    }

    

    long double sigmoid_derivative(long double x){
        //sigmoid
        return 1.0/(1.0+exp(-x));
    }

    void softmax(vector<long double> &Layer){
        //softmax
        long double sum = 0.0;
        for(int i=0;i<Layer.size();i++){
            Layer[i] = exp(Layer[i]);
            sum+=Layer[i];
        }
        for(int i=0;i<Layer.size();i++){
            Layer[i] = Layer[i]/sum;
        }
    }

    int neurons;
    vector<long double> w;
    long double b;
    Layer(int neurons){
        this->neurons = neurons;
        this->w.resize(neurons);
        assign_random_weights(this->w);
    }
};
class Sequential{
    public:
        vector<Layer>sequential;
        Sequential(int input){
            this->sequential.push_back(Layer(input));
        }
        void add_layer(int nueron){
            this->sequential.push_back(Layer(nueron));
        }
        long double sigmoid(long double x){
            //sigmoid
            return x*(1-x);
        }
        void softmax(vector<long double> &Layer){
            //softmax
            long double sum = 0.0;
            for(int i=0;i<Layer.size();i++){
                Layer[i] = exp(Layer[i]);
                sum+=Layer[i];
            }
            for(int i=0;i<Layer.size();i++){
                Layer[i] = Layer[i]/sum;
            }
        }
        long double feed_forward(){
            vector<Layer> copy_network = sequential;
            for(int i = 1;i<copy_network.size();i++){ //select layer
                for(int j = 0;j<copy_network[i].size();j++){ // select neuron of i th layer
                    long double w_sum = 0.0;
                    for(int prev = 0;prev<copy_network[i-1].size();prev){//select neuron of i-1 th layer
                        w_sum += (copy_network[i].w[j] * copy_network[i-1].w[prev]) + copy_network[i].b;  
                    }
                }
            }
            copy_network[copy_network.size()-1].w;
        }
};
int main(){
    Sequential hidden(2);
    hidden.add_layer(4);
    for(auto a : hidden.sequential){
        string s = typeid(a).name();
        cout<<s<<endl;
    }
        
    return 0;
}
