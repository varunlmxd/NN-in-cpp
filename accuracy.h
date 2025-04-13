#ifndef ACCURACY_H
#define ACCURACY_H

#include <bits/stdc++.h>
using namespace std;

class Accuracy {
public:
    virtual ~Accuracy() = default;

    long double accumulated_sum,accumulated_count;
    
    void new_pass(){
        this->accumulated_sum = 0;
        this->accumulated_count = 0;     
    }
    
    // Base calculate method that calls compare and returns the mean
    template<typename PredictionType, typename TruthType>
    long double calculate(const PredictionType& predictions, const TruthType& y) {
        // Get comparison results (1 for correct, 0 for incorrect)
        vector<int> comparisons = compare(predictions, y);
        
        // Calculate mean (sum of comparisons divided by number of samples)
        long double sum = accumulate(comparisons.begin(), comparisons.end(), 0.0);

        this->accumulated_sum += sum;
        this->accumulated_count += comparisons.size();

        return sum / comparisons.size();
    }

    long double calculate_accumulated(){
        return this->accumulated_sum / this->accumulated_count ;
    }
    
    // Virtual compare methods to be overridden by derived classes
    virtual vector<int> compare(const vector<vector<long double>>& predictions, const vector<int>& y) {
        throw runtime_error("This accuracy metric doesn't support this combination of types");
    }
    
    virtual vector<int> compare(const vector<vector<long double>>& predictions, const vector<vector<int>>& y) {
        throw runtime_error("This accuracy metric doesn't support this combination of types");
    }
    
    virtual vector<int> compare(const vector<vector<long double>>& predictions, const vector<vector<long double>>& y) {
        throw runtime_error("This accuracy metric doesn't support this combination of types");
    }
    
    virtual vector<int> compare(const vector<long double>& predictions, const vector<int>& y) {
        throw runtime_error("This accuracy metric doesn't support this combination of types");
    }
    
    virtual vector<int> compare(const vector<long double>& predictions, const vector<long double>& y) {
        throw runtime_error("This accuracy metric doesn't support this combination of types");
    }
};

// Categorical accuracy - for classification with multiple classes
class Accuracy_Categorical : public Accuracy {
public:
    // For 2D predictions and 1D class indices
    vector<int> compare(const vector<vector<long double>>& predictions, const vector<int>& y) override {
        vector<int> comparisons;
        comparisons.reserve(predictions.size());
        
        for (size_t i = 0; i < predictions.size(); i++) {
            // Find index of max value in prediction (predicted class)
            auto max_element_idx = max_element(predictions[i].begin(), predictions[i].end()) - predictions[i].begin();
            
            // Compare with true class
            comparisons.push_back(max_element_idx == y[i] ? 1 : 0);
        }
        
        return comparisons;
    }
};

// Regression accuracy - for regression problems
class Accuracy_Regression : public Accuracy {
    public:
        // Precision property initialized to nullptr
        long double precision = 0.0;  // Will be calculated from data
        
        // Default constructor
        Accuracy_Regression() {}
        
        // Helper function to calculate standard deviation
        long double calculate_std(const vector<vector<long double>>& y) {
            // First calculate mean
            long double sum = 0.0;
            long double count = 0.0;
            
            for (const auto& row : y) {
                for (const auto& val : row) {
                    sum += val;
                    count++;
                }
            }
            
            long double mean = sum / count;
            
            // Now calculate variance
            long double variance_sum = 0.0;
            for (const auto& row : y) {
                for (const auto& val : row) {
                    variance_sum += pow(val - mean, 2);
                }
            }
            
            return sqrt(variance_sum / count);
        }
        
        long double calculate_std(const vector<long double>& y) {
            // Calculate mean
            long double sum = accumulate(y.begin(), y.end(), 0.0);
            long double mean = sum / y.size();
            
            // Calculate variance
            long double variance_sum = 0.0;
            for (const auto& val : y) {
                variance_sum += pow(val - mean, 2);
            }
            
            return sqrt(variance_sum / y.size());
        }
        
        // Initialize precision based on ground truth
        void init(const vector<vector<long double>>& y, bool reinit = false) {
            if (precision == 0.0 || reinit) {
                precision = calculate_std(y) / 250.0;
            }
        }
        
        void init(const vector<long double>& y, bool reinit = false) {
            if (precision == 0.0 || reinit) {
                precision = calculate_std(y) / 250.0;
            }
        }
        
        // For 2D predictions and 2D continuous values
        vector<int> compare(const vector<vector<long double>>& predictions, const vector<vector<long double>>& y) override {
            // Initialize precision if needed
            if (precision == 0.0) {
                init(y);
            }
            
            vector<int> comparisons;
            comparisons.reserve(predictions.size());
            
            for (size_t i = 0; i < predictions.size(); i++) {
                bool sample_correct = true;
                for (size_t j = 0; j < predictions[i].size(); j++) {
                    // Check if prediction is within precision of true value
                    if (abs(predictions[i][j] - y[i][j]) >= precision) {
                        sample_correct = false;
                        break;
                    }
                }
                comparisons.push_back(sample_correct ? 1 : 0);
            }
            
            return comparisons;
        }
        
        // For 1D predictions and 1D continuous values
        vector<int> compare(const vector<long double>& predictions, const vector<long double>& y) override {
            // Initialize precision if needed
            if (precision == 0.0) {
                init(y);
            }
            
            vector<int> comparisons;
            comparisons.reserve(predictions.size());
            
            for (size_t i = 0; i < predictions.size(); i++) {
                // Check if prediction is within precision of true value
                comparisons.push_back(abs(predictions[i] - y[i]) < precision ? 1 : 0);
            }
            
            return comparisons;
        }
    };

#endif // ACCURACY_H