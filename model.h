#ifndef MODEL_H
#define MODEL_H

#include <bits/stdc++.h>

#include "layers.h"

#include "activation.h"

#include "loss.h"

#include "optimizer.h"

#include "accuracy.h"

using namespace std;

// detect if T has a member named `weights`
// To check if a class has a specific variable in C++, SFINAE (Substitution Failure Is Not An Error) 
template < typename T, typename = void >
  struct has_weights: false_type {};

template < typename T >
  struct has_weights < T, void_t < decltype(declval < T > () -> weights) > >: true_type {};

class Model {
  private: bool compiled = false;
  public: vector < variant < unique_ptr < Layer_Dense > ,
  unique_ptr < Activation >>> layers;

  vector < reference_wrapper < variant < unique_ptr < Layer_Dense > ,
  unique_ptr < Activation >>> > trainable_layers;

  unique_ptr < Loss > loss;
  unique_ptr < Optimizer > optimizer;
  unique_ptr < Accuracy > accuracy;

  template < typename T >
  void add(T layer) {
    layers.push_back(make_unique < T > (move(layer)));
  }

  template < typename L,typename O,typename A >
  void set(L loss_func, O optimizer_func, A accuracy) {
    // Store loss function using polymorphism
    this -> loss = make_unique < L > (move(loss_func));
    this -> optimizer = make_unique < O > (move(optimizer_func));
    this -> accuracy = make_unique < A > (move(accuracy));
  }
  // Function to identify trainable layers (layers with weights)
  void finalize() {
    trainable_layers.clear();

    for (auto & layer: layers) {

      bool has_weights_flag = visit([](auto & lyr) {
        return has_weights < decay_t < decltype(lyr) >> ::value;
      }, layer);

      if (has_weights_flag) {
        trainable_layers.push_back(ref(layer));
      }
    }
  }

  pair < long double,long double > calculate_loss(variant < vector < vector < long double >> , vector < int >> & batch_y) {

    // Calculate data loss
    long double data_loss = visit([ & ](auto & batch_y_val) -> long double {
      return visit([ & ](auto & last_layer) -> long double {
        return this -> loss -> calculate(last_layer.get() -> output, batch_y_val);
      }, layers.back());
    }, batch_y);

    // cout<<data_loss<<" 4 "<<endl;

    // Calculate regularization loss
    long double regularization_loss = 0;
    for (auto & layer: trainable_layers) {
      regularization_loss += visit([this](auto & lyr) -> long double {
        // Check if lyr is a Layer_Dense before calculating regularization
        using T = std::decay_t < decltype( * lyr) > ;
        if constexpr(std::is_same_v < T, Layer_Dense > ) {
          // Access directly without casting
          return this -> loss -> regularization_loss( * lyr);
        } else {
          // Debug output to see what type we actually have
          cout << "Layer is not a Layer_Dense: " << typeid(T).name() << endl;
          return 0.0L; // Return 0 loss for non-Dense layers
        }
        
      }, layer.get());
    }

    return {
      data_loss,
      regularization_loss
    };
  }

  void optimize() {
    this -> optimizer.get() -> pre_update_params();

    for (auto & layer: trainable_layers) {
      visit([this](auto & lyr) {
        // Check if lyr is a Layer_Dense before casting
        using T = std::decay_t < decltype( * lyr) > ;
        if constexpr(std::is_same_v < T, Layer_Dense > ) {
          // Access directly without casting
          this -> optimizer -> update_params( * lyr);
        } else {
          // Debug output to see what type we actually have
          cout << "Layer is not a Layer_Dense: " << typeid(T).name() << endl;
        }
      }, layer.get());
    }

    // cout<<step<<" 10 "<<endl;
    this -> optimizer.get() -> post_update_params();
  }
  void backward(variant < vector < vector < long double >> , vector < int >> & batch_y) {
    visit([this](auto & layer, auto & batch_y_val) {
      this -> loss -> backward(layer.get() -> output, batch_y_val);
    }, layers.back(), batch_y);
    // cout<<step<<" 7 "<<endl;
    // Starting from the last layer, perform backward pass
    size_t last_layer_idx = layers.size() - 1;

    // First backward pass for the last layer
    visit([this](auto & layer) {
      layer.get() -> backward(this -> loss -> dinputs);
    }, layers[last_layer_idx]);
    // cout<<step<<" 8 "<<endl;
    // Backward pass through remaining layers in reverse order
    for (int i = last_layer_idx - 1; i >= 0; i--) {
      visit([this, i](auto & current_layer) {
        // Get dinputs from the next layer
        visit([ & current_layer](auto & next_layer) {
          current_layer.get() -> backward(next_layer.get() -> dinputs);
        }, layers[i + 1]);
      }, layers[i]);
    }
  }

  void forward(vector < vector < long double >> & batch_X) {
    // cout<<step<<" 1 "<<endl;
    visit([ & batch_X](auto & layer) {
      layer.get() -> forward(batch_X); // layer saves its output internally
    }, layers[0]);
    // cout<<step<<" 2 "<<endl;
    // Forward pass through remaining layers, using previous layer's output
    for (size_t i = 1; i < layers.size(); i++) {
      visit([this, i](auto & current_layer) {
        visit([ & current_layer](auto & previous_layer) {
          // cout<<"size "<<previous_layer.get()->output.size()<<" "<<previous_layer.get()->output[0].size()<<endl;
          current_layer.get() -> forward(previous_layer.get() -> output);
        }, layers[i - 1]);
      }, layers[i]);
    }
  }

  void evaluate(vector < vector < long double >> & X_val, variant < vector < vector < long double >> , vector < int >> & y_val, int batch_size = 0) {
    int validation_steps = 1;

    if (batch_size > 0) {
      validation_steps = X_val.size() / batch_size;
      if (validation_steps * batch_size < X_val.size()) validation_steps += 1;
    }
    // cout<<"Validation Steps "<<validation_steps<<endl;
    // Start the new epoch pass
    this -> loss -> new_pass();
    this -> accuracy -> new_pass();

    for (int step = 0; step < validation_steps; step++) {
      vector < vector < long double >> batch_X;
      variant < vector < vector < long double >> , vector < int >> batch_y;
    //   cout << step << endl;
      // If batch size is not set - train using full dataset
      if (batch_size <= 0) {
        batch_X = X_val;
        batch_y = y_val;
      }
      // Otherwise extract a batch
      else {
        size_t start_idx = step * batch_size;
        size_t end_idx = min(start_idx + batch_size, X_val.size());

        // Reserve space for efficiency
        batch_X.reserve(end_idx - start_idx);

        for (size_t i = start_idx; i < end_idx; i++) {
          batch_X.push_back(X_val[i]);
        }

        // Extract batch y based on its type
        visit([ & ](auto & y_values) {
          using T = decay_t < decltype(y_values) > ;

          if constexpr(is_same_v < T, vector < vector < long double >>> ) {
            vector < vector < long double >> batch_y_values;
            batch_y_values.reserve(end_idx - start_idx);

            for (size_t i = start_idx; i < end_idx; i++) {
              batch_y_values.push_back(y_values[i]);
            }

            batch_y = move(batch_y_values);
          }
          else if constexpr(is_same_v < T, vector < int >> ) {
            vector < int > batch_y_values;
            batch_y_values.reserve(end_idx - start_idx);

            for (size_t i = start_idx; i < end_idx; i++) {
              batch_y_values.push_back(y_values[i]);
            }

            batch_y = move(batch_y_values);
          }
        }, y_val);
      }
    //   cout<<step<<" 2 "<<endl;
      this -> forward(batch_X);
    //   cout<<step<<" 3 "<<endl;
      auto[data_loss, regularization_loss] = calculate_loss(batch_y);

    //   cout<<step<<" 5 "<<endl;
      // calculates accuracy
      visit([this](auto & layer, auto & batch_y_val) {
        this -> accuracy -> calculate(layer.get() -> output, batch_y_val);
      }, layers.back(), batch_y);
    //   cout<<step<<" 6 "<<endl;
    }
    cout << "validation, " <<
      "acc: " << this -> accuracy -> calculate_accumulated() << ", " <<
      "loss: " << this -> loss -> calculate_accumulated() << endl;
  }

  void train(vector < vector < long double >> & X, variant < vector < vector < long double >> , vector < int >> & y, int epochs = 1, int print_every = 1, int batch_size = 0) {
    if (!compiled) {
      throw runtime_error("Please compile the neural network object before attempting to train it.");
    }
    int train_steps = 1, validation_steps = 1;

    if (trainable_layers.empty()) {
      finalize();
    }
    // cout << trainable_layers.size() << endl;

    if (batch_size > 0) {
      train_steps = X.size() / batch_size;
      if (train_steps * batch_size < X.size()) train_steps += 1;
    }
    cout << "train_steps " << train_steps << endl;
    for (size_t epoch = 0; epoch < epochs; epoch++) {

      cout << "Epoch: " << (epoch + 1) << endl;

      // Start the new epoch pass
      this -> loss -> new_pass();
      this -> accuracy -> new_pass();

      // Forward pass through first layer with input X
      for (int step = 0; step < train_steps; step++) {

        // cout<<step<<" 0 "<<endl;
        // Create batch data containers
        vector < vector < long double >> batch_X;
        variant < vector < vector < long double >> , vector < int >> batch_y;

        // If batch size is not set - train using full dataset
        if (batch_size <= 0) {
          batch_X = X;
          batch_y = y;
        }
        // Otherwise extract a batch
        else {
          size_t start_idx = step * batch_size;
          size_t end_idx = min(start_idx + batch_size, X.size());

          // Reserve space for efficiency
          batch_X.reserve(end_idx - start_idx);

          for (size_t i = start_idx; i < end_idx; i++) {
            batch_X.push_back(X[i]);
          }

          // Extract batch y based on its type
          visit([ & ](auto & y_values) {
            using T = decay_t < decltype(y_values) > ;

            if constexpr(is_same_v < T, vector < vector < long double >>> ) {
              vector < vector < long double >> batch_y_values;
              batch_y_values.reserve(end_idx - start_idx);

              for (size_t i = start_idx; i < end_idx; i++) {
                batch_y_values.push_back(y_values[i]);
              }

              batch_y = move(batch_y_values);
            }
            else if constexpr(is_same_v < T, vector < int >> ) {
              vector < int > batch_y_values;
              batch_y_values.reserve(end_idx - start_idx);

              for (size_t i = start_idx; i < end_idx; i++) {
                batch_y_values.push_back(y_values[i]);
              }

              batch_y = move(batch_y_values);
            }
          }, y);
        }

        this -> forward(batch_X);
        // cout<<step<<" 3 "<<endl;

        auto[data_loss, regularization_loss] = calculate_loss(batch_y);

        // cout<<step<<" 5 "<<endl;
        // calculates accuracy
        auto acc = visit([this](auto & layer, auto & batch_y_val) {
          return this -> accuracy -> calculate(layer.get() -> output, batch_y_val);
        }, layers.back(), batch_y);
        // cout<<step<<" 6 "<<endl;
        // backprop

        this -> backward(batch_y);
        // cout<<step<<" 9 "<<endl;
        // update params 
        this -> optimize();

        // cout<<step<<" 11 "<<endl;
        if ((step % print_every == 0) || (step == train_steps - 1)) {
          // Format floating point to 3 decimal places
          cout << "  step: " << step << ", " <<
            "acc: " << acc << ", " <<
            "loss: " << data_loss + regularization_loss << " ( " <<
            "data_loss: " << data_loss << " , " <<
            "regularization_loss: " << regularization_loss << " ) " <<
            "lr: " << this -> optimizer -> current_learning_rate << endl;
        }
      }
      // Print epoch information
      cout << "  training, " <<
        "acc: " << this -> accuracy -> calculate_accumulated() << ", " <<
        "loss: " << this -> loss -> calculate_accumulated() << ", " <<
        "lr: " << this -> optimizer -> current_learning_rate << endl;
    }
  }

  void save(string path) {
    ofstream ofs(path, ios::binary);
    cout << "Saving Model\n";
    // Write number of layers
    size_t num_layers = layers.size();
    ofs.write(reinterpret_cast <
      const char * > ( & num_layers), sizeof(size_t));

    for (auto & layer_variant: layers) {
      visit([ & ofs](auto & layer_ptr) {
        using LayerType = decay_t < decltype( * layer_ptr) > ;

        if constexpr(is_same_v < LayerType, Layer_Dense > ) {
          // Write layer type identifier
          int type_id = 0; // 0 for Dense
          ofs.write(reinterpret_cast < char * > ( & type_id), sizeof(int));

          // Write layer dimensions
          size_t n_inputs = layer_ptr -> weights.size();
          size_t n_neurons = layer_ptr -> weights[0].size();
          ofs.write(reinterpret_cast < char * > ( & n_inputs), sizeof(size_t));
          ofs.write(reinterpret_cast < char * > ( & n_neurons), sizeof(size_t));

          // Write weights
          for (size_t i = 0; i < n_inputs; i++) {
            for (size_t j = 0; j < n_neurons; j++) {
              long double weight = layer_ptr -> weights[i][j];
              ofs.write(reinterpret_cast < char * > ( & weight), sizeof(long double));
            }
          }

          // Write biases
          for (size_t j = 0; j < n_neurons; j++) {
            long double bias = layer_ptr -> biases[j];
            ofs.write(reinterpret_cast < char * > ( & bias), sizeof(long double));
          }

        } else if constexpr(is_base_of_v < Activation, LayerType > ) {
          // Write layer type identifier
          int type_id = 1; // 1 for Activation
          ofs.write(reinterpret_cast < char * > ( & type_id), sizeof(int));

          // Write activation type
          int activation_type;
          if (typeid( * layer_ptr) == typeid(Activation_ReLU)) {
            activation_type = 0;
          } else if (typeid( * layer_ptr) == typeid(Activation_Sigmoid)) {
            activation_type = 1;
          } else if (typeid( * layer_ptr) == typeid(Activation_Softmax)) {
            activation_type = 2;
          } else if (typeid( * layer_ptr) == typeid(Activation_Linear)) {
            activation_type = 3;
          } else {
            activation_type = -1; // Unknown
          }

          ofs.write(reinterpret_cast < char * > ( & activation_type), sizeof(int));
        }
      }, layer_variant);
    }

    // Save loss function type
    int loss_type = -1;
    if (typeid( * loss) == typeid(Loss_CategoricalCrossentropy)) {
      loss_type = 0;
    } else if (typeid( * loss) == typeid(Loss_BinaryCrossentropy)) {
      loss_type = 1;
    } else if (typeid( * loss) == typeid(Loss_MeanSquaredError)) {
      loss_type = 2;
    } else if (typeid( * loss) == typeid(Loss_MeanAbsoluteError)) {
      loss_type = 3;
    }
    ofs.write(reinterpret_cast < char * > ( & loss_type), sizeof(int));

    int accuracy_type = -1;
    if (typeid( * accuracy) == typeid(Accuracy_Categorical)) {
      accuracy_type = 0;
    } else if (typeid( * accuracy) == typeid(Accuracy_Regression)) {
      accuracy_type = 1;
    }
    ofs.write(reinterpret_cast < char * > ( & accuracy_type), sizeof(int));

    cout << "Model Saved\n";
    ofs.close();
  }

  void load(string path) {
    ifstream ifs(path, ios::binary);

    if (!ifs) {
      cerr << "Error: Could not open file " << path << " for reading" << endl;

      stringstream error_msg;
      error_msg << "Error: Could not open file " << path << " for reading";
      throw runtime_error(error_msg.str());
    }
    cout << "Loading Model\n";
    // Clear existing layers
    layers.clear();
    trainable_layers.clear();

    // Read number of layers
    size_t num_layers;
    ifs.read(reinterpret_cast < char * > ( & num_layers), sizeof(size_t));
    for (size_t i = 0; i < num_layers; i++) {
      // Read layer type
      int type_id;
      ifs.read(reinterpret_cast < char * > ( & type_id), sizeof(int));

      if (type_id == 0) { // Dense Layer
        // Read dimensions
        size_t n_inputs, n_neurons;
        ifs.read(reinterpret_cast < char * > ( & n_inputs), sizeof(size_t));
        ifs.read(reinterpret_cast < char * > ( & n_neurons), sizeof(size_t));
        // cout<<"Inputs: "<<n_inputs<<" Neuron: "<<n_neurons<<endl;
        // Create weights and biases
        vector < vector < long double >> weights(n_inputs, vector < long double > (n_neurons));
        vector < long double > biases(n_neurons);

        // Read weights
        for (size_t i = 0; i < n_inputs; i++) {
          for (size_t j = 0; j < n_neurons; j++) {
            long double weight;
            ifs.read(reinterpret_cast < char * > ( & weight), sizeof(long double));
            weights[i][j] = weight;
          }
        }

        // Read biases
        for (size_t j = 0; j < n_neurons; j++) {
          long double bias;
          ifs.read(reinterpret_cast < char * > ( & bias), sizeof(long double));
          biases[j] = bias;
        }

        // // Create dense layer
        Layer_Dense dense_layer(n_inputs, n_neurons);
        dense_layer.weights = weights;
        dense_layer.biases = biases;

        // Add to layers
        add(dense_layer);

      } else if (type_id == 1) { // Activation Layer
        int activation_type;
        ifs.read(reinterpret_cast < char * > ( & activation_type), sizeof(int));

        switch (activation_type) {
        case 0: // ReLU
          add(Activation_ReLU());
          break;
        case 1: // Sigmoid
          add(Activation_Sigmoid());
          break;
        case 2: // Softmax
          add(Activation_Softmax());
          break;
        case 3: // Linear
          add(Activation_Linear());
          break;
        default:
          cerr << "Error: Unknown activation type " << activation_type << endl;
          stringstream error_msg;
          error_msg << "Error: Unknown activation type " << activation_type;

          throw runtime_error(error_msg.str());
        }
      } else {
        cerr << "Error: Unknown layer type " << type_id << endl;
        stringstream error_msg;
        error_msg << "Error: Unknown layer type " << type_id;

        throw runtime_error(error_msg.str());
      }
    }

    // Load loss function
    int loss_type;
    ifs.read(reinterpret_cast < char * > ( & loss_type), sizeof(int));

    switch (loss_type) {
    case 0: // CategoricalCrossentropy
      loss = make_unique < Loss_CategoricalCrossentropy > ();
      break;
    case 1: // BinaryCrossentropy
      loss = make_unique < Loss_BinaryCrossentropy > ();
      break;
    case 2: // MeanSquaredError
      loss = make_unique < Loss_MeanSquaredError > ();
      break;
    case 3: // MeanAbsoluteError
      loss = make_unique < Loss_MeanAbsoluteError > ();
      break;
    default:
      cerr << "Error: Unknown loss type " << loss_type << endl;
      stringstream error_msg;
      error_msg << "Error: Unknown loss type " << loss_type;

      throw runtime_error(error_msg.str());
    }

    // Load accuracy
    int accuracy_type;
    ifs.read(reinterpret_cast < char * > ( & accuracy_type), sizeof(int));

    switch (accuracy_type) {
    case 0: // Categorical
      accuracy = make_unique < Accuracy_Categorical > ();
      break;
    case 1: // Regression
      accuracy = make_unique < Accuracy_Regression > ();
      break;
    default:
      cerr << "Error: Unknown accuracy type " << accuracy_type << endl;

      stringstream error_msg;
      error_msg << "Error: Unknown accuracy type " << accuracy_type;

      throw runtime_error(error_msg.str());
    }

    // Finalize the model to rebuild trainable_layers
    finalize();
    cout << "Model Loaded\n";
    ifs.close();
  }

  void compile() {

    cout << "Compiling model..." << endl;

    compiled = false;
    int input_dim = -1;
    int output_dim = -1;

    if (layers.empty()) {
      cerr << "ERROR: Please add at least one layer to your neural network before compiling." << endl;
      throw runtime_error("ERROR: Please add at least one layer to your neural network before compiling.");
    }

    bool prev_is_activation = false;
    int prev_out = -1;

    for (size_t i = 0; i < layers.size(); i++) {
      visit([ & ](auto & layer_ptr) {
        using LayerType = decay_t < decltype( * layer_ptr) > ;

        if constexpr(is_base_of_v < Activation, LayerType > ) {
          // Current layer is an activation layer
          if (i == 0) {
            cerr << "WARNING: Applying an activation function as the first layer can distort input data." << endl;
          } else if (prev_is_activation) {
            cerr << "WARNING: Applying multiple activation functions in a row may cause issues with learning." << endl;
          }
          prev_is_activation = true;
        }
        else if constexpr(is_same_v < LayerType, Layer_Dense > ) {
          // Current layer is a dense layer
          int cur_input = layer_ptr -> weights.size();
          int cur_output = layer_ptr -> biases.size();

          if (prev_out != -1 && cur_input != prev_out) {
            cerr << "ERROR: Shape mismatch detected. Previous layer output shape (" <<
              prev_out << ") does not match current layer input shape (" <<
              cur_input << ")." << endl;

            stringstream error_msg;
            error_msg << "ERROR: Shape mismatch detected. Previous layer output shape (" <<
              prev_out << ") does not match current layer input shape (" <<
              cur_input << ").";

            throw runtime_error(error_msg.str());
          }

          prev_out = cur_output;
          prev_is_activation = false;

          if (input_dim == -1) {
            input_dim = cur_input;
          }
          output_dim = cur_output;
        }
      }, layers[i]);
    }

    // Final validation - make sure last layer has outputs
    if (output_dim <= 0) {
      cerr << "ERROR: Model output dimension is invalid." << endl;
      throw runtime_error("ERROR: Model output dimension is invalid.");
    }

    // Check if loss function is set
    if (!loss) {
      cerr << "WARNING: Loss function not set. Use the set() method before training." << endl;
      throw runtime_error("WARNING: Loss function not set. Use the set() method before training.");
    }

    // Check if optimizer is set
    if (!optimizer) {
      cerr << "WARNING: Optimizer not set. Use the set() method before training." << endl;
    }

    // Check if accuracy metric is set
    if (!accuracy) {
      cerr << "WARNING: Accuracy metric not set. Use the set() method before training." << endl;
      throw runtime_error("WARNING: Accuracy metric not set. Use the set() method before training.");
    }

    // Update trainable layers
    finalize();

    compiled = true;
    cout << "Model Compiled" << endl;
  }

};

#endif // MODEL_H