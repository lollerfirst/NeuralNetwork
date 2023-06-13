#include <iostream>
#include <algorithm>
#include <neuralnet.hpp>
#include <cassert>

#define DIM1 10UL
#define DIM2 10UL
#define DIM3 2UL
#define DEPTH 10UL

#define BATCH 1UL
#define EPOCHS 100UL

int main(void)
{
    // Define the input and label sets
    std::array<float, DIM1*DEPTH> train_set; /* = {...}; */
    std::array<float, DIM3*DEPTH> labels_set; /* = {...}; */
    
    // Define the layers in the network
    nn::Dense<float, DIM1, DIM2> dense_1{0.01f};
    nn::Dense<float, DIM2, DIM3> dense_2{0.01f};
    nn::Activation<float, nn::SIGMOID, DIM2> activation_1;
    nn::Activation<float, nn::RELU, DIM3> activation_2;

    // Define the loss function
    nn::Loss<nn::MEAN_SQUARED, DIM3> loss;
    
    // Call the train function
    float train_error = nn::train<DIM1, DIM3, DEPTH, BATCH>(train_set, labels_set, EPOCHS, loss, dense_1, activation_1, dense_2, activation_2);

    // Call the test function
    float test_error = nn::test<DIM1, DIM3, DEPTH>(train_set, labels_set, loss, dense_1, activation_1, dense_2, activation_2);
    
    // Check if the result is within the expected range
    std::cout << "train_loss:  " << train_error << "\n";
    std::cout << "test_loss: " << test_error << "\n"; 
}
