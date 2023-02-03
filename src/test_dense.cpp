#include <dense.hpp>
#include <iostream>
#include <array>

#define DIM1 10
#define DIM2 2

std::array<float, DIM1> input = {0.1, 0.09, 0.02, 0.03, 0.3, 0.5, 0.25, 0.4, 0.33, 0.11};
std::array<float, DIM2> target = {0.15, 0.10};

std::array<float, DIM1*DIM2> initial_weights = {0.1, 0.09, 0.02, 0.03, 0.3, 0.5, 0.25, 0.4, 0.33, 0.11,
                            0.15, 0.10, 0.3, 0.31, 0.2, 0.55, 0.22, 0.4, 0.35, 0.12};

std::array<float, DIM2> initial_biases = {0.1, 0.09};

int main(void)
{
    nn::Dense<float, DIM1, DIM2> dense{std::move(initial_weights), std::move(initial_biases), 0.01f};
    
    // Print the weights and biases before the update
    std::cout << "Weights before:\n";
    for (int i=0; i<DIM1; ++i)
    {
        for (int j=0; j<DIM2; ++j)
        {
            std::cout << dense.weight_matrix[i*DIM2 + j] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";


    std::cout << "Biases before:\n";
    for (int i=0; i<DIM2; ++i)
    {
        std::cout << dense.bias_vector[i] << " ";
    }

    std::cout << "\n\n";

    // Process
    std::array output = nn::apply(dense, input);

    // print the output
    std::cout << "Output: ";
    for (const auto& i : output)
    {
        std::cout << i << " ";
    }
    std::cout << "\n\n";

    // Calculate Error
    std::array<float, DIM2> error;
    std::cout << "Error: ";
    for (int i=0; i<DIM2; ++i)
    {
        error[i] = target[i] - output[i];
        std::cout << error[i] << " ";
    }
    std::cout << "\n\n";

    std::array gradient = nn::update(dense, error);

    // Print the weights and biases after the update
    std::cout << "Weights after:\n";
    for (int i=0; i<DIM1; ++i)
    {
        for (int j=0; j<DIM2; ++j)
        {
            std::cout << dense.weight_matrix[i*DIM2 + j] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";


    std::cout << "Biases after:\n";
    for (int i=0; i<DIM2; ++i)
    {
        std::cout << dense.bias_vector[i] << " ";
    }

    std::cout << "\n";

    return 0;
}