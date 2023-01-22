#include <loss.hpp>
#include <iostream>

int main() {

    std::array<float, 4> in_vector = {0.1, 0.5, 0.2, 0.9};
    std::array<float, 4> target_vector = {0.0, 1.0, 0.0, 1.0};

    // Create an object of the Loss class for MEAN_SQUARED loss
    nn::Loss<float, 4, nn::MEAN_SQUARED> mean_squared_loss;

    // Calculate the loss
    auto loss = mean_squared_loss.calculate_loss(in_vector, target_vector);
    std::cout << "Mean Squared Loss: " << loss << std::endl;

    // Calculate the gradient vector
    auto gradient_vector = mean_squared_loss.calculate_gradient_vector(in_vector, target_vector);

    // Print the gradient vector
    std::cout << "Gradient Vector: ";
    for (auto el : gradient_vector) {
        std::cout << el << " ";
    }
    std::cout << std::endl;

    // Create an object of the Loss class for CROSS_ENTROPY loss
    nn::Loss<float, 4, nn::CROSS_ENTROPY> cross_entropy_loss;

    // Calculate the loss
    auto cross_entropy_loss_value = cross_entropy_loss.calculate_loss(in_vector, target_vector);
    std::cout << "Cross Entropy Loss: " << cross_entropy_loss_value << std::endl;

    // Calculate the gradient vector
    auto gradient_cross_entropy_vector = cross_entropy_loss.calculate_gradient_vector(in_vector, target_vector);

    // Print the gradient vector
    std::cout << "Gradient Vector: ";
    for (auto el : gradient_cross_entropy_vector) {
        std::cout << el << " ";
    }
    std::cout << std::endl;


    // Create an object of the Loss class for CROSS_ENTROPY loss
    nn::Loss<float, 4, nn::MEAN_ABSOLUTE> mean_absolute_loss;

    // Calculate the loss
    auto mean_absolute_loss_value = mean_absolute_loss.calculate_loss(in_vector, target_vector);
    std::cout << "Mean Absolute Loss: " << mean_absolute_loss_value << std::endl;

    // Calculate the gradient vector
    auto gradient_mean_absolute_loss = mean_absolute_loss.calculate_gradient_vector(in_vector, target_vector);

    // Print the gradient vector
    std::cout << "Gradient Vector: ";
    for (auto el : gradient_mean_absolute_loss) {
        std::cout << el << " ";
    }
    std::cout << std::endl;

    return 0;
}