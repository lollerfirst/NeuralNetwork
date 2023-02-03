#include <loss.hpp>
#include <iostream>

int main() {

    std::array<float, 4> in_vector = {0.1, 0.5, 0.2, 0.9};
    std::array<float, 4> target_vector = {0.0, 1.0, 0.0, 1.0};

    // Calculate the loss
    auto loss = nn::calculate_loss<nn::MEAN_SQUARED>(in_vector, target_vector);
    std::cout << "Mean Squared Loss: " << loss << std::endl;

    // Calculate the gradient vector
    auto gradient_vector = nn::calculate_gradient_vector<nn::MEAN_SQUARED>(in_vector, target_vector);

    // Print the gradient vector
    std::cout << "Gradient Vector: ";
    for (auto el : gradient_vector) {
        std::cout << el << " ";
    }
    std::cout << std::endl;

    // Calculate the loss
    auto cross_entropy_loss_value = nn::calculate_loss<nn::CROSS_ENTROPY>(in_vector, target_vector);
    std::cout << "Cross Entropy Loss: " << cross_entropy_loss_value << std::endl;

    // Calculate the gradient vector
    auto gradient_cross_entropy_vector = nn::calculate_gradient_vector<nn::CROSS_ENTROPY>(in_vector, target_vector);

    // Print the gradient vector
    std::cout << "Gradient Vector: ";
    for (auto el : gradient_cross_entropy_vector) {
        std::cout << el << " ";
    }
    std::cout << std::endl;

    // Calculate the loss
    auto mean_absolute_loss_value = nn::calculate_loss<nn::MEAN_ABSOLUTE>(in_vector, target_vector);
    std::cout << "Mean Absolute Loss: " << mean_absolute_loss_value << std::endl;

    // Calculate the gradient vector
    auto gradient_mean_absolute_loss = nn::calculate_gradient_vector<nn::MEAN_ABSOLUTE>(in_vector, target_vector);

    // Print the gradient vector
    std::cout << "Gradient Vector: ";
    for (auto el : gradient_mean_absolute_loss) {
        std::cout << el << " ";
    }
    std::cout << std::endl;

    return 0;
}