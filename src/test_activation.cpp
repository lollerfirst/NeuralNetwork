#include <activation.hpp>
#include <array>
#include <iostream>
#include <algorithm>

std::array<float, 10> input{0.1, 0.09, 0.02, 0.03, 0.3, 0.5, 0.25, 0.4, 0.33, 0.11};
std::array<float, 10> target{0.15, 0.10, 0.3, 0.31, 0.2, 0.55, 0.22, 0.4, 0.35, 0.12};

int main(void)
{
    // Create activation layer

    nn::Activation<float, 10, nn::SOFTMAX> activation{};

    std::array output = activation.apply(input);

    std::array<float, 10> gradient{};

    for (int i=0; i<10; i++)
    {
        gradient[i] = target[i] - output[i];
    }

    gradient = std::move(activation.update(gradient));

    std::cout << "output: ";
    for (const auto& i : output)
    {
        std::cout << i << " ";
    }
    std::cout << "\n";

    std::cout << "gradient: ";
    for (const auto& i : gradient)
    {
        std::cout << i << " ";
    }

    std::cout << "\n";
    return 0;
}