#include <activation.hpp>
#include <vector>
#include <iostream>
#include <algorithm>
#include <cassert>

std::vector<float> input{0.1, 0.09, 0.02, 0.03, 0.3, 0.5, 0.25, 0.4, 0.33, 0.11};
std::vector<float> target{0.15, 0.10, 0.3, 0.31, 0.2, 0.55, 0.22, 0.4, 0.35, 0.12};

int main(void)
{
    // Create activation layer
    nn::Activation<float, nn::RELU> activation{input.size()};

    std::vector output = std::move(nn::apply(activation, input));

    assert(output.size() == target.size());

    std::vector<float> gradient;

    for (auto i = output.begin(), j = target.begin();
        i != output.end(); ++i, ++j)
    {
        // fictional gradient
        gradient.push_back((*i) - (*j));
    }

    gradient = std::move(nn::update(activation, gradient));

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