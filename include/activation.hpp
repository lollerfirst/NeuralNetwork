#ifndef _ACTIVATION_H
#define _ACTIVATION_H

#include <array>
#include <cmath>
#include <cassert>

namespace nn
{
    typedef enum
    {
        RELU,
        SOFTMAX,
        SIGMOID
    } actmode_t;

    // Activation Layer
    template <typename TYPE, actmode_t ACT_MODE, std::size_t DIM>
    struct Activation
    {
        public:
            std::array<TYPE, DIM> output;

    };

    template <typename TYPE, actmode_t ACT_MODE, std::size_t DIM>
    std::array<TYPE, DIM> apply(Activation<TYPE, ACT_MODE, DIM>& activation, const std::array<TYPE, DIM>& in_vector) noexcept
    {
        std::array<TYPE, DIM> out_vector;

        if constexpr (ACT_MODE == RELU)
        {
            for (auto i = in_vector.begin(),
                j = out_vector.begin(),
                k = activation.output.begin(); i != in_vector.end(); ++i, ++j, ++k)
            {
                *k = *j = std::max(static_cast<TYPE>(0), *i);
            }
        }
        else if constexpr (ACT_MODE == SOFTMAX)
        {
            // Find the max value in the input vector
            auto max_val = in_vector[0];
            for (auto i = in_vector.begin(); i != in_vector.end(); ++i)
            {
                max_val = std::max(max_val, *i);
            }

            // Subtract the max value from all input values
            for (auto i = in_vector.begin(),
                j = out_vector.begin(); i != in_vector.end(); ++i, ++j)
            {  
                *j = *i - max_val;
            }

            // Compute exponentials and sum them up
            auto sum = 0;
            for (auto i = out_vector.begin(); i != out_vector.end(); ++i)
            {
                sum += std::exp(*i);
            }

            // Normalize the output vector
            for (auto i = out_vector.begin(),
                j = activation.output.begin(); i != out_vector.end(); ++i, ++j)
            {
                *j = (*i /= sum);
            }
        }
        else if constexpr (ACT_MODE == SIGMOID)
        {
            // SIGMOID IMPLEMENTATION
            for (auto i = in_vector.begin(), j = out_vector.begin(), k = activation.output.begin();
                i != in_vector.end(); ++i, ++j, ++k)
            {
                *k = *j = (TYPE)1 / ((TYPE)1 + std::exp(*i));
            }
        }

        return out_vector;
    }

    template<typename TYPE, actmode_t ACT_MODE, std::size_t DIM>
    std::array<TYPE, DIM> update(const Activation<TYPE, ACT_MODE, DIM>& activation, const std::array<TYPE, DIM>& in_gradient) noexcept
    {
        std::array<TYPE, DIM> out_gradient{};

        if constexpr (ACT_MODE == RELU)
        {
            for (auto i = in_gradient.begin(), j = out_gradient.begin(), k = activation.output.begin(); i != in_gradient.end(); ++i)
            {
                *j = (*k > 0) ? *i : 0;
            }
        }
        else if constexpr (ACT_MODE == SOFTMAX || ACT_MODE == SIGMOID)
        {
            for (auto i = in_gradient.begin(), j = out_gradient.begin(), k = activation.output.begin(); i != in_gradient.end(); ++i)
            {
                *j = (*i) * ((*k) * (1 - *k));
            }
        }

        return out_gradient;
    }
}

#endif