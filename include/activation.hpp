#ifndef _ACTIVATION_H
#define _ACTIVATION_H

#include <vector>
#include <cmath>
#include <cassert>

namespace nn
{
    typedef enum
    {
        RELU,
        SOFTMAX,
        SIGMOID
    } ActMode;

    // Activation Layer
    template <typename TYPE, ActMode ACT_MODE>
    struct Activation
    {
        public:
            std::vector<TYPE> output;
            std::size_t dimension;

            constexpr Activation(size_t dim) : output{}, dimension{dim}
            {}

    };

    template <typename TYPE, ActMode ACT_MODE>
    std::vector<TYPE> apply(Activation<TYPE, ACT_MODE>& activation, const std::vector<TYPE>& in_vector) noexcept
    {
        assert(activation.dimension == in_vector.size());

        std::vector<TYPE> out_vector;
        out_vector.reserve(in_vector.size());

        if constexpr (ACT_MODE == RELU)
        {
            for (auto i = in_vector.begin(),
                j = out_vector.begin(); i != in_vector.end(); ++i, ++j)
            {
                *j = std::max(static_cast<TYPE>(0), *i);
                activation.output.push_back(*j);
            }
        }
        else if constexpr (ACT_MODE == SOFTMAX)
        {
            // Find the max value in the input vector
            auto max_val = in_vector.at(0);
            for (auto i = ++in_vector.begin(); i != in_vector.end(); ++i)
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
                *k = *j = std::exp(*i) / (1 + std::exp(*i));
            }
        }

        return out_vector;
    }

    template<typename TYPE, ActMode ACT_MODE>
    std::vector<TYPE> update(const Activation<TYPE, ACT_MODE>& activation, const std::vector<TYPE>& in_gradient) noexcept
    {
        assert(activation.dimension == in_gradient.size());

        std::vector<TYPE> out_gradient{};
        out_gradient.reserve(in_gradient.size());

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