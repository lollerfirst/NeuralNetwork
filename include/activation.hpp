#ifndef _ACTIVATION_H
#define _ACTIVATION_H

#include <array>
#include <cmath>

namespace nn
{
    typedef enum
    {
        RELU,
        SOFTMAX,
        SIGMOID
    } ActMode;

    // Activation Layer
    template <typename TYPE, int DIM, ActMode ACT_MODE>
    class Activation
    {
        public:
            std::array<TYPE, DIM> output;

            Activation() : output{} {}

            std::array<TYPE, DIM> apply(const std::array<TYPE, DIM>&) noexcept;
            std::array<TYPE, DIM> update(const std::array<TYPE, DIM>&) const noexcept;
    };

    template <typename TYPE, int DIM, ActMode ACT_MODE>
    std::array<TYPE, DIM> Activation<TYPE, DIM, ACT_MODE>::apply(const std::array<TYPE, DIM>& in_vector) noexcept
    {
        std::array<TYPE, DIM> out_vector{};

        if constexpr (ACT_MODE == RELU)
        {
            for (int i = 0; i < DIM; ++i)
            {
                this->output[i] = out_vector[i] = std::max(static_cast<TYPE>(0), in_vector[i]);
            }
        }
        else if constexpr (ACT_MODE == SOFTMAX)
        {
            // Find the max value in the input vector
            TYPE max_val = in_vector[0];
            for (int i = 1; i < DIM; ++i) {
                max_val = std::max(max_val, in_vector[i]);
            }
            // Subtract the max value from all input values
            for (int i = 0; i < DIM; ++i) {
                out_vector[i] = in_vector[i] - max_val;
            }
            // Compute exponentials and sum them up
            TYPE sum = 0.0;
            for (int i = 0; i < DIM; ++i)
            {
                out_vector[i] = std::exp(out_vector[i]);
                sum += out_vector[i];
            }

            // Normalize the output vector
            for (int i = 0; i < DIM; ++i)
            {
                this->output[i] = (out_vector[i] /= sum);
            }
        }
        else if constexpr (ACT_MODE == SIGMOID)
        {
            // SIGMOID IMPLEMENTATION
            for (int i=0; i < DIM; ++i)
            {
                this->output[i] = out_vector[i] = std::exp(in_vector[i]) / (1 + std::exp(in_vector[i]));
            }
        }

        return out_vector;
    }

    template<typename TYPE, int DIM, ActMode ACT_MODE>
    std::array<TYPE, DIM> Activation<TYPE, DIM, ACT_MODE>::update(const std::array<TYPE, DIM>& in_gradient) const noexcept
    {
        std::array<TYPE, DIM> out_gradient{};

        if constexpr (ACT_MODE == RELU)
        {
            for (int i=0; i<DIM; i++)
            {
                out_gradient[i] = (this->output[i] > 0) ? in_gradient[i] : 0;
            }
        }
        else if constexpr (ACT_MODE == SOFTMAX || ACT_MODE == SIGMOID)
        {
            for (int i=0; i<DIM; i++)
            {
                out_gradient[i] = in_gradient[i] * (this->output[i] * (1 - this->output[i]));
            }
        }

        return out_gradient;
    }
}

#endif