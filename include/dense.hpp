#ifndef NN_FEEDFORWARD_H
#define NN_FEEDFORWARD_H

#include <array>
#include <cassert>
#include <algorithm>

namespace NN
{
    template <typename TYPE, int DIM1, int DIM2>
    class Dense
    {
        public:

            std::array<TYPE, DIM1*DIM2> weight_matrix;
            std::array<TYPE, DIM2> bias_vector;

            Dense(std::initializer_list<TYPE> mat_init_list,
                std::initializer_list<TYPE> bias_init_list)
            {
                assert(mat_init_list.size() == DIM1*DIM2 && bias_init_list.size() == DIM2);
                std::copy(mat_init_list.begin(), mat_init_list.end(), weight_matrix.begin());
                std::copy(bias_init_list.begin(), bias_init_list.end(), bias_vector.begin());
            }

            Dense(std::array<TYPE, DIM1*DIM2>&& mat_init, std::array<TYPE, DIM2>&& bias_init) :
                weight_matrix{std::forward(mat_init)},
                bias_vector{std::forward(bias_init)} {}

            Dense(const std::array<TYPE, DIM1*DIM2>& mat_init, const std::array<TYPE, DIM2>& bias_init) :
                weight_matrix{mat_init},
                bias_vector{bias_init} {}


            std::array<TYPE, DIM2> apply(const std::array<TYPE, DIM1>& in_vector) const noexcept;
            std::array<TYPE, DIM1> update(const std::array<TYPE, DIM2>& in_gradient, const TYPE learning_rate) noexcept;
    };

    // Processing
    template<typename TYPE, int DIM1, int DIM2>
    std::array<TYPE, DIM2> Dense<TYPE, DIM1, DIM2>::apply(const std::array<TYPE, DIM1>& in_vector) const noexcept
    {
        std::array<TYPE, DIM2> out_vector;

        for (int i = 0; i < DIM2; ++i)
        {
            out_vector[i] = bias_vector[i];
            
            for (int j = 0; j < DIM1; ++j)
                out_vector[i] += in_vector[j] * weight_matrix[i * DIM1 + j];
        }
        
        return out_vector;
    }

    // Backpropagation
    template <typename TYPE, int DIM1, int DIM2>
    std::array<TYPE, DIM1> Dense<TYPE, DIM1, DIM2>::update(const std::array<TYPE, DIM2>& in_gradient, const TYPE learning_rate) noexcept
    {
        std::array<TYPE, DIM1> out_gradient{};

        // Compute out_gradient
        for (int i = 0; i < DIM1; ++i) {
            out_gradient[i] = 0;
            for (int j = 0; j < DIM2; ++j) {
                out_gradient[i] += in_gradient[j] * weight_matrix[i * DIM2 + j];
            }
        }

        // Update weight_matrix
        for (int i = 0; i < DIM1; ++i) {
            for (int j = 0; j < DIM2; ++j) {
                weight_matrix[j * DIM1 + i] -= learning_rate * in_gradient[j] * out_gradient[i];
            }
        }

        // Update bias_vector
        for (int i = 0; i < DIM2; ++i) {
            bias_vector[i] -= learning_rate * in_gradient[i];
        }

        return out_gradient;
    }


}

#endif