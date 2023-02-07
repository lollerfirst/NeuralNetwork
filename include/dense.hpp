#ifndef NN_FEEDFORWARD_H
#define NN_FEEDFORWARD_H

#include <cstddef>
#include <array>
#include <span>
#include <cassert>
#include <algorithm>
#include <random>

#ifdef __CUDA_ARCH__
#include <cublas_v2.h>
#endif

namespace nn
{

    // Dense Layer
    template <typename TYPE, std::size_t DIM1, std::size_t DIM2>
    struct Dense
    {
        public:

            std::array<TYPE, DIM1*DIM2> weight_matrix;
            std::array<TYPE, DIM2> bias_vector;
            TYPE learning_rate;

            Dense(std::initializer_list<TYPE> mat_init_list,
                std::initializer_list<TYPE> bias_init_list,
                TYPE learning_rate) : 
                learning_rate{learning_rate}
            {
                assert(mat_init_list.size() == DIM1*DIM2 && bias_init_list.size() == DIM2);
                std::copy(mat_init_list.begin(), mat_init_list.end(), weight_matrix.begin());
                std::copy(bias_init_list.begin(), bias_init_list.end(), bias_vector.begin());
            }

            constexpr Dense(std::array<TYPE, DIM1*DIM2>&& mat_init, std::array<TYPE, DIM2>&& bias_init, TYPE learning_rate) :
                weight_matrix{mat_init},
                bias_vector{bias_init},
                learning_rate{learning_rate}
                {}

            constexpr Dense(const std::array<TYPE, DIM1*DIM2>& mat_init, const std::array<TYPE, DIM2>& bias_init, TYPE learning_rate) :
                weight_matrix{mat_init},
                bias_vector{bias_init},
                learning_rate{learning_rate}
                {}
            
            Dense(TYPE learning_rate) : learning_rate{learning_rate}
            {
                std::default_random_engine engine(std::random_device{}());
                std::uniform_real_distribution<TYPE> dist(static_cast<TYPE>(0), static_cast<TYPE>(1));
                std::generate(std::begin(weight_matrix), std::end(weight_matrix), [&]{ return dist(engine); });
                std::generate(std::begin(bias_vector), std::end(bias_vector), [&]{ return dist(engine); });
            }

    };

    // Processing
    template<typename TYPE, std::size_t DIM1, std::size_t DIM2>
    std::array<TYPE, DIM2> apply(const Dense<TYPE, DIM1, DIM2>& dense, auto in_vector) noexcept
    {
        std::array<TYPE, DIM2> out_vector;

        for (std::size_t i = 0; i < DIM2; ++i)
        {
            out_vector[i] = dense.bias_vector[i];
            
            for (std::size_t j = 0; j < DIM1; ++j)
                out_vector[i] += in_vector[j] * dense.weight_matrix[i * DIM1 + j];
        }
        
        return out_vector;
    }

    // Backpropagation
    template <typename TYPE, std::size_t DIM1, std::size_t DIM2>
    std::array<TYPE, DIM1> update(Dense<TYPE, DIM1, DIM2>& dense, auto in_gradient) noexcept
    {
        std::array<TYPE, DIM1> out_gradient;

        // Compute out_gradient
        for (std::size_t i = 0; i < DIM1; ++i) {
            out_gradient[i] = 0;
            
            for (std::size_t j = 0; j < DIM2; ++j) {
                out_gradient[i] += in_gradient[j] * dense.weight_matrix[j * DIM1 + i];
            }
        }

        // Update weight_matrix
        for (std::size_t i = 0; i < DIM1; ++i) {
            for (std::size_t j = 0; j < DIM2; ++j) {
                dense.weight_matrix[j * DIM1 + i] -= dense.learning_rate * in_gradient[j] * out_gradient[i];
            }
        }

        // Update bias_vector
        for (std::size_t i = 0; i < DIM2; ++i) {
            dense.bias_vector[i] -= dense.learning_rate * in_gradient[i];
        }

        return out_gradient;
    }


}

#endif