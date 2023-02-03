#ifndef NN_FEEDFORWARD_H
#define NN_FEEDFORWARD_H

#include <cstddef>
#include <vector>
#include <algorithm>

namespace nn
{

    // Dense Layer
    template <typename TYPE>
    struct Dense
    {
        public:

            std::vector<TYPE> weight_matrix;
            std::vector<TYPE> bias_vector;
            TYPE learning_rate;

            Dense(std::initializer_list<TYPE> mat_init_list,
                std::initializer_list<TYPE> bias_init_list,
                TYPE learning_rate) : 
                learning_rate{learning_rate}
            {
                static_assert(mat_init_list.size() % bias_init_list.size() == 0);
                std::copy(mat_init_list.begin(), mat_init_list.end(), weight_matrix.begin());
                std::copy(bias_init_list.begin(), bias_init_list.end(), bias_vector.begin());
            }

            constexpr Dense(std::vector<TYPE>&& mat_init, std::vector<TYPE>&& bias_init, TYPE learning_rate) :
                weight_matrix{mat_init},
                bias_vector{bias_init},
                learning_rate{learning_rate}
                {}

            constexpr Dense(const std::vector<TYPE>& mat_init, const std::vector<TYPE>& bias_init, TYPE learning_rate) :
                weight_matrix{mat_init},
                bias_vector{bias_init},
                learning_rate{learning_rate}
                {}

    };

    // Processing
    template<typename TYPE>
    std::vector<TYPE> apply(const Dense<TYPE>& dense, const std::vector<TYPE>& in_vector) noexcept
    {

        static_assert(dense.weight_matrix.size() / dense.bias_vector.size() == in_vector.size());
        std::vector<TYPE> out_vector;
        out_vector.reserve(dense.bias_vector.size());

        for (std::size_t i = 0; i < dense.bias_vector.size(); ++i)
        {
            out_vector[i] = dense.bias_vector[i];
            
            for (std::size_t j = 0; j < in_vector.size(); ++j)
                out_vector[i] += in_vector[j] * dense.weight_matrix[i * in_vector.size() + j];
        }
        
        return out_vector;
    }

    // Backpropagation
    template <typename TYPE>
    std::vector<TYPE> update(Dense<TYPE>& dense, const std::vector<TYPE>& in_gradient) noexcept
    {
        static_assert(in_gradient.size() == dense.bias_vector.size());

        std::vector<TYPE> out_gradient{};
        std::size_t out_gradient_size = dense.weight_matrix.size() / dense.bias_vector.size();
        out_gradient.reserve(out_gradient_size);

        // Compute out_gradient
        for (std::size_t i = 0; i < out_gradient_size; ++i) {
            out_gradient[i] = 0;
            
            for (std::size_t j = 0; j < dense.bias_vector.size(); ++j) {
                out_gradient[i] += in_gradient[j] * weight_matrix[j * out_gradient_size + i];
            }
        }

        // Update weight_matrix
        for (std::size_t i = 0; i < out_gradient_size; ++i) {
            for (std::size_t j = 0; j < dense.bias_vector.size(); ++j) {
                dense.weight_matrix[j * out_gradient_size + i] -= dense.learning_rate * in_gradient[j] * out_gradient[i];
            }
        }

        // Update bias_vector
        for (std::size_t i = 0; i < dense.bias_vector.size(); ++i) {
            dense.bias_vector[i] -= dense.learning_rate * in_gradient[i];
        }

        return out_gradient;
    }


}

#endif