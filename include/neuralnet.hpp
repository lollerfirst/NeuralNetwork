#ifndef _NEURALNET_H
#define _NEURALNET_H

#include <component.hpp>
#include <loss.hpp>
#include <dense.hpp>
#include <activation.hpp>

#include <array>
#include <span>
#include <cstddef>

namespace nn
{
    template <typename TYPE, LossType LOSS, Component ... COMPONENTS> 
    class NeuralNet
    {
        public:
            std::array<Component, sizeof...(COMPONENTS)> layers;
            Loss<TYPE, LOSS> loss;
             
            constexpr NeuralNet() : layers{COMPONENTS}, loss{}
            {
                // Verify with a static assert that dimensions between
                // Components are compatible
            }

            // Returns the latest error after last epoch
            template <size_t TRAIN_DIM, size_t LABEL_DIM, size_t DEPTH>
            TYPE train(
                const std::array<TYPE, TRAIN_DIM*DEPTH>& train_set,
                const std::array<TYPE, LABEL_DIM*DEPTH> labels,
                size_t epochs,
                size_t batch = 1UL);

            template <size_t TEST_DIM, size_t LABEL_DIM, size_t DEPTH>
            TYPE test(
                const std::array<TYPE, TEST_DIM*DEPTH>& test_set,
                const std::array<TYPE, LABEL_DIM*DEPTH>& labels);
            
    };

    template <typename TYPE, LossType LOSS, Component ... COMPONENTS>
    template <size_t TRAIN_DIM, size_t LABEL_DIM, size_t DEPTH>
    TYPE train(
        const std::array<TYPE, TRAIN_DIM*DEPTH>& train_set,
        const std::array<TYPE, LABEL_DIM*DEPTH> labels,
        size_t epochs,
        size_t batch = 1UL)
    {
        TYPE error = 0;

        for (size_t i=0; i<epochs; ++i)
        {
            std::span<TYPE> train_set_batch_view{train_set.data(), TRAIN_DIM*batch};
            std::span<TYPE> labels_batch_view{labels.data(), LABEL_DIM*batch};

            // Train and accumulate error
        }

        return error;
    }
}

#endif