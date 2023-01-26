#ifndef _NEURALNET_H
#define _NEURALNET_H

#include <component.hpp>
#include <loss.hpp>
#include <dense.hpp>
#include <activation.hpp>

#include <array>
#include <span>
#include <cstddef>
#include <algorithm>
#include <memory>

namespace nn
{
    template <typename TYPE, LossType LOSS, typename ... COMPONENTS> 
    class NeuralNet
    {
        public:
            std::array<std::unique_ptr<Component>, sizeof...(COMPONENTS)> layers;
            Loss<TYPE, LOSS> loss;
             
            constexpr NeuralNet(const std::unique_ptr<Component>& components...) :
            layers{components}, loss{}
            {}

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

    template <typename TYPE, LossType LOSS, typename ... COMPONENTS>
    template <size_t TRAIN_DIM, size_t LABEL_DIM, size_t DEPTH>
    TYPE NeuralNet<TYPE, LOSS, COMPONENTS>::train(
        const std::array<TYPE, TRAIN_DIM*DEPTH>& train_set,
        const std::array<TYPE, LABEL_DIM*DEPTH> labels,
        size_t epochs,
        size_t batch = 1UL)
    {
        TYPE error = 0;

        for (size_t i=0; i<epochs; ++i)
        {
            std::span<TYPE> train_set_batch_view{train_set.begin(), TRAIN_DIM*batch};
            std::span<TYPE> labels_batch_view{labels.begin(), LABEL_DIM*batch};

            // Train
            size_t labels_offset = 0U;
            for (size_t train_offset = 0U; train_offset < TRAIN_DIM*DEPTH;
                train_offset += TRAIN_DIM*batch, labels_offset += LABEL_DIM*batch)
            {
                std::span<TYPE> train_span = train_set_batch_view.subspan(train_offset);
                std::span<TYPE> labels_span = labels_batch_view.subspan(labels_offset);

                // Accumulate error
                size_t labels_subspan_offset = 0U;
                for (size_t train_subspan_offset = 0U; train_subspan_offset < TRAIN_DIM*batch;
                    train_subspan_offset += TRAIN_DIM, labels_subspan_offset += LABEL_DIM)
                {
                    std::span<TYPE> input_vector = train_span.subspan(train_subspan_offset);
                    std::span<TYPE> target_vector = labels_span.subspan(labels_subspan_offset);

                    for (const auto& component : this->layers)
                    {
                        switch (component->comptype)
                        {
                            case DENSE:
                            case ACTIVATION:
                            case MAXPOOL:
                            case CONV:
                        }
                    }
                }
            }
        }

        return error;
    }
}

#endif