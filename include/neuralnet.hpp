#ifndef _NEURALNET_H
#define _NEURALNET_H

#include <activation.hpp>
#include <dense.hpp>
#include <loss.hpp>
#include <span>
#include <array>
#include <tuple>

namespace nn
{

    // Template function for training a neural network
    // The function trains a neural network on a given training set and labels set
    // The function uses the specified number of epochs and layers
    // The function calculates the compounded loss after each epoch and updates the layers

    template <std::size_t TRAIN_DIM,
        std::size_t LABELS_DIM,
        std::size_t DEPTH,
        std::size_t BATCH,
        losstype_t LOSS,
        typename TYPE,
        typename ... LAYERS>
    TYPE train(const std::array<TYPE, TRAIN_DIM*DEPTH>& train_set,
        const std::array<TYPE, LABELS_DIM*DEPTH>& labels_set,
        const std::size_t epochs,
        LAYERS&... layers) noexcept
    {
        // Stores the total loss after each epoch
        TYPE compounded_loss;

        // Loop over the specified number of epochs
        for (std::size_t k = 0; k<epochs; ++k)
        {
            // Slice the train_set and labels_set into smaller batches
            std::span train_span{std::begin(train_set), TRAIN_DIM*BATCH};
            std::span labels_span{std::begin(labels_set), LABELS_DIM*BATCH};
            std::size_t i = 0;

            // Loop over all the batches in the training set
            while (std::size(train_span) > 0 && std::size(labels_span) > 0)
            {
                std::span train_batch_span {std::begin(train_span), TRAIN_DIM};
                std::span labels_batch_span {std::begin(labels_span), LABELS_DIM};
                std::size_t j = 0;
                
                // Reset the total loss and gradient after each batch
                compounded_loss = 0;
                std::array<TYPE, LABELS_DIM> compounded_gradient;
                std::fill(std::begin(compounded_gradient), std::end(compounded_gradient), static_cast<TYPE>(0));
                
                // Loop over all samples in the batch
                while (std::size(train_batch_span) > 0 && std::size(labels_batch_span) > 0)
                {
                    std::array<TYPE, TRAIN_DIM> train_slice;
                    std::copy(std::begin(train_batch_span), std::end(train_batch_span), std::begin(train_slice)); 
                    std::array<TYPE, LABELS_DIM> labels_slice;
                    std::copy(std::begin(labels_batch_span), std::end(labels_batch_span), std::begin(labels_slice));

                    // Apply the layers on the sample data
                    auto statically_recursive_apply = [](auto& self, const auto& in_vector, auto& layer, auto&... layers)
                    {
                        std::array intermediate_result = nn::apply(layer, in_vector);

                        if constexpr (sizeof...(layers) > 0)
                        {
                            return self(self, intermediate_result, layers...);
                        }
                        else
                        {
                            return intermediate_result;
                        }
                    };

                    std::array result = statically_recursive_apply(statically_recursive_apply, train_slice, layers...);

                    // Compute and compound the loss
                    compounded_loss += calculate_loss<LOSS>(result, labels_slice);
                    std::array gradient_vector = calculate_gradient_vector<LOSS>(result, labels_slice);
                    
                    auto gradient_vector_iter = std::begin(gradient_vector);
                    std::for_each(std::begin(compounded_gradient), std::end(compounded_gradient), [&](auto& val){
                        val += *gradient_vector_iter;
                        ++gradient_vector_iter;
                    });

                    // Set up next iteration
                    ++j;
                    train_batch_span = train_batch_span.subspan(TRAIN_DIM * j);
                    labels_batch_span = labels_batch_span.subspan(LABELS_DIM * j);
                }

                // Divide loss and elements of gradient by batch size
                compounded_loss /= static_cast<TYPE>(BATCH);
                std::for_each(std::begin(compounded_gradient), std::end(compounded_gradient), [&](auto& val){
                    val /= static_cast<TYPE>(BATCH);
                });

                
                // Update every layer with the computed gradient
                auto statically_recursive_update = [](auto& self, const auto& in_gradient, auto& layer, auto&... layers)
                {
                
                    if constexpr (sizeof...(layers) > 0)
                    {
                        std::array intermediate_result = self(self, in_gradient, layers...);
                        return std::move(nn::update(layer, intermediate_result));
                    }
                    else
                    {
                        return std::move(nn::update(layer, in_gradient));
                    }
                };

                (void) statically_recursive_update(statically_recursive_update, compounded_gradient, layers...);

                // Set up next iteration
                ++i;
                train_span = train_span.subspan(i*TRAIN_DIM*BATCH);
                labels_span = labels_span.subspan(i*LABELS_DIM*BATCH);
            }
        }


        return compounded_loss;
    }

    template <std::size_t TEST_DIM,
        std::size_t LABELS_DIM,
        std::size_t DEPTH,
        losstype_t LOSS,
        typename TYPE,
        typename ... LAYERS>
    TYPE test(const std::array<TYPE, TEST_DIM*DEPTH>& test_set,
        const std::array<TYPE, LABELS_DIM*DEPTH>& labels_set,
        LAYERS&... layers) noexcept
        {
            TYPE compounded_loss = 0;
            std::span test_span{std::begin(test_set), TEST_DIM};
            std::span labels_span{std::begin(labels_set), LABELS_DIM};
            std::size_t i = 0;

            while (std::size(test_span) > 0 && std::size(labels_span) > 0)
            {
                std::array<TYPE, TEST_DIM> test_slice;
                std::copy(std::begin(test_span), std::end(test_span), std::begin(test_slice)); 
                std::array<TYPE, LABELS_DIM> labels_slice;
                std::copy(std::begin(labels_span), std::end(labels_span), std::begin(labels_slice));

                // Apply the layers on the sample data
                auto statically_recursive_apply = [](auto& self, const auto& in_vector, auto& layer, auto&... layers)
                {
                    std::array intermediate_result = nn::apply(layer, in_vector);

                    if constexpr (sizeof...(layers) > 0)
                    {
                        return self(self, intermediate_result, layers...);
                    }
                    else
                    {
                        return intermediate_result;
                    }
                };

                std::array result = statically_recursive_apply(statically_recursive_apply, test_slice, layers...);
                compounded_loss += calculate_loss<LOSS>(result, labels_slice);

                ++i;
                test_span = test_span.subspan(TEST_DIM*i);
                labels_span = labels_span.subspan(LABELS_DIM*i);
            }
            return (compounded_loss / static_cast<TYPE>(DEPTH));
        }
    
}

#endif