#ifndef _NEURALNET_H
#define _NEURALNET_H

#include <activation.hpp>
#include <dense.hpp>
#include <loss.hpp>
#include <span>
#include <array>
#include <tuple>
#include <ranges>

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
        typename TYPE,
        losstype_t LOSS,
        typename ... LAYERS>
    TYPE train(const std::array<TYPE, TRAIN_DIM*DEPTH>& train_set,
        const std::array<TYPE, LABELS_DIM*DEPTH>& labels_set,
        const std::size_t epochs,
        const Loss<LOSS, LABELS_DIM> loss,
        LAYERS&... layers) noexcept
    {
        // Stores the total loss after each epoch
        TYPE compounded_loss;

        // Loop over the specified number of epochs
        for (std::size_t k = 0; k<epochs; ++k)
        {
            // C++ spans are broken (or very little intuitive to use) so using raw pointers to slice. Sue me, retards.
            TYPE* train_span = std::data(std::begin(train_set));
            TYPE* labels_span = std::data(std::begin(labels_set));
            std::size_t i = 0;

            // Loop over all of the batches in the training set
            while (i < DEPTH)
            {
                TYPE* train_batch_span = std::data(std::begin(train_span));
                TYPE* labels_batch_span = std::data(std::begin(labels_span));
                std::size_t train_batch_index = 0;
                std::size_t labels_batch_index = 0;
                
                // Reset the total loss and gradient after each batch
                compounded_loss = 0;
                std::array<TYPE, LABELS_DIM> compounded_gradient;
                std::fill(std::begin(compounded_gradient), std::end(compounded_gradient), static_cast<TYPE>(0));
                
                // Loop over all samples in the batch
                while (train_batch_index < BATCH && labels_batch_index < BATCH)
                {
                    std::array<TYPE, TRAIN_DIM> train_slice;
                    std::copy(train_batch_span, train_batch_span + TRAIN_DIM, std::begin(train_slice)); 
                    std::array<TYPE, LABELS_DIM> labels_slice;
                    std::copy(labels_batch_span, labels_batch_span+LABELS_DIM, std::begin(labels_slice));

                    // Apply the layers on the sample data
                    auto statically_recursive_apply = [](auto& self, const auto& in_vector, auto& layer, auto&... layers)
                    {
                        std::array intermediate_result = nn::apply(layer, in_vector);
                        //std::span result_span{std::begin(intermediate_result), std::end(intermediate_result)};

                        if constexpr (sizeof...(layers) > 0)
                        {
                            return self(self, intermediate_result, layers...);
                        }
                        else
                        {
                            return intermediate_result;
                        }
                    };

                    std::array result = statically_recursive_apply(statically_recursive_apply, train_batch_span, layers...);
                    //std::span result_span{std::begin(result), std::end(result)};


                    // Compute and compound the loss
                    compounded_loss += calculate_loss(loss, result, labels_batch_span);
                    std::array gradient_vector = calculate_gradient_vector(loss, result, labels_batch_span);
                    
                    auto gradient_vector_iter = std::begin(gradient_vector);
                    std::for_each(std::begin(compounded_gradient), std::end(compounded_gradient), [&](auto& val){
                        val += *gradient_vector_iter;
                        ++gradient_vector_iter;
                    });

                    // Set up next iteration
                    ++labels_batch_index;
                    ++train_batch_index;
                    train_batch_span += TRAIN_DIM;
                    labels_batch_span += LABELS_DIM;
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
                        //std::span result_span{std::begin(intermediate_result), std::end(intermediate_result)};

                        return std::move(nn::update(layer, intermediate_result));
                    }
                    else
                    {
                        return std::move(nn::update(layer, in_gradient));
                    }
                };

                //std::span compounded_gradient_span{std::begin(compounded_gradient), std::end(compounded_gradient)};
                (void) statically_recursive_update(statically_recursive_update, compounded_gradient, layers...);

                // Set up next iteration
                ++i;
                train_span += TRAIN_DIM*BATCH;
                labels_span += LABELS_DIM*BATCH;
            }
        }


        return compounded_loss;
    }

    template <std::size_t TEST_DIM,
        std::size_t LABELS_DIM,
        std::size_t DEPTH,
        typename TYPE,
        losstype_t LOSS,
        typename ... LAYERS>
    TYPE test(const std::array<TYPE, TEST_DIM*DEPTH>& test_set,
        const std::array<TYPE, LABELS_DIM*DEPTH>& labels_set,
        Loss<LOSS, LABELS_DIM> loss,
        LAYERS&... layers) noexcept
        {
            TYPE compounded_loss = 0;
            std::span<TYPE, TEST_DIM> test_span{std::ranges::all_of()};
            std::span<TYPE, LABELS_DIM> labels_span; labels_span{std::begin(labels_set), LABELS_DIM};
            std::size_t i = 0;

            while (std::size(test_span) > 0 && std::size(labels_span) > 0)
            {
                // Apply the layers on the sample data
                auto statically_recursive_apply = [](auto& self, const auto& in_vector, auto& layer, auto&... layers)
                {
                    std::array intermediate_result = nn::apply(layer, in_vector);
                    //std::span span_result{std::begin(intermediate_result), std::end(intermediate_result)};

                    if constexpr (sizeof...(layers) > 0)
                    {
                        return self(self, intermediate_result, layers...);
                    }
                    else
                    {
                        return intermediate_result;
                    }
                };

                std::array result = statically_recursive_apply(statically_recursive_apply, test_span, layers...);
                //std::span span_result{std::begin(result), std::end(result)};
                compounded_loss += calculate_loss(loss, result, labels_span);

                ++i;
                test_span = test_span.subspan(TEST_DIM*i, TEST_DIM);
                labels_span = labels_span.subspan(LABELS_DIM*i, LABELS_DIM);
            }
            return (compounded_loss / static_cast<TYPE>(DEPTH));
        }
    
    
    template <std::size_t DATA_DIM, std::size_t DATA_DEPTH, typename TYPE>
    void split_data_labels(std::array<TYPE, DIM*DEPTH> data,
        std::array<TYPE, (DATA_DIM-1)*DEPTH> train_set,
        std::array<TYPE, DEPTH>)
    {
        
    }
}

#endif