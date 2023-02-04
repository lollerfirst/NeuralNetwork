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
    template <typename TYPE,
        std::size_t TRAIN_DIM,
        std::size_t LABELS_DIM,
        std::size_t DEPTH,
        typename ... LAYERS>
    TYPE train(const std::array<TYPE, TRAIN_DIM*DEPTH>& train_set,
        const std::array<TYPE, LABELS_DIM*DEPTH>& labels_set,
        const std::size_t batch,
        const std::size_t epochs,
        const losstype_t loss,
        LAYERS... layers) noexcept
    {
    
        for (std::size_t k = 0; k<epochs; ++k)
        {
            // Slice train_set and labels_set into batches
            std::span train_span{train_set.begin(), TRAIN_DIM*batch};
            std::span labels_span{labels_set.begin(), LABELS_DIM*batch};
            std::size_t i = 0;

            while (std::size(train_span) > 0 && std::size(labels_span) > 0)
            {
                // TODO: Apply for every input vector in the batch
                std::span train_batch_span{train_span.begin(), TRAIN_DIM};
                std::span labels_batch_span{labels_span.begin(), LABELS_DIM};
                std::size_t j = 0;
                
                TYPE compounded_loss = 0;
                std::array<TYPE, LABELS_DIM> compounded_gradient;
                std::fill(std::begin(compounded_gradient), std::end(compounded_gradient), static_cast<TYPE>(0));
                
                while (std::size(train_batch_span) > 0 && std::size(labels_batch_span) > 0)
                {
                    std::array<TYPE, TRAIN_DIM> train_array_slice{std::begin(train_batch_span), std::end(train_batch_span)};
                    std::array<TYPE, LABELS_DIM> labels_array_slice{labels_batch_span};

                    // Calls apply for every layer in the parameter pack
                    auto statically_recursive_apply = [&statically_recursive_apply](const auto& in_vector, auto& layer, auto&... layers)
                    {
                        std::array intermediate_result = nn::apply(layer, in_vector);

                        if constexpr (sizeof...(layers) > 0)
                        {
                            return statically_recursive_apply(intermediate_result, layers...);
                        }
                        else
                        {
                            return intermediate_result;
                        }
                    };

                    std::array result = statically_recursive_apply(train_array_slice, layers...);

                    // Compute and compound the loss
                    compounded_loss += calculate_loss<loss>(result, labels_array_slice);
                    std::array gradient_vector = calculate_gradient_vector<loss>(result, labels_array_slice);
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
                compounded_loss /= static_cast<TYPE>(batch);
                std::for_each(std::begin(compounded_gradient), std::end(compounded_gradient), [&](auto& val){
                    val /= static_cast<TYPE>(batch);
                });

                
                // Call update for every layer
                auto statically_recursive_update = [&statically_recursive_update](const auto& in_gradient, auto& layer, auto&... layers)
                {
                    std::array intermediate_result = nn::update(layer, in_gradient);

                    if constexpr (sizeof...(layers) > 0)
                    {
                        statically_recursive_update(intermediate_result, layers...);
                    }
                };

                statically_recursive_update(compounded_gradient, layers...);

                // Set up next iteration
                ++i;
                train_span = train_span.subspan(i*TRAIN_DIM*batch);
                labels_span = labels_span.subspan(i*LABELS_DIM*batch);
            }
        }
    }

    
}

#endif