#ifndef _NEURALNET_H
#define _NEURALNET_H

#include <activation.hpp>
#include <dense.hpp>
#include <loss.hpp>
#include <span>
#include <array>

namespace nn
{
    template <typename TYPE,
        std::size_t TRAIN_DIM,
        std::size_t LABELS_DIM,
        std::size_t DEPTH,
        typename ... LAYERS>
    TYPE train(const std::array<TYPE, TRAIN_DIM*DEPTH>& train_set,
        const std::array<TYPE, LABELS_DIM*DEPTH>& labels_set,
        std::size_t batch,
        losstype_t loss,
        LAYERS... layers) noexcept
    {
        // Slice train_set and labels_set into batches
        std::span train_span{train_set.begin(), TRAIN_DIM*batch};
        std::span labels_span{labels_set.begin(), LABELS_DIM*batch};
        std::size_t i = 0;

        while (train_span.size() > 0 && labels_span.size() > 0)
        {
            // 
            // TODO: Apply for every input vector in the batch
            
            ++i;
            train_span = train_span.subspan(i*TRAIN_DIM*batch);
            labels_span = labels_span.subspan(i*LABELS_DIM*batch);
        }
    }

    
}

#endif