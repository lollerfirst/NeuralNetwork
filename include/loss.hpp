#ifndef _LOSS_H
#define _LOSS_H

#include <array>
#include <numeric>
#include <cmath>
#include <algorithm>

namespace nn
{
    typedef enum
    {
        MEAN_SQUARED,
        MEAN_ABSOLUTE,
        CROSS_ENTROPY
    }
    losstype_t;

    template <losstype_t LOSS_TYPE, typename TYPE, std::size_t DIM>
    std::array<TYPE, DIM> calculate_gradient_vector(const std::array<TYPE, DIM>& in_vector,
        const std::array<TYPE, DIM>& target) noexcept
    {
        std::array<TYPE, DIM> out_gradient;
        
        // Calculate derivative with respect to each input
        if constexpr (LOSS_TYPE == MEAN_SQUARED)
        {
            // (2/n) * (x-y)
            auto x = in_vector.begin();
            auto y = target.begin();

            std::for_each(out_gradient.begin(), out_gradient.end(), [=](auto& el) mutable{
                el = static_cast<TYPE>((2.0/DIM) * ((*x) - (*y)));
                ++x; ++y;
            });
        }

        if constexpr (LOSS_TYPE == MEAN_ABSOLUTE)
        {
            auto x = in_vector.begin();
            auto y = target.begin();

            auto sign = [](auto x) -> decltype(x) {
                if (x == 0) return 0;
                return (x > 0) ? 1 : -1;
            };

            std::for_each(out_gradient.begin(), out_gradient.end(), [=](auto& el) mutable{
                el = static_cast<TYPE>((1.0/DIM) * sign((*x) - (*y)));
                ++x; ++y;
            });
        }

        if constexpr (LOSS_TYPE == CROSS_ENTROPY)
        {
            auto x = in_vector.begin();
            auto y = target.begin();

            std::for_each(out_gradient.begin(), out_gradient.end(), [=](auto& el) mutable{
                el = -1.0 * ((*y) / (*x));
                ++x; ++y;
            });

        }
        
        return out_gradient;
    }

    template <losstype_t LOSS_TYPE, typename TYPE, std::size_t DIM>
    TYPE calculate_loss(const std::array<TYPE, DIM>& in_vector, const std::array<TYPE, DIM>& target_vector)
    {

        if constexpr (LOSS_TYPE == MEAN_SQUARED)
        {
            auto y = target_vector.begin();

            TYPE sum = std::accumulate(in_vector.begin(), in_vector.end(), static_cast<TYPE>(0),
                [=](auto acc, auto x) mutable{
                    auto res = acc + std::pow(x - (*y), 2);
                    ++y;

                    return res;
            });

            return (sum / static_cast<TYPE>(DIM));
        }

        if constexpr (LOSS_TYPE == MEAN_ABSOLUTE)
        {
            auto y = target_vector.begin();

            TYPE sum = std::accumulate(in_vector.begin(), in_vector.end(), static_cast<TYPE>(0),
                [=](auto acc, auto x) mutable{
                    auto res = acc + std::abs(x - (*y));
                    ++y;

                    return res;
            });

            return (sum / static_cast<TYPE>(DIM));
        }

        
        if constexpr (LOSS_TYPE == CROSS_ENTROPY)
        {
            auto y = target_vector.begin();

            TYPE sum = std::accumulate(in_vector.begin(), in_vector.end(), static_cast<TYPE>(0),
                [=](auto acc, auto x) mutable{
                    auto res = acc + (*y) * std::log(x) + (1 - (*y)) * std::log(1 - x);
                    ++y;

                    return res;
            });

            return (static_cast<TYPE>(-1) * sum / static_cast<TYPE>(DIM));
        }

    }
}

#endif