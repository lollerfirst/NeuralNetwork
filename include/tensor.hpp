#ifndef _TENSOR_H
#define _TENSOR_H

#include <concepts>
#include <array>

namespace nn
{
    template <typename TYPE, std::size_t ... DIMS>
    struct tensor
    {
        std::array<TYPE, (DIMS * ...)> data;
        
        constexpr tensor(){}

        tensor(TYPE* in_data_ptr, bool in_place = false)
        {
            (in_place) ? data.data = in_data_ptr : std::copy(data.begin(), data.end(), in_data_ptr);
        }

        tensor(const tensor<TYPE, DIMS...>& tns)
        {
            std::copy(data.begin(), data.end(), tns.data.begin());
        }
        
        tensor(tensor<TYPE, DIMS...>&& tns)
        {
            this->data = tns.data;
            tns.data.data = nullptr;
        }


        template <std::integral ... DIMSl>
        TYPE at(DIMSl ... indices)
        {
            static_assert(sizeof...(DIMSl) == sizeof...(DIMS));

            auto total_index = ((indices * DIMS) + ...);

            return data[total_index];
        }

        /*
        template <std::size_t ... DIMSl>
        tensor<TYPE, DIMSl...> slice(){}
        */
    };

    
}

#endif