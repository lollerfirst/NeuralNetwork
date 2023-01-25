#ifndef _COMPONENT_H
#define _COMPONENT_H

namespace nn
{
    typedef enum
    {
        ACTIVATION,
        DENSE,
        MAXPOOL,
        CONV
    }
    comptype_t;


    class Component
    {
        public:
            comptype_t comptype;
    };
}

#endif