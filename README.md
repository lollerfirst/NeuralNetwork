## A simple Feed Forward Neural Network

*** work in progress ***
The ultimate objective is to have a components which are composable in whatever order in the Neural Network.

A Layer is:
  * an Activation
  * a Dense Layer
  * (soon) a Convolution
  * (soon) a MaxPool
 
They will all inherit from a generic class Component which will have __apply__ and __update__ methods.
