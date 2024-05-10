# bANN

Nothing fancy here, just a header-only C library to play with neural networks with few twists. 

This library is similar to / inspired by gennann and Tinn. 

I use a cached sigmoid for speed and a single allocation for all the weights (like gennann) but I added a basic optimizer (that makes a lot of difference in several cases) and the possibility to save and work with a 8bit quantized network, after the training.

# Usage

I kept everything very simple:

  bann N;
  bann_new(&N,INPUT_SIZE,HIDDEN_SIZE,HIDDEN_NUMBER,OUTPUT_SIZE);

  bann_train(&N,input,output);

  output=bann_predict(&N,input);
  
  bann_deleye(&N);

Once in your project you need to add 

  #define BANN_IMPLEMENTATION
  #include "bann.h"

to compiler also its C code


# Examples

In examples folder there are few source files to easily check quality and speed of the little library. There you can find:
- XOR
- MNIST
- FASHIN-MNIST
