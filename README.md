# bANN

bANN is just a header-only simple C library to play with neural networks (using CPU only). 

This library is similar to / inspired by gennann and Tinn - surely more readable than mine. 

Like gennann, bANN uses cached sigmoids for speed (if those are used) and a single allocation for all the weights. 

But it has also a basic optimizer (that makes a lot of difference in several cases) and the possibility to save and work also with a 8bit quantized network, after the training. 

You can save & read the quantized file version with standard bANN (so you're just saving disk space) but you can also play it with bANNQ (a minimal autonomous derived library) that with dequantize data on-the-fly, using less memory (the price is of course a bit of speed & quality)

# Usage

I kept everything very simple:

    // declare the network
    bann N;
	// create it
    bann_new(&N,INPUT_SIZE,HIDDEN_SIZE,HIDDEN_NUMBER,OUTPUT_SIZE);
	// train it
	...	
	for(epoch=0;epoch<maxepochs;epoch++)
		for(sample=0;sample<maxsamples;sample++)
			...
			bann_train(&N,input,output);		
			...
	...
	// then use it
    output=bann_predict(&N,input);
	// and of course you need to delete it in the end
    bann_delete(&N);
	
You can save and read the network using bann_write and bann_read (there's also a bann_corewrite and bann_coreread if you want to stream it inside an opened FILE*)

Being a header-only librarey, remember that in your project you need to add 

  #define BANN_IMPLEMENTATION
  #include "bann.h"

ONCE to compiler also its C code

# Examples

In examples folder there are few source files to easily check quality and speed of the little library. 

There you can find:
- XOR: minimal sample to get a XOR behaviour from a 2-2-1 ann. Here it's shown how to change a layer activation function (using tanh instead of sigmoid for the first part of the network) and it's also used to quantized save, and bANNQ (quantized bANN) for an extra run
- MNIST: it's more or less the "standard" test for anns. Here a 784-64-10 ann reaches 96.25% accuracy with 5 epochs, and that's a matter of few minutes.
- FASHIN-MNIST: a more complex task, still done with a common dataset. Here a similar 784-64-10 ann needs to reach 96.25% accuracy.
