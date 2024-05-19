# bANN

**bANN** is just a header-only C library for playing with neural networks (using only the CPU). It's meant to be simple, small, relatively fast, and have few extra features compared to similar minimal libraries. I've made it because I need it for some small toy projects, and I've released it under the MIT licence because I've used, studied and loved other open source libraries and wanted to give something back.

This library is similar to / inspired by other great C libraries, [genann](https://github.com/codeplea/genann) and [tinn](https://github.com/glouw/tinn), certainly more readable, used and tested than mine. 

Like genann, bANN uses cached sigmoids for speed and a single allocation for all weights. 

But it also has a basic optimiser (which makes a big difference in most cases) and the ability to save and work with an 8bit quantized network after training. 

You can save & read the quantized file version with standard bANN (so you're just saving disk space), but you can also play it with bANNQ (a minimal autonomous derived library) that dequantize data on-the-fly, using less memory (the price is of course a bit of speed & quality).

Another difference is the fact that **bANN** let you use as input/output an array of active cells - instead of a whole array of values. IMHO, more clear, usable and fast - of course just using CPU only code.

# Usage

I tried to keep everything very simple:

    // declare the network
    bann N;
	// declare input / output variables
	bann_io in,out;
	// create it
    bann_new(&N,INPUT_SIZE,HIDDEN_SIZE,HIDDEN_NUMBER,OUTPUT_SIZE);
	// create in/out vars
	// these vars have unions - you can use them in a standard way (as usual) OR
	// with a indexes array - useful when most of input or output are 0s and the few other 1s
	in.kind=standard;
	in.linear=(real*)malloc(sizeof(in.linear[0])*INPUT_SIZE);
	out.kind=standard;
	out.linear=(real*)malloc(sizeof(out.linear[0])*OUTPUT_SIZE);
	// train it
	...	
	for(epoch=0;epoch<maxepochs;epoch++)
		for(sample=0;sample<maxsamples;sample++)
			...
			bann_train(&N,&in,&out);		
			...
	...
	// then use it
    output=bann_predict(&N,&in);
	// and of course you need to delete it in the end
	free(in.linear);
	free(out.linear);
    bann_delete(&N);
	
You can save and read the network using bann_write and bann_read (there's also a bann_corewrite and bann_coreread if you want to stream it inside an opened FILE*)

Being a header-only librarey, remember that in your project you need to add 

	#define BANN_IMPLEMENTATION
	#include "bann.h"

ONCE to compiler also its C code

If you just want to use a bANN quantized file (generated with bANN) you can include just this 

	#define BANN_IMPLEMENTATION
	#include "bannQ.h"
	
	// declare the network
	bannQ NQ;
	// you can't create it empty - you can only create it reading a file
	if(bannQ_read(&NQ,"bin/xor.bannQ"))
	{
		...
		output=bannQ_predict(&NQ,&input);   
		...
		bannQ_delete(&NQ); 
	}
	
if bANN code is anyway minimal - bANNQ code is way smaller so you don't have any reason to avoid usign a bit of anns capabilities if you need them
	

# Examples

In examples folder there are few source files to easily check quality and speed of the little library. 

There you can find:

### XOR (xor.c)

Minimal sample to get a XOR behaviour from a **2-2-1** ann. It also used to quantized save, and bANNQ (quantized bANN) for an extra run.

### MNIST (fashion-mnist.c) 

It's more or less the "standard" test for anns. Here a **784-48-10** ann reaches around **96.70%** accuracy with **8** epochs, and that's a matter of few minutes.

### FASHIN-MNIST (fashion-mnist.c) 

A more complex task, still done with a common dataset. Here to get a little more than **87.20%** accuracy you need a **784-96-10** and around **8** epochs.

(same source file - you need to uncomment #define FASHION_TEST to use this instead of mnist)

### TAGGER (tagger.c) 

This is NOT a standard test for ann but a toy project to see how **bann_io** works (and how anns can be used to get TWO - or more - return values).  This example uses **ConLLU 2003 English dataset** to create a minimal **POS/NER tagger**. Code shows how a moving window of context is passed as input, and how two set of data (POS and NER) are retrieved as output.

Input is **125741** cell wide BUT, using bann_io structure, training (and usage) keeps the average speed not too slow even on a small PC. And saving ann in a quantized way makes also the memory usage relatively small.

Results - with such a simple implementation - are far from astonishing (92% for POS and 78% (F1) for NER) but here we don't use embeddings nor RNN (so everything should be faster and smaller) 