# bANN

**bANN** is a header-only C library for playing with neural networks (using the CPU). It's meant to be simple, small, relatively fast, and have few extra features compared to similar minimal libraries. I've made it because I need it for some small **toy projects**, and I've released it under the MIT licence. I don't think YOU may need it (considering), but, well, who knows :-)

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
	in.kind=iokind_standard;
	in.linear=(real*)malloc(sizeof(in.linear[0])*INPUT_SIZE);
	out.kind=iokind_standard;
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

For single-label classification, set the final layer to `bann_activation_softmax`
and call `bann_setloss(&N, bann_loss_cross_entropy)` before training. Pass a
one-hot target or a distribution over classes. Squared error is the default
and applies to non-softmax outputs. Set the loss again if you resume training
from a saved model.

Being a header-only librarey, remember that in your project you need to add 

	#define BANN_IMPLEMENTATION
	#include "bann.h"

ONCE to compiler also its C code

If you just want to use a bANN quantized file (generated with bANN) you can include just this 

	#define BANNQ_IMPLEMENTATION
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
	
bANNQ can also run 8-bit recurrent models saved by bANN. Call `bannQ_reset(&NQ)` before predicting a new independent sequence.
	

# Examples

In examples folder there are few source files to easily check quality and speed of the little library. 

There you can find:

### XOR (xor.c)

Minimal sample to get a XOR behaviour from a **2-2-1** ann. It also used to quantized save, and bANNQ (quantized bANN) for an extra run.

### Sparse text classification (text-classifier.c)

Classifies short messages about cooking, weather, or software. The example
hashes byte trigrams and sends only occupied bins and their counts via
`iokind_fidx`. It trains with softmax and cross-entropy on an embedded corpus,
then compares float and 8-bit predictions on nine held-out messages.

    cc -std=c11 -O2 -I. examples/text-classifier.c -lm -o text-classifier
    ./text-classifier

The example writes `text-classifier.bannQ`. Pass a path as the first argument
to choose where to save the model.

### Recurrent memory (rnn-memory.c)

Given a sequence such as `B x y x ?`, the network recalls the first symbol
when it sees `?`. Only the query has a target. Backpropagation through time
carries its error across the distractors. The example reports accuracy for
different delays, including sequences longer than those used in training.

    cc -std=c11 -O2 -I. examples/rnn-memory.c -lm -o rnn-memory
    ./rnn-memory

### Continuous bag of words (cbow.c)

Predicts a missing word from up to two neighboring words on either side. The
input is a sparse average of the context words. After training with a linear
hidden layer and softmax cross-entropy, the example reads the embeddings from
the first weight matrix and prints the nearest words by cosine similarity.
It uses a small embedded corpus and a full-vocabulary softmax.

    cc -std=c11 -O2 -I. examples/cbow.c -lm -o cbow
    ./cbow

### MNIST (fashion-mnist.c) 

It's more or less the "standard" test for anns. Here a **784-48-10** ann reaches around **96.48%** accuracy with **8** epochs, and that's a matter of few minutes.

### FASHION-MNIST (fashion-mnist.c) 

A more complex task, still done with a common dataset. Here to get a little more than **87.20%** accuracy you need a **784-96-10** and around **8** epochs.

(same source file - you need to uncomment #define FASHION_TEST to use this instead of mnist)

### TAGGER (tagger.c) 

This is NOT a standard test for ann but a toy project to see how **bann_io** works (and how anns can be used to get THREE - or more - return values).  This example uses **[ConLLU 2003 English dataset](https://www.clips.uantwerpen.be/conll2003/ner/)** to create a minimal **POS/NER tagger**. Code shows how a moving window of context is passed as input, and how three set of data (POS, CHUNK and NER) are retrieved as output.

Input is **100k** cell wide BUT, using bann_io structure, training (and usage) keeps the average speed not too slow even on a small PC. And saving ann in a quantized way makes also the memory usage relatively small.

Results - with such a simple implementation - are far from astonishing (**85.4** of accuracy considering all the three fields together - **92.3%** POS, **94.2%** CHUNK and **81.8%** NER as **F1** considering each field in a separate way) but here we don't use ANY outside information - no embeddings or any statistical data or dictionary outside what we took from train dataset.

These are quality values for test dataset (after **32** epochs):

	NER	F1	PREC	RECALL	CORRECT	MISS	ADD
	I-ORG	77.33	77.63	77.04	1919	553	572
	I-MISC	70.58	69.67	71.51	650	283	259
	I-PER	88.45	88.23	88.68	2459	328	314
	I-LOC	83.97	82.54	85.46	1640	347	279

These are quality values for **dev** dataset:

	NER	F1	PREC	RECALL	CORRECT	MISS	ADD
	I-ORG	80.70	84.36	77.34	1618	300	474
	I-MISC	80.23	82.58	78.01	986	208	278
	I-PER	91.37	91.15	91.58	2884	280	265
	I-LOC	87.25	86.77	87.73	1837	280	257

Out of curiosity, there are values for test dataset for the other two fields (**POS** and **CHUNK/TYPE**)

	NNP	93.41	90.92	96.06	8256	825	339
	VBZ	77.61	87.50	69.72	350	50	152
	JJ	76.24	79.53	73.21	1752	451	641
	NN	87.00	85.64	88.40	4359	731	572
	TO	99.51	99.75	99.27	812	2	6
	VB	86.88	90.68	83.39	778	80	155
	.	99.94	99.88	100.00	1630	2	0
	CD	98.26	97.44	99.09	5908	155	54
	DT	98.95	98.89	99.00	2771	31	28
	VBD	88.95	89.56	88.35	1501	175	198
	IN	97.99	97.77	98.21	3946	90	72
	PRP	97.83	98.98	96.69	585	6	20
	NNS	87.19	85.83	88.59	1926	318	248
	VBP	82.74	89.75	76.74	254	29	77
	MD	97.94	98.13	97.76	262	5	6
	VBN	77.07	77.11	77.02	667	198	199
	POS	96.77	94.51	99.14	344	20	3
	JJR	64.39	58.41	71.74	66	47	26
	"	99.88	99.76	100.00	421	1	0
	RB	81.19	86.11	76.80	682	110	206
	,	100.00	100.00	100.00	1637	0	0
	FW	24.66	22.50	27.27	9	31	24
	CC	99.28	99.87	98.69	755	1	10
	WDT	91.43	94.12	88.89	96	6	12
	(	99.78	99.85	99.71	686	1	2
	)	99.85	100.00	99.71	684	0	2
	:	100.00	100.00	100.00	599	0	0
	PRP$	98.66	97.68	99.66	295	7	1
	VBG	76.05	79.19	73.14	354	93	130
	WP	92.50	88.10	97.37	111	15	3
	WRB	94.12	91.14	97.30	72	7	2
	$	97.33	97.85	96.81	91	2	3
	RP	67.35	73.33	62.26	66	24	40
	NNPS	37.33	64.62	26.25	42	23	118
	SYM	96.23	94.26	98.29	115	7	2
	JJS	6.06	20.00	3.57	2	8	54

	O	95.30	93.63	97.02	6026	410	185
	I-NP	96.86	95.70	98.05	27744	1248	552
	I-VP	90.07	91.42	88.76	5006	470	634
	I-PP	95.55	94.97	96.13	3797	201	153
	B-NP	74.36	85.46	65.82	570	97	296
	I-SBAR	76.76	81.30	72.70	213	49	80
	I-ADJP	52.80	65.77	44.11	146	76	185
	I-ADVP	67.91	75.05	62.01	364	121	223
	I-PRT	70.59	76.60	65.45	72	22	38
	B-PP	62.50	69.44	56.82	25	11	19
