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
	
if bANN code is anyway minimal - bANNQ code is way smaller so you don't have any reason to avoid usign a bit of anns capabilities if you need them
	

# Examples

In examples folder there are few source files to easily check quality and speed of the little library. 

There you can find:

### XOR (xor.c)

Minimal sample to get a XOR behaviour from a **2-2-1** ann. It also used to quantized save, and bANNQ (quantized bANN) for an extra run.

### MNIST (fashion-mnist.c) 

It's more or less the "standard" test for anns. Here a **784-48-10** ann reaches around **96.48%** accuracy with **8** epochs, and that's a matter of few minutes.

### FASHIN-MNIST (fashion-mnist.c) 

A more complex task, still done with a common dataset. Here to get a little more than **87.20%** accuracy you need a **784-96-10** and around **8** epochs.

(same source file - you need to uncomment #define FASHION_TEST to use this instead of mnist)

### TAGGER (tagger.c) 

This is NOT a standard test for ann but a toy project to see how **bann_io** works (and how anns can be used to get THREE - or more - return values).  This example uses **[ConLLU 2003 English dataset](https://www.clips.uantwerpen.be/conll2003/ner/)** to create a minimal **POS/NER tagger**. Code shows how a moving window of context is passed as input, and how three set of data (POS, CHUNK and NER) are retrieved as output.

Input is **100k** cell wide BUT, using bann_io structure, training (and usage) keeps the average speed not too slow even on a small PC. And saving ann in a quantized way makes also the memory usage relatively small.

Results - with such a simple implementation - are far from astonishing (**84.9%** of accuracy considering all the three fields together - **92.2%** POS, **94.1%** CHUNK and **78.1%** NER as **F1** considering each field in a separate way) but here we don't use ANY outside information - no embeddings or any statistical data or dictionary outside what we took from train dataset.

These are quality values for test dataset (after **32** epochs):

	NER	F1	PREC	RECALL	CORRECT	MISS	ADD
	I-ORG	74.67	72.59	76.88	1915	723	576
	I-MISC	68.87	66.80	71.07	646	321	263
	I-PER	85.25	85.04	85.47	2370	417	403
	I-LOC	78.39	78.52	78.27	1502	411	417

These are quality values for **dev** dataset:

	NER	F1	PREC	RECALL	CORRECT	MISS	ADD
	I-ORG	76.03	75.72	76.34	1597	512	495
	I-MISC	75.31	80.23	70.97	897	221	367
	I-PER	87.91	88.53	87.30	2749	356	400
	I-LOC	81.99	82.33	81.66	1710	367	384

Out of curiosity, there are values for test dataset for the other two fields (**POS** and **CHUNK/TYPE**)

	NNP	92.87	90.88	94.95	8161	819	434
	VBZ	78.09	88.41	69.92	351	46	151
	JJ	74.76	76.57	73.05	1748	535	645
	NN	86.96	86.56	87.37	4308	669	623
	TO	99.20	100.00	98.41	805	0	13
	VB	86.05	89.00	83.28	777	96	156
	.	99.91	99.94	99.88	1628	1	2
	CD	98.21	97.27	99.16	5912	166	50
	DT	98.61	98.30	98.93	2769	48	30
	VBD	89.59	90.07	89.11	1514	167	185
	IN	97.98	97.79	98.16	3944	89	74
	PRP	95.82	96.95	94.71	573	18	32
	NNS	88.24	86.82	89.70	1950	296	224
	VBP	78.40	83.33	74.02	245	49	86
	MD	96.98	98.09	95.90	257	5	11
	VBN	77.71	79.40	76.10	659	171	207
	POS	96.51	93.51	99.71	346	24	1
	JJR	60.22	59.57	60.87	56	38	36
	"	100.00	100.00	100.00	421	0	0
	RB	81.24	86.22	76.80	682	109	206
	,	99.97	99.94	100.00	1637	1	0
	CC	99.08	99.47	98.69	755	4	10
	WDT	90.14	91.43	88.89	96	9	12
	(	99.85	100.00	99.71	686	0	2
	)	100.00	100.00	100.00	686	0	0
	:	99.33	99.83	98.83	592	1	7
	PRP$	98.66	97.68	99.66	295	7	1
	RBR	26.47	36.00	20.93	9	16	34
	VBG	76.29	76.13	76.45	370	116	114
	EX	82.19	76.92	88.24	30	9	4
	WP	96.58	94.17	99.12	113	7	1
	WRB	96.55	98.59	94.59	70	1	4
	-X-	99.78	99.57	100.00	232	1	0
	$	95.74	95.74	95.74	90	4	4
	RP	64.65	69.57	60.38	64	28	42
	NNPS	42.75	54.90	35.00	56	46	104
	SYM	95.08	91.34	99.15	116	11	1
	JJS	51.79	51.79	51.79	29	27	27

	I-NP	96.79	95.69	97.90	27703	1247	593
	I-VP	90.19	91.53	88.88	5013	464	627
	O	95.15	93.45	96.92	6019	422	191
	I-PP	95.80	94.86	96.76	3822	207	128
	B-NP	73.81	84.40	65.59	568	105	298
	I-SBAR	77.68	82.95	73.04	214	44	79
	I-ADJP	49.21	58.26	42.60	141	101	190
	I-ADVP	66.67	74.53	60.31	354	121	233
	I-PRT	69.39	79.07	61.82	68	18	42
	B-PP	65.00	72.22	59.09	26	10	18
