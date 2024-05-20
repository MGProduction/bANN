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

This is NOT a standard test for ann but a toy project to see how **bann_io** works (and how anns can be used to get THREE - or more - return values).  This example uses **[ConLLU 2003 English dataset](https://www.clips.uantwerpen.be/conll2003/ner/)** to create a minimal **POS/NER tagger**. Code shows how a moving window of context is passed as input, and how three set of data (POS, CHUNK and NER) are retrieved as output.

Input is **100k** cell wide BUT, using bann_io structure, training (and usage) keeps the average speed not too slow even on a small PC. And saving ann in a quantized way makes also the memory usage relatively small.

Results - with such a simple implementation - are far from astonishing (**84.6%** of accuracy considering all the three fields together - **91.9%** POS, **94.1%** CHUNK and **77.1%** NER as **F1** considering each field in a separate way) but here we don't use ANY outside information - no embeddings or any statistical data or dictionary outside what we took from train dataset.

These are quality values for test dataset (after just **16** epochs):

	NER	F1	PREC	RECALL	CORRECT	MISS	ADD
	I-ORG	73.94	73.50	74.39	1853	668	638
	I-MISC	66.41	59.63	74.92	681	461	228
	I-PER	84.60	85.74	83.48	2315	385	458
	I-LOC	76.97	77.77	76.19	1462	418	457	

These are quality values for **dev** dataset:

	NER	F1	PREC	RECALL	CORRECT	MISS	ADD
	I-ORG	75.58	77.18	74.04	1549	458	543
	I-MISC	75.06	73.52	76.66	969	349	295
	I-PER	88.19	89.99	86.47	2723	303	426
	I-LOC	81.70	82.83	80.61	1688	350	406	

Out of curiosity, there are values for test dataset for the other two fields (**POS** and **CHUNK/TYPE**)

	NNP	92.70	91.46	93.98	8043	751	515
	VBZ	81.93	88.44	76.32	390	51	121
	JJ	77.72	77.61	77.82	2368	683	675
	NN	87.08	86.52	87.65	5282	823	744
	TO	99.61	99.89	99.34	900	1	6
	VB	86.57	90.86	82.66	925	93	194
	.	99.89	99.95	99.84	1876	1	3
	CD	96.17	93.92	98.53	4233	274	63
	DT	98.90	98.64	99.15	3491	48	30
	VBD	90.24	90.49	90.00	2007	211	223
	IN	97.99	97.99	97.99	4877	100	100
	PRP	96.41	97.74	95.13	820	19	42
	NNS	89.02	89.14	88.89	2225	271	278
	VBP	79.17	80.28	78.08	285	70	80
	MD	97.98	98.98	97.00	291	3	9
	VBN	77.45	79.24	75.73	752	197	241
	POS	98.14	96.57	99.76	422	15	1
	JJR	62.80	63.73	61.90	65	37	40
	"	99.92	99.84	100.00	640	1	0
	RB	79.51	79.76	79.27	784	199	205
	,	100.00	100.00	100.00	1949	0	0
	CC	99.08	99.89	98.28	916	1	16
	WDT	89.20	96.97	82.58	128	4	27
	(	100.00	100.00	100.00	679	0	0
	)	99.78	99.56	100.00	680	3	0
	:	99.92	99.84	100.00	624	1	0
	PRP$	97.33	95.24	99.53	420	21	2
	VBG	77.81	78.49	77.14	540	148	160
	WP	95.69	95.31	96.06	122	6	5
	WRB	96.67	100.00	93.55	87	0	6
	-X-	99.54	99.08	100.00	216	2	0
	$	98.54	97.12	100.00	101	3	0
	RP	70.78	68.99	72.67	109	49	41
	NNPS	47.21	51.80	43.37	72	67	94
	SYM	84.52	86.59	82.56	71	11	15
	JJS	19.61	41.67	12.82	10	14	68

	I-NP	96.81	96.03	97.59	29303	1210	724
	I-VP	91.28	91.92	90.64	6190	544	639
	O	97.13	96.45	97.83	7032	259	156
	I-PP	95.75	95.47	96.04	4656	221	192
	B-NP	80.59	89.96	72.99	708	79	262
	I-SBAR	78.08	83.54	73.30	269	53	98
	I-ADJP	48.28	58.57	41.06	147	104	211
	I-ADVP	66.35	71.38	61.99	424	170	260
	I-PRT	65.62	61.90	69.80	104	64	45
	B-PP	48.94	56.10	43.40	23	18	30
