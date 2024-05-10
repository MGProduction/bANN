#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <assert.h>

#define BANN_IMPLEMENTATION
#include "bann.h"

#define learningrate      0.75f
/*
int mnistrev()
{
 bann          N; 
 int           epoch,sample,mnist_size,mnist_sample;
 int          *idx;
 unsigned char*trainingset;
 unsigned char*trainingoutput;
 
 read_imagesets("bin/data/train-images.idx3-ubyte",
                "bin/data/train-labels.idx1-ubyte",
                &mnist_size,&mnist_sample,
                &trainingset,&trainingoutput);

 idx=(int*)malloc(mnist_sample*sizeof(int));
 for(sample=0;sample<mnist_sample;sample++)
  idx[sample]=sample;

 bann_new(&N,10,64,4,mnist_size);
 bann_setlearningrate(&N,1.0f,bann_autooptimize_on);

 // training
 for(epoch=0;epoch<256;epoch++)
  {
   int good=0,bad=0;

   shuffle(idx,mnist_sample);

   for(sample=0;sample<mnist_sample;sample++)
    {
     int  id=idx[sample],n;
     real output[imnist_size],input[mnist_outputsize],bestval=-100;

     // set input
     for(n=0;n<mnist_size;n++)
      {
       int gray=trainingset[id*mnist_size+n];
       if(gray)
        output[n]=gray/255.0f;
       else
        output[n]=0;
      }
     memset(input,0,sizeof(input));
     input[trainingoutput[id]]=1.0f;
     
     bann_train(&N,input,output);
    
     if((sample&0xFF)==0xFF)
      printf("epoch: %d samples: %d\r",epoch,sample);
     if(sample+1==mnist_sample)
      {
       char nm[256];
       sprintf(nm,"bin/sampleX.%03d.png",epoch+1);
       for(n=0;n<mnist_size;n++)
        output[n]=N.weights[N.weightscount-1].o[n]*255;
       //stbi_write_png(nm,28,28,1,output,0);    
      }
    }
   printf("epoch: %d samples: %d\n",epoch,sample);
  }
 // save
 bann_write(&N,"bin/mnistrev.bann",0);
 bann_delete(&N); 
 // destroy training set
 free(trainingset);
 free(trainingoutput);

 return 1;
}*/

int main(int argc,const char*argv[])
{

 
 return 0;
}