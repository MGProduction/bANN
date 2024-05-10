#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <assert.h>

#define BANN_IMPLEMENTATION
#include "bann.h"

#define TRAINING_IMAGES "bin/data/fashion/train-images-idx3-ubyte"
#define TRAINING_LABELS "bin/data/fashion/train-labels-idx1-ubyte"

#define TESTING_IMAGES "bin/data/fashion/t10k-images-idx3-ubyte"
#define TESTING_LABELS "bin/data/fashion/t10k-labels-idx1-ubyte"

#define BANN_HIDDENSIZE     128
#define BANN_HIDDENNUMBER    1

int bann_gethighestoutput(bann*net)
{
 int  i,best=-1,hm=net->weights[net->weightscount-1].outs;
 real top,*v=net->weights[net->weightscount-1].o;
 for(i=0;i<hm;i++)
  if((best==-1)||(v[i]>top))
  {
   best=i;
   top=net->weights[net->weightscount-1].o[i];
  }
 return best;
}

void shuffle(int*idx,int size)
{
 int i;
 for(i=0;i<size;i++)
  {
   int a=rand()%size;
   int b=rand()%size;
   if(a!=b)
   {
    int t=idx[a];
    idx[a]=idx[b];
    idx[b]=t;
   }
  }
}

int read_imagesets(const char*f1,const char*f2,int*pmnist_size,int*pmnist_sample,unsigned char**ptrainingset,unsigned char**ptrainingoutput)
{
 int           ret=0,mnist_size,mnist_sample;
 unsigned char*trainingset;
 unsigned char*trainingoutput;
 FILE         *f=fopen(f1,"rb"); //
 if(f)
  {  
   unsigned char header[16];  
   size_t        read;
   fread(header,sizeof(header),1,f);
   mnist_sample=header[6]*256+header[7];
   mnist_size=(header[10]*256+header[11])*(header[14]*256+header[15]);   
   trainingset=(unsigned char*)malloc(mnist_size*mnist_sample);
   read=fread(trainingset,1,mnist_size*mnist_sample,f);
   if(ptrainingset)
    *ptrainingset=trainingset;
   if(pmnist_size)
    *pmnist_size=mnist_size;
   if(pmnist_sample)
    *pmnist_sample=mnist_sample;
   fclose(f);
   ret|=1;
  }
 f=fopen(f2,"rb"); // 
 if(f)
  {  
   unsigned char header[8];  
   size_t        read;
   fread(header,sizeof(header),1,f);
   trainingoutput=(unsigned char*)malloc(mnist_sample);
   read=fread(trainingoutput,1,mnist_sample,f);
   if(ptrainingoutput)
    *ptrainingoutput=trainingoutput;
   fclose(f);
   ret|=2;
  }
 return ret;
}

#define mnist_outputsize    10

int main()
{
 bann          N; 
 int           epoch,max_epoch=8,sample,mnist_size,mnist_sample;
 int          *idx;
 unsigned char*trainingset;
 unsigned char*trainingoutput;
 real         *input;
 real         *output;
 
 if(read_imagesets(TRAINING_IMAGES,
                   TRAINING_LABELS,
                   &mnist_size,&mnist_sample,
                   &trainingset,&trainingoutput)==3)
  {

   bann_new(&N,mnist_size,BANN_HIDDENSIZE,BANN_HIDDENNUMBER,mnist_outputsize);
   bann_setlearningrate(&N,1.0f,bann_autooptimize_on);


   idx=(int*)malloc(mnist_sample*sizeof(int));
   for(sample=0;sample<mnist_sample;sample++)
    idx[sample]=sample;

   // training
   input=(real*)malloc(sizeof(input[0])*mnist_size);
   output=(real*)malloc(sizeof(output[0])*mnist_outputsize);

   for(epoch=0;epoch<max_epoch;epoch++)
    {
     int good=0,bad=0;

     shuffle(idx,mnist_sample);

     for(sample=0;sample<mnist_sample;sample++)
      {
       int  sel,id=idx[sample],n;

       // set input
       for(n=0;n<mnist_size;n++)
        {
         int gray=trainingset[id*mnist_size+n];
         if(gray)
          input[n]=gray/255.0f;
         else
          input[n]=0;
        }

       bann_predict(&N,input);
       sel=bann_gethighestoutput(&N);

       // check results
       if(sel==trainingoutput[id])
        good++;
       else
        bad++;

       memset(output,0,sizeof(output[0])*mnist_outputsize);
       output[trainingoutput[id]]=1.0f;
       bann_train(&N,input,output);
       
       if((sample&0xFF)==0xFF)
        printf("epoch: %d samples: %d accuracy: %.2f%%       \r",epoch,sample,(real)good*100.0f/(real)(good+bad));
      }
     printf("epoch: %d samples: %d accuracy: %.2f%%       \n",epoch,sample,(real)good*100.0f/(real)(good+bad));
    }
   free(output);
   free(input);
   
   // save
   bann_write(&N,"bin/fashion-mnist.bann",0);
   bann_delete(&N); 

   // destroy training set
   free(trainingset);
   free(trainingoutput);

   // reload
   if(bann_read(&N,"bin/fashion-mnist.bann"))
    {
     int good=0,bad=0;

     read_imagesets(TESTING_IMAGES,
                    TESTING_LABELS,
                    &mnist_size,&mnist_sample,
                    &trainingset,&trainingoutput);

     // test
     input=(real*)malloc(sizeof(input[0])*mnist_size);
     for(sample=0;sample<mnist_sample;sample++)
      {
       int  n,sel,id=sample;
       real bestval=-100;

       // set input
       for(n=0;n<mnist_size;n++)
        {
         int gray=trainingset[id*mnist_size+n];
         if(gray)
          input[n]=gray/255.0f;
         else
          input[n]=0;
        }

       bann_predict(&N,input);     
       sel=bann_gethighestoutput(&N);     
       // check results
       if(sel==trainingoutput[id])
        good++;
       else
        bad++;

       printf("samples: %d accuracy: %.2f%%       \r",sample,(real)good*100.0f/(real)(good+bad));
      }
     free(input);

     printf("samples: %d accuracy: %.2f%%       \n",sample,(real)good*100.0f/(real)(good+bad));
     bann_delete(&N); 
     free(trainingset);
     free(trainingoutput);
    }
   else
    printf("can't find testing set (check README.md in bin/data/mnist folder)\n");
  }
 else
  printf("can't find training set (check README.md in bin/data/mnist folder)\n");
 return 1;
}