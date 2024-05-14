#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <assert.h>

#define BANN_IMPLEMENTATION
#include "bann.h"

// 
#define FASHION_TEST
// 

#if defined(FASHION_TEST)

#define TEST_NAME       "fashion"

#define TRAINING_IMAGES "bin/data/fashion/train-images-idx3-ubyte"
#define TRAINING_LABELS "bin/data/fashion/train-labels-idx1-ubyte"

#define TESTING_IMAGES "bin/data/fashion/t10k-images-idx3-ubyte"
#define TESTING_LABELS "bin/data/fashion/t10k-labels-idx1-ubyte"

#define BANN_HIDDENSIZE        96
#define BANN_HIDDENNUMBER       1

#define MAX_EPOCHS             10

#else

#define TEST_NAME       "mnist"

#define TRAINING_IMAGES "bin/data/mnist/train-images.idx3-ubyte"
#define TRAINING_LABELS "bin/data/mnist/train-labels.idx1-ubyte"

#define TESTING_IMAGES "bin/data/mnist/t10k-images.idx3-ubyte"
#define TESTING_LABELS "bin/data/mnist/t10k-labels.idx1-ubyte"

#define BANN_HIDDENSIZE        64
#define BANN_HIDDENNUMBER       1

#define MAX_EPOCHS              8

#endif

#define DESIRED_MSE         0.005f

void history_add(bann*N,const char*file,const char*name,int e,real quality)
{
 FILE*f=fopen(file,"ab");
 if(f)
 {
  fprintf(f,"[%s]: %d x %d x %d (epochs: %d): %.2f%%\r\n",name,N->weights[0].ins,N->weights[0].outs,N->weights[N->weightscount-1].outs,e,(float)quality);
  fclose(f);
 } 
}

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
 int           epoch,sample,mnist_size,mnist_sample;
 int          *index;
 unsigned char*trainingset;
 unsigned char*trainingoutput;
 
 if(read_imagesets(TRAINING_IMAGES,
                   TRAINING_LABELS,
                   &mnist_size,&mnist_sample,
                   &trainingset,&trainingoutput)==3)
  {
   int      tmp[2]={-1,-1};
   char     nm[256];
   bann_idx bo;
   bann_io  in,out;

   bann_new(&N,mnist_size,BANN_HIDDENSIZE,BANN_HIDDENNUMBER,mnist_outputsize);      
   //bann_setlearningrate(&N,0.9f,bann_autooptimize_progressive);

   index=(int*)malloc(mnist_sample*sizeof(int));
   for(sample=0;sample<mnist_sample;sample++)
    index[sample]=sample;

   // training
   in.kind=standard;
   in.linear=(real*)malloc(sizeof(in.linear[0])*mnist_size);

   out.kind=idx;

   bo.hotpos=&tmp[0];
   bo.hotcnt=1;
   out.idx=&bo;

   printf("[%s] - %d x %d x %d\n",TEST_NAME,N.weights[0].ins,N.weights[0].outs,N.weights[N.weightscount-1].outs);

   for(epoch=0;epoch<MAX_EPOCHS;epoch++)
    {
     int  good=0,bad=0;
     real glerr=0;

     shuffle(index,mnist_sample);

     for(sample=0;sample<mnist_sample;sample++)
      {
       int  sel,id=index[sample],n;
       real err;

       // set input
       for(n=0;n<mnist_size;n++)
        {
         int gray=trainingset[id*mnist_size+n];
         if(gray)
          in.linear[n]=gray/255.0f;
         else
          in.linear[n]=0;
        }

       out.idx->hotpos[0]=trainingoutput[id];
       err=bann_train(&N,&in,&out);

       sel=bann_gethighestoutput(&N);

       // check results
       if(sel==trainingoutput[id])
        good++;
       else
        bad++;

       glerr+=err;
       
       if((sample&0xFF)==0xFF)
        printf("epoch: %d samples: %d accuracy: %.2f%% mse: %.2f rate: %.2f       \r",epoch,sample,(real)good*100.0f/(real)(good+bad),glerr/sample,N.opt.learningrate);
      }     
     printf("epoch: %d samples: %d accuracy: %.2f%% mse: %.2f rate: %.2f       \n",epoch,sample,(real)good*100.0f/(real)(good+bad),glerr/sample,N.opt.learningrate);

     if(glerr/sample<=DESIRED_MSE)
      break;
    }
   free(in.linear);
   free(index);
   
   // save
   sprintf(nm,"bin/%s.bannQ",TEST_NAME);
   bann_write(&N,nm,bann_8bit);

   sprintf(nm,"bin/%s.bann",TEST_NAME);
   bann_write(&N,nm,0);

   bann_delete(&N); 

   // destroy training set
   free(trainingset);
   free(trainingoutput);

   // reload
   if(bann_read(&N,nm))
    {
     int        good=0,bad=0;

     read_imagesets(TESTING_IMAGES,
                    TESTING_LABELS,
                    &mnist_size,&mnist_sample,
                    &trainingset,&trainingoutput);

     // test
     in.kind=standard;
     in.linear=(real*)malloc(sizeof(in.linear[0])*mnist_size);

     for(sample=0;sample<mnist_sample;sample++)
      {
       int  n,sel,id=sample;
       real bestval=-100;

       // set input
       for(n=0;n<mnist_size;n++)
        {
         int gray=trainingset[id*mnist_size+n];
         if(gray)
          in.linear[n]=gray/255.0f;
         else
          in.linear[n]=0;
        }

       bann_predict(&N,&in);     
       sel=bann_gethighestoutput(&N);     
       // check results
       if(sel==trainingoutput[id])
        good++;
       else
        bad++;

       printf("samples: %d accuracy: %.2f%%       \r",sample,(real)good*100.0f/(real)(good+bad));
      }

     free(in.linear);

     printf("samples: %d accuracy: %.2f%%       \n",sample,(real)good*100.0f/(real)(good+bad));

     history_add(&N,"bin/bann.log",TEST_NAME,MAX_EPOCHS,(real)good*100.0f/(real)(good+bad));

     bann_delete(&N); 
     free(trainingset);
     free(trainingoutput);
    }
   else
    printf("can't find testing set (check README.md in bin/data/%s folder)\n",TEST_NAME);
  }
 else
  printf("can't find training set (check README.md in bin/data/%s folder)\n",TEST_NAME);
 return 1;
}