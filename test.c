#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <assert.h>

#define BANN_IMPLEMENTATION
#define BANN_DEBUG
#include "bann.h"

#define learningrate      0.75f

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

int xor()
{
 bann N; 
 int      xor_maxepoch=8192;
 int      e,v,cnt=0;
 real     err=0;
 const real input[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
 const real output[4] = {0, 1, 1, 0};

 bann_new(&N,2,2,1,1);
 bann_setlearningrate(&N,1.0f,bann_autooptimize_on);

 for(e=0;e<xor_maxepoch;e++)
  for(v=0;v<4;v++)
   {
    real dif;

    bann_train(&N,&input[v][0],&output[v]);
    
    dif=0.5f*(N.weights[1].o[0]-output[v])*(N.weights[1].o[0]-output[v]);
    err+=dif; 
    cnt++;
    
    printf("e:%d v:%d err: %.2f \r",e,v,err/cnt);
   }
 printf("\n");

 for(v=0;v<4;v++)
  {
   bann_predict(&N,input[v]);   
   printf("value %d: %.2f / %.2f\n",v+1,N.weights[1].o[0],output[v]);
  }

 bann_delete(&N); 

 return 1;
}

#define mnist_outputsize    10
#define imnist_size       28*28	
#define imnist_sample     60000
#define mnist_max_epoch       8

#define STB_IMAGE_WRITE_IMPLEMENTATION
//include "libs/stb_image_write.h"

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

int mnist()
{
 bann          N; 
 int           epoch,sample,mnist_size,mnist_sample;
 int          *idx;
 unsigned char*trainingset;
 unsigned char*trainingoutput;
 
 read_imagesets("bin/data/mnist/train-images.idx3-ubyte",
                "bin/data/mnist/train-labels.idx1-ubyte",
                &mnist_size,&mnist_sample,
                &trainingset,&trainingoutput);

 idx=(int*)malloc(mnist_sample*sizeof(int));
 for(sample=0;sample<mnist_sample;sample++)
  idx[sample]=sample;

 bann_new(&N,mnist_size,64,1,10);
 bann_setlearningrate(&N,1.0f,bann_autooptimize_on);

 // training
 for(epoch=0;epoch<mnist_max_epoch;epoch++)
  {
   int good=0,bad=0;

   shuffle(idx,mnist_sample);

   for(sample=0;sample<mnist_sample;sample++)
    {
     int  sel,id=idx[sample],n;
     real input[imnist_size],output[mnist_outputsize],bestval=-100;

     // set input
     for(n=0;n<mnist_size;n++)
      {
       int gray=trainingset[id*mnist_size+n];
       if(gray)
        input[n]=gray/255.0f;
       else
        input[n]=0;
      }

     //stbi_write_png("bin/sampleX.png",28,28,1,trainingset+id*mnist_size,0);

     bann_predict(&N,input);
     sel=bann_gethighestoutput(&N);

     // check results
     if(sel==trainingoutput[id])
      good++;
     else
      bad++;

     memset(output,0,sizeof(output));
     output[trainingoutput[id]]=1.0f;
     bann_train(&N,input,output);
     
     if((sample&0xFF)==0xFF)
      printf("epoch: %d samples: %d accuracy: %.2f%%       \r",epoch,sample,(real)good*100.0f/(real)(good+bad));
    }
   printf("epoch: %d samples: %d accuracy: %.2f%%       \n",epoch,sample,(real)good*100.0f/(real)(good+bad));
  }
 // save
 bann_write(&N,"bin/mnist.bann",0);
 bann_delete(&N); 
 // destroy training set
 free(trainingset);
 free(trainingoutput);
 // reload
 if(bann_read(&N,"bin/mnist.bann"))
  {
   int good=0,bad=0;

   read_imagesets("bin/data/mnist/t10k-images.idx3-ubyte",
                "bin/data/mnist/t10k-labels.idx1-ubyte",
                &mnist_size,&mnist_sample,
                &trainingset,&trainingoutput);

   // test
   for(sample=0;sample<mnist_sample;sample++)
    {
     int  n,sel,id=sample;
     real input[imnist_size],bestval=-100;

     // set input
     for(n=0;n<mnist_size;n++)
      {
       int gray=trainingset[id*mnist_size+n];
       if(gray)
        input[n]=gray/255.0f;
       else
        input[n]=0;
      }

     //stbi_write_png("bin/sample.png",28,28,1,trainingset+id*mnist_size,0);

     bann_predict(&N,input);     
     sel=bann_gethighestoutput(&N);     
     // check results
     if(sel==trainingoutput[id])
      good++;
     else
      bad++;

     printf("samples: %d accuracy: %.2f%%       \r",sample,(real)good*100.0f/(real)(good+bad));
    }
   printf("samples: %d accuracy: %.2f%%       \n",sample,(real)good*100.0f/(real)(good+bad));
   bann_delete(&N); 
   free(trainingset);
   free(trainingoutput);
  }

 return 1;
 }

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
 }

int main(int argc,const char*argv[])
{
 //mnistrev();
 mnist();

 xor();
 
 
 return 0;
}