#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <assert.h>

#define BANN_IMPLEMENTATION
#define BANN_RNN
#define BANN_DEBUG
#include "bann.h"

#define MAX_CHARS        256
#define NET_HIDDEN_COUNT   4

#define SEQLEN             2
#define MAX_EPOCHS        32

int ch2id[MAX_CHARS];
int id2ch[MAX_CHARS];

int char_get(const char*sz,int*step)
{
 if(step) *step=1;
 return (int)(unsigned char*)sz[0];
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

int createnetfromtextfile(bann*N,const char*file,char**text,size_t*psize,int*pvocabsize)
{
 FILE*f=fopen(file,"rb");
 if(f)
 {
  byte   used[MAX_CHARS]; // one byte for char - change that AND char_get to use utf8
  size_t size,i,needed,pos;
  fseek(f,0,2);
  size=ftell(f);
  (*text)=(char*)malloc(size);
  (*psize)=size;
  fseek(f,0,0);
  fread((*text),1,size,f);
  fclose(f);

  memset(used,0,sizeof(used));

  needed=i=0;
  while(i<size)
  {
   int a,c=char_get((*text)+i,&a);
   if(used[c]==0)
    {
     used[c]=1;
     needed++;
    }
   i+=a;
  }  
  if(pvocabsize) *pvocabsize=(int)needed;

  for(pos=i=0;i<MAX_CHARS;i++)
   if(used[i])
    {
     id2ch[pos]=(int)i;
     ch2id[i]=(int)pos++;
    }  

  bann_new(N,needed,NET_HIDDEN_COUNT,1,needed);
  bann_setlearningrate(N,0.25f,bann_optimizer_off,256);
  bann_setweightactivation(N,0,bann_activation_tanh);
  bann_setweightactivation(N,1,bann_activation_softmax);
  return 1;
 }
 else
  return 0;
}

int getnextch(bann*N)
{
 int op,ix,outputcnt=N->weights[N->weightscount-1].outs;
 real*output=N->weights[N->weightscount-1].o;
 real cdf[MAX_CHARS]; 
 real sum=0.0f,uniform_samples;
 /*real softmax[MAX_CHARS];
 for(op=0;op<outputcnt;op++)
 {
  softmax[op]=expf(output[op]);
  sum+=softmax[op];
 }
 for(op=0;op<outputcnt;op++)
  softmax[op]=softmax[op]/sum;
 */
 sum=0;
 for(op=0;op<outputcnt;op++)
  {
	  sum+=output[op];//softmax[op];
	  cdf[op]=sum;
	 }
	for(op=0;op<outputcnt;op++)
		cdf[op]/=cdf[outputcnt-1];
	uniform_samples=(real)rand()/(real)RAND_MAX;
	//searchsorted
	ix=0;
	for(op=outputcnt-1;op>0;op--)
  {
		 if(uniform_samples>=cdf[op-1] && uniform_samples<=cdf[op])
    {
			  ix=op;
			  break;
		  }
		 }
 return ix;
}

void generatetext(bann*N,int hm,int e)
{
 char    output[256];
 int     i,sel,I[2];
 bann_idx idxI;
 bann_io trainI;
 
 trainI.kind=idx;trainI.idx=&idxI;trainI.idx->hotpos=&I[0];
 
 sel=getnextch(N);
 trainI.idx->hotcnt=1;trainI.idx->hotpos[0]=sel;
 i=0;
 output[i++]=id2ch[sel];
 while(i<hm)
  {
   // ---bann_setweightactivation(N,1,bann_activation_softmax);
   bann_predict(N,&trainI);
   // ---bann_setweightactivation(N,1,bann_activation_sigmoid);

   sel=getnextch(N);
   trainI.idx->hotpos[0]=sel;
   output[i++]=id2ch[sel];
  }
 output[i]=0;
 if(e==-1)
  printf("%s\n",output);
 else
  printf("%d: %s\n\n",e,output);
}

real calcloss(real*output,bann_io*out,int hm)
{
 real loss=0;
 int  i,c=0;
 for(i=0;i<hm;i++)
 {
  real val=0;
  if((c<out->idx->hotcnt)&&(out->idx->hotpos[c]==i))
   {val=1;c++;}
  loss+=-logf(output[i]);
 }
 return loss;
}

int main()
{
 bann   N; 
 char*  file;
 int    vocab_size;
 size_t i,size;

 if(createnetfromtextfile(&N,"bin/data/text/skip_sherlock.txt",&file,&size,&vocab_size))
  {
   int     e,I[2],O[2],good=0,bad=0,sel;   
   size_t  seqlen=SEQLEN,s;
   real*   outseq,*hprev,*hidseq;
   bann_idx idxI,idxO;
   bann_io trainI,trainO; 
   trainI.kind=idx;trainI.idx=&idxI;trainI.idx->hotpos=&I[0];
   trainO.kind=idx;trainO.idx=&idxO;trainO.idx->hotpos=&O[0];
   printf("training...\n");
   // timed output values
   outseq=(real*)malloc(N.weights[N.weightscount-1].outs*seqlen*sizeof(outseq[0]));
   hidseq=(real*)malloc(N.weights[0].outs*seqlen*sizeof(outseq[0]));
   hprev=(real*)malloc(N.rweights[0].ins*seqlen*sizeof(outseq[0]));

   bann_allocrnntraining(&N);
   
   for(e=0;e<MAX_EPOCHS;e++)
    {                    
     for(i=0;i<size-seqlen;i++)
      {
       real loss=0,err=0;
       // forward pass;
       bann_startrnn(&N);
       for(s=0;s<seqlen;s++)
        {
         trainI.idx->hotcnt=1;trainI.idx->hotpos[0]=ch2id[file[i+s]]; 
         trainO.idx->hotcnt=1;trainO.idx->hotpos[0]=ch2id[file[i+s+1]];
         // save previous state
         memcpy(hprev+s*N.rweights[0].ins,N.rweights[0].o,N.rweights[0].ins*sizeof(hprev[0]));
         // forward
         bann_predict(&N,&trainI);
         // save output state
         memcpy(hidseq+s*N.weights[0].outs,N.weights[0].o,N.weights[0].outs*sizeof(outseq[0]));         
         memcpy(outseq+s*N.weights[1].outs,N.weights[1].o,N.weights[1].outs*sizeof(hidseq[0]));                  

         if((N.weights[1].flags&7)==bann_activation_softmax)
          loss+=-logf(outseq[s*seqlen+trainO.idx->hotpos[0]]);
         else
          loss+=0.5f*(1-outseq[s*seqlen+trainO.idx->hotpos[0]])*(1-outseq[s*seqlen+trainO.idx->hotpos[0]]);
        }
       // backward pass
       while(s--)
        {
         trainI.idx->hotcnt=1;trainI.idx->hotpos[0]=ch2id[file[i+s]];         
         trainO.idx->hotcnt=1;trainO.idx->hotpos[0]=ch2id[file[i+s+1]];
         // restore output and hidden
         memcpy(N.weights[0].o,hidseq+s*N.weights[0].outs,N.weights[0].outs*sizeof(hidseq[0]));         
         memcpy(N.weights[1].o,outseq+s*N.weights[1].outs,N.weights[1].outs*sizeof(outseq[0]));                  
         // restore previous
         memcpy(N.rweights[0].d,hprev+s*N.rweights[0].ins,N.rweights[0].ins*sizeof(hprev[0]));
         // back 
         err+=bann_backwardrnn(&N,&trainI,&trainO);
        }
       bann_updatenntraining(&N,err,seqlen);
       if(((i&0xFF)==0xFF)||(i+1==size-seqlen))
        printf("epochs: %d - pos: %d/%d - loss: %.2f \r",e,i,size,loss);
       if((i&0x3FF)==0x3FF)
        if(1)
         {
          printf("\ngenerating: ");
          generatetext(&N,64,e+1);
         }
        else
         {
          size_t i;
          printf("\ncheck on input text...\n");
          bann_startrnn(&N);
          for(i=0;i<size-1;i++)
           {     
            trainI.idx->hotcnt=1;trainI.idx->hotpos[0]=ch2id[file[i]];
            bann_predict(&N,&trainI);
            sel=bann_gethighestoutput(&N);
            if(id2ch[sel]==file[i+1])
             good++;
            else
             bad++;
            printf("epochs: %d - pos: %d/%d - accuracy: %.2f \r",e,i,size,(float)good*100.0f/(float)(bad+good));
           }
          printf("\n");
         }
      }
     printf("\ngenerating: ");
     generatetext(&N,64,e+1);
    }
   free(outseq);
   free(hidseq);
   free(hprev);
   bann_freernntraining(&N);
   printf("check on input text...\n");
   bann_startrnn(&N);
   for(i=0;i<size-1;i++)
    {     
     trainI.idx->hotcnt=1;trainI.idx->hotpos[0]=ch2id[file[i]];
     bann_predict(&N,&trainI);
     sel=bann_gethighestoutput(&N);
     if(id2ch[sel]==file[i+1])
      good++;
     else
      bad++;
    }
   printf("generating text...\n");
   generatetext(&N,64,-1);
   i=0;
   bann_delete(&N);
   free(file);
  }
 return 1;
}