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
#define NET_HIDDEN_COUNT 128

#define SEQLEN            24
#define MAX_EPOCHS        32

#define ERRORLEVEL      0.20f

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

int main()
{
 bann   N; 
 char*  file;
 int    vocab_size;
 size_t i,size;

 if(createnetfromtextfile(&N,"bin/data/text/skip_sherlock.txt",&file,&size,&vocab_size))
  {
   int      e,I[2],O[2],good=0,bad=0,sel,turn=0;   
   real     smoothloss;
   size_t   seqlen=SEQLEN,s;
   bann_idx idxI,idxO;
   bann_io  trainI,trainO; 
   bannrnn  hlp;
   trainI.kind=idx;trainI.idx=&idxI;trainI.idx->hotpos=&I[0];
   trainO.kind=idx;trainO.idx=&idxO;trainO.idx->hotpos=&O[0];
   printf("training...\n");
   // timed output values
   bannrnn_new(&hlp,&N,seqlen);

   smoothloss=-logf(1.0f/vocab_size)*seqlen;	// inital loss

   for(e=0;(e<MAX_EPOCHS)&&(smoothloss>ERRORLEVEL);e++)
    {                    
     i=0;
     while(i<size-seqlen)
      {
       real loss=0,err=0;

       // forward pass using rnn
       bannrnn_start(&hlp,&N,(i==0));
       for(s=0;s<seqlen;s++)
        {
         real*output;
         int  outid=ch2id[file[i+s+1]];
         trainI.idx->hotcnt=1;trainI.idx->hotpos[0]=ch2id[file[i+s]]; 
         trainO.idx->hotcnt=1;trainO.idx->hotpos[0]=outid;
         // save previous state
         bannrnn_hprev(&hlp,&N,s,bannrnn_push);
         // forward
         output=bann_predict(&N,&trainI);
         // save output state
         bannrnn_hidout(&hlp,&N,s,bannrnn_push);
         if((N.weights[1].flags&7)==bann_activation_softmax)
          loss+=-logf(output[outid]);
         else
          loss+=0.5f*(1-output[outid])*(1-output[outid]);
        }
       // save last hidden output
       bannrnn_hprev(&hlp,&N,s,bannrnn_push);    

       smoothloss = smoothloss * 0.999f + loss * 0.001f;

       // backward pass - back propagation thru time 
       while(s--)
        {
         trainI.idx->hotcnt=1;trainI.idx->hotpos[0]=ch2id[file[i+s]];         
         trainO.idx->hotcnt=1;trainO.idx->hotpos[0]=ch2id[file[i+s+1]];
         err+=bannrnn_backward(&hlp,&N,s,&trainI,&trainO);
        }
       bannrnn_updatenntraining(&hlp,&N,err,seqlen);
       
       // restore last hidden output
       bannrnn_hprev(&hlp,&N,s,bannrnn_pull);              
       i+=seqlen;

       turn++;

       if((turn&0x7F)==0x7F)
        {
         printf("epochs: %d - pos: %d/%d - loss: %.2f \r",e,i,size,smoothloss);
         printf("\ngenerating: ");
          
         bannrnn_hprev(&hlp,&N,s,bannrnn_push);       
         generatetext(&N,64,-1);
         bannrnn_hprev(&hlp,&N,s,bannrnn_pull);
        }
        
      }
     printf("\ngenerating: ");
     generatetext(&N,64,e+1);
    }

   bannrnn_delete(&hlp);

   printf("check on input text...\n");
   bannrnn_start(NULL,&N,1);
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