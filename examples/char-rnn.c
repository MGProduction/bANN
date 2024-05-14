#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <assert.h>

#define BANN_IMPLEMENTATION
#define BANN_RNN
#include "bann.h"

#define MAX_CHARS        256
#define NET_HIDDEN_COUNT  64

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
 for(op=0;op<outputcnt;op++)
  {
	  sum+=output[op];
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
 bannidx trainI;
 trainI.m=2;trainI.b=&I[0];trainI.c=1; 
 sel=getnextch(N);
 trainI.b[0]=sel;
 i=0;
 output[i++]=id2ch[sel];
 while(i<64)
  {
   bann_predictidx(N,&trainI);
   sel=getnextch(N);
   trainI.b[0]=sel;
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

 if(createnetfromtextfile(&N,"bin/data/text/test.txt",&file,&size,&vocab_size))
  {
   int     e,I[2],O[2],good=0,bad=0,sel;   
   size_t  seqlen=20,s;
   bannidx trainI,trainO;
   trainI.m=trainO.m=2;trainI.b=&I[0];trainO.b=&O[0];
   printf("training...\n");
   for(e=0;e<32;e++)
    {
     bann_startrnn(&N);
     for(i=0;i<size-seqlen;i++)
      {
       for(s=0;s<seqlen;s++)
        {
         trainI.c=1;trainI.b[0]=ch2id[file[i+s]];
         trainO.c=1;trainO.b[0]=ch2id[file[i+s+1]];
         bann_trainidx(&N,&trainI,&trainO);
        }
      }
     generatetext(&N,64,e+1);
    }
   printf("check on input text...\n");
   for(i=0;i<size-1;i++)
    {     
     trainI.c=1;trainI.b[0]=ch2id[file[i]];
     bann_predictidx(&N,&trainI);
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