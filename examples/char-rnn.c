#include <errno.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define BANN_IMPLEMENTATION
#define BANN_RNN
#include "bann.h"

enum { BYTE_VALUES=256, HIDDEN=64, BLOCK=25, GENERATION_LIMIT=64 };

static unsigned char *read_text(const char *path,size_t *size)
{
 FILE *file=fopen(path,"rb");
 unsigned char *text=NULL;
 long length;
 if(!file)
  return NULL;
 if(fseek(file,0,SEEK_END)==0 && (length=ftell(file))>0 &&
    (uintmax_t)length<SIZE_MAX && fseek(file,0,SEEK_SET)==0)
  {
   text=(unsigned char*)malloc((size_t)length);
   if(text && fread(text,1,(size_t)length,file)==(size_t)length)
    *size=(size_t)length;
   else
    {
     free(text);
     text=NULL;
    }
  }
 fclose(file);
 return text;
}

static int most_likely(const real *scores,int count)
{
 int i,best=0;
 for(i=1;i<count;i++)
  if(scores[i]>scores[best])
   best=i;
 return best;
}

/* Lowering the temperature sharpens the distribution before sampling. */
static int sample(const real *scores,int count,real temperature)
{
 int i,best=most_likely(scores,count);
 real weights[BYTE_VALUES+1],sum=0,draw;
 for(i=0;i<count;i++)
  {
   weights[i]=powf(scores[i]/scores[best],1.0f/temperature);
   sum+=weights[i];
  }
 draw=(real)(rand()/(RAND_MAX+1.0))*sum;
 for(i=0;i<count;i++)
  {
   if(draw<weights[i])
    return i;
   draw-=weights[i];
  }
 return best;
}

static void generate(bann *net,const int *byte_to_id,const unsigned char *id_to_byte,
                     int boundary,const unsigned char *prompt,real temperature)
{
 const real *scores;
 int step,symbol=boundary;
 bann_idx indexed={&symbol,1};
 bann_io input;
 input.kind=iokind_idx;
 input.idx=&indexed;

 bannrnn_start(NULL,net,1);
 scores=bann_predict(net,&input);
 for(;*prompt;prompt++)
  {
   putchar(*prompt);
   symbol=byte_to_id[*prompt];
   scores=bann_predict(net,&input);
  }
 for(step=0;step<GENERATION_LIMIT;step++)
  {
   symbol=temperature>0 ? sample(scores,boundary+1,temperature) : most_likely(scores,boundary+1);
   if(symbol==boundary)
    break;
   putchar(id_to_byte[symbol]);
   scores=bann_predict(net,&input);
  }
 printf("\n  (%s)\n",step==GENERATION_LIMIT ? "generation limit reached" : "learned end of text");
}

int main(int argc,char **argv)
{
 const char *path=argc>1 ? argv[1] : "bin/data/text/test.txt";
 const unsigned char *prompt=(const unsigned char*)(argc>3 ? argv[3] : "");
 unsigned char *text,id_to_byte[BYTE_VALUES],seen[BYTE_VALUES]={0};
 int byte_to_id[BYTE_VALUES],boundary=0,epochs,epoch,i,symbol,label;
 size_t size,offset,transitions,correct=0;
 double evaluation_loss=0;
 bann net;
 bannrnn sequence;
 bann_idx input_index={&symbol,1},target_index={&label,1};
 bann_io input,target;

 if(argc>4)
  {
   fprintf(stderr,"Usage: %s [corpus [epochs [prompt]]]\n",argv[0]);
   return 1;
  }
 text=read_text(path,&size);
 if(!text)
  {
   fprintf(stderr,"Could not read a nonempty corpus from %s.\n",path);
   return 1;
  }
 epochs=size<BLOCK ? 128 : 32;
 if(argc>2)
  {
   char *end;
   long parsed;
   errno=0;
   parsed=strtol(argv[2],&end,10);
   if(errno || end==argv[2] || *end || parsed<=0 || parsed>INT_MAX)
    {
     fprintf(stderr,"Epochs must be a positive integer.\n");
     free(text);
     return 1;
    }
   epochs=(int)parsed;
  }

 for(offset=0;offset<size;offset++)
  seen[text[offset]]=1;
 for(i=0;i<BYTE_VALUES;i++)
  if(seen[i])
   {
    byte_to_id[i]=boundary;
    id_to_byte[boundary++]=(unsigned char)i;
   }
 for(const unsigned char *p=prompt;*p;p++)
  if(!seen[*p])
   {
    fprintf(stderr,"The prompt contains a byte absent from the corpus: 0x%02X.\n",(unsigned)*p);
    free(text);
    return 1;
   }

 input.kind=iokind_idx;
 input.idx=&input_index;
 target.kind=iokind_idx;
 target.idx=&target_index;
 srand(1);
 bann_new(&net,boundary+1,HIDDEN,1,boundary+1);
 bann_setlearningrate(&net,0.05f,bann_optimizer_off,0);
 bann_setweightactivation(&net,0,bann_activation_tanh);
 bann_setweightactivation(&net,1,bann_activation_softmax);
 bannrnn_new(&sequence,&net,BLOCK);
 if(!sequence.seqlen)
  {
   fprintf(stderr,"Could not initialize the recurrent training buffers.\n");
   bann_delete(&net);
   free(text);
   return 1;
  }

 /* The same boundary symbol opens the stream and marks its end. */
 transitions=size+1;
 printf("%zu bytes, %d distinct bytes, %d training epochs.\n",size,boundary,epochs);
 for(epoch=0;epoch<epochs;epoch++)
  {
   double loss=0;
   for(offset=0;offset<transitions;)
    {
     int step,length=transitions-offset<BLOCK ? (int)(transitions-offset) : BLOCK;
     real error=0;
     bannrnn_start(&sequence,&net,offset==0);
     bannrnn_hprev(&sequence,&net,-1,bannrnn_push);
     for(step=0;step<length;step++)
      {
       size_t position=offset+step;
       symbol=position ? byte_to_id[text[position-1]] : boundary;
       bann_predict(&net,&input);
       bannrnn_hidout(&sequence,&net,step,bannrnn_push);
      }
     /* Keep the final forward state for the next block. */
     bannrnn_hprev(&sequence,&net,BLOCK,bannrnn_push);
     for(step=length-1;step>=0;step--)
      {
       size_t position=offset+step;
       symbol=position ? byte_to_id[text[position-1]] : boundary;
       label=position<size ? byte_to_id[text[position]] : boundary;
       error+=bannrnn_backward(&sequence,&net,step,&input,&target);
      }
     bannrnn_updatenntraining(&sequence,&net,error,length);
     bannrnn_hprev(&sequence,&net,BLOCK,bannrnn_pull);
     loss+=error;
     offset+=length;
    }
   if(epoch==0 || (epoch+1)%8==0 || epoch+1==epochs)
    printf("Epoch %d/%d: mean training loss %.4f\n",epoch+1,epochs,loss/transitions);
  }
 bannrnn_delete(&sequence);

 bannrnn_start(NULL,&net,1);
 for(offset=0;offset<transitions;offset++)
  {
   const real *scores;
   symbol=offset ? byte_to_id[text[offset-1]] : boundary;
   label=offset<size ? byte_to_id[text[offset]] : boundary;
   scores=bann_predict(&net,&input);
   evaluation_loss-=logf(fmaxf(scores[label],FLT_MIN));
   correct+=most_likely(scores,boundary+1)==label;
  }
 printf("Next-symbol prediction on training text: %zu/%zu correct, mean loss %.4f.\n",
        correct,transitions,evaluation_loss/transitions);
 puts("\nGreedy generation:");
 generate(&net,byte_to_id,id_to_byte,boundary,prompt,0);
 puts("\nSampled generation (temperature 0.7):");
 generate(&net,byte_to_id,id_to_byte,boundary,prompt,0.7f);

 bann_delete(&net);
 free(text);
 return 0;
}
