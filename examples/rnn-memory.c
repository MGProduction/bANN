#include <stdio.h>
#include <stdlib.h>

#define BANN_IMPLEMENTATION
#define BANN_RNN
#include "bann.h"

enum { A, B, X, Y, QUERY, INPUTS };
enum { HIDDEN=16, TRAIN_DELAY=6, TRAIN_STEPS=4000, TEST_TRIALS=200 };

int main(void)
{
 static const int test_delays[]={1,3,6,8,12,20};
 bann net;
 bannrnn sequence;
 int tokens[TRAIN_DELAY+2],symbol,label,trial,step;
 real loss=0;
 bann_idx token={&symbol,1},target={&label,1};
 bann_io input,output;

 input.kind=iokind_idx;
 input.idx=&token;
 output.kind=iokind_idx;
 output.idx=&target;
 srand(1);
 bann_new(&net,INPUTS,HIDDEN,1,2);
 bann_setweightactivation(&net,0,bann_activation_tanh);
 bann_setweightactivation(&net,1,bann_activation_softmax);
 bann_setlearningrate(&net,0.04f,bann_optimizer_off,0);
 bannrnn_new(&sequence,&net,TRAIN_DELAY+2);
 if(!sequence.seqlen)
  {
   fprintf(stderr,"Could not initialize the recurrent training buffers.\n");
   bann_delete(&net);
   return 1;
  }

 for(trial=0;trial<TRAIN_STEPS;trial++)
  {
   int delay=1+rand()%TRAIN_DELAY,length=delay+2;
   real error=0;
   label=rand()%2;
   tokens[0]=label;
   for(step=1;step<=delay;step++)
    tokens[step]=X+rand()%2;
   tokens[length-1]=QUERY;

   bannrnn_start(&sequence,&net,1);
   bannrnn_hprev(&sequence,&net,-1,bannrnn_push);
   for(step=0;step<length;step++)
    {
     symbol=tokens[step];
     bann_predict(&net,&input);
     bannrnn_hidout(&sequence,&net,step,bannrnn_push);
    }
   for(step=length-1;step>=0;step--)
    {
     symbol=tokens[step];
     /* Empty targets contribute no local loss. Future gradients still flow. */
     target.hotcnt=step==length-1;
     error+=bannrnn_backward(&sequence,&net,step,&input,&output);
    }
   bannrnn_updatenntraining(&sequence,&net,error,length);
   if(trial>=TRAIN_STEPS-100)
    loss+=error;
  }
 printf("Mean query loss over the last 100 training sequences: %.4f\n",loss/100);

 for(size_t i=0;i<sizeof(test_delays)/sizeof(test_delays[0]);i++)
  {
   int delay=test_delays[i],correct=0;
   for(trial=0;trial<TEST_TRIALS;trial++)
    {
     const real *scores;
     label=trial%2;
     bannrnn_start(NULL,&net,1);
     symbol=label;
     bann_predict(&net,&input);
     for(step=0;step<delay;step++)
      {
       symbol=X+rand()%2;
       bann_predict(&net,&input);
      }
     symbol=QUERY;
     scores=bann_predict(&net,&input);
     correct+=(scores[B]>scores[A] ? B : A)==label;
    }
   printf("%2d distractors: %d/%d correct%s\n",delay,correct,TEST_TRIALS,
          delay>TRAIN_DELAY ? " (longer than training)" : "");
  }
 bannrnn_delete(&sequence);
 bann_delete(&net);
 return 0;
}
