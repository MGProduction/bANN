//
// bANN - v0.01 - bare neural network single-file header-only implementation
//                without any dependency apart from standard libs
//
//                by Marco Giorgini (@marcogiorgini) 2024 -
//
//                This small library is similar and inspired by Tinn / Gennann
//                
//                Features:
//                - multilayer network (with one or more hidden layers)
//                - cached) signmoid activation / tanh / reLU 
//                - single memory allocation for weights
//                - basic autooptimization of learning rate
//                - multiple input/output structures
//                - write / save (normal&quantized)
//                - predict normal&quantized (bannQ.h)
//
//   COMPILING & LINKING
//      In one C/C++ file that #includes this file, do this:
//         #define BANN_IMPLEMENTATION
//      before the #include. That will create the implementation in that file.
//
// LICENSE
//
//   See end of file.
//

#ifndef _H_BANN_H
#define _H_BANN_H

#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------------------

typedef float         real;
//typedef double        real;
typedef unsigned char byte;

#if !defined(BANN_IO_DEFINED)
enum bann_iokind{
 iokind_standard=0,
 iokind_idx=1,
 iokind_fidx=2,
};

typedef struct{
 int*hotpos;
 int hotcnt;
}bann_idx;
typedef struct{
 int *hotpos; 
 real*hotval;
 int  hotcnt;
}bann_fidx;
typedef struct{
 union{
  real     *linear;
  bann_idx *idx;
  bann_fidx*fidx;
 };
 enum bann_iokind kind;
}bann_io;
#define BANN_IO_DEFINED
#endif

// ---------------------------------------------------------------------

typedef real (*bact)(real f);

// matrix between layers
typedef struct{
 int  ins,outs; 
 real*bias;       // bias values
 real*w;          // weight values
 real*o;          // output values
 real*d;          // delta values
 bact activation; // activation function
 bact prime;      // activation function
 int  flags;
}weight;

typedef struct{
 real   learningrate;
 int    optimize;

 real   errsum,preverrsum;
 int    errcnt,speed;
}bopt;

#if defined(BANN_RNN)
typedef struct{
 weight*_weights;
 weight*_rweights;
 int    weightscount;

 real  *outseq;
 real  *hprev;
 real  *hidseq; 
 int    seqlen;

 real  *_adagrad[3];				// memory variable for adagrad

}bannrnn;
#endif

// nnet struct
typedef struct{
 weight*weights;
#if defined(BANN_RNN)
 weight*rweights;
#endif
 int    weightscount;
 int    linearsize;
 real  *linear;
 
 bopt   opt;
}bann;

#define bann_float 0
#define bann_8bit  1

void bann_new(bann*net,int input,int hidden,int hiddencount,int output);
void bann_delete(bann*net);

#define bann_activation_sigmoid   0
#define bann_activation_linear    1
#define bann_activation_relu      2
#define bann_activation_tanh      3
#define bann_activation_softmax   4

void bann_setweightactivation(bann*net,int layernumber,int actmethod);

#define bann_optimizer_off              0
#define bann_optimizer_adaptive        10
#define bann_optimizer_adaptivesteady  11
#define bann_optimizer_adaptiveslow    12
#define bann_optimizer_decay           20

void bann_setlearningrate(bann*net,real learningrate,int optimizemode,int speed);

real*bann_predict(bann*net,const bann_io*input);
real bann_backward(bann*net,const bann_io*input,const bann_io*output);
real bann_train(bann*net,const bann_io*input,const bann_io*output);
void bann_optimize(bann*net,real err);
#if defined(BANN_RNN)
#define bannrnn_push 1
#define bannrnn_pull 0
void bannrnn_new(bannrnn*hlp,bann*net,int seqlen);
void bannrnn_delete(bannrnn*hlp);
void bannrnn_start(bannrnn*hlp,bann*net,int first);
real bannrnn_backward(bannrnn*hlp,bann*net,int s,const bann_io*input,const bann_io*output);
void bannrnn_updatenntraining(bannrnn*hlp,bann*net,real err,int seqlen);
int  bannrnn_hprev(bannrnn*hlp,bann*N,int s,int mode);
int  bannrnn_hidout(bannrnn*hlp,bann*N,int s,int mode);
#endif

int  bann_write(bann*net,const char*file,int way);
int  bann_read(bann*net,const char*file);

#define BIAS_LIMIT 4
#if defined(BANN_RNN)
#define bann_protectbias(a) {if(a<-BIAS_LIMIT) a=-BIAS_LIMIT; else if(a>BIAS_LIMIT) a=BIAS_LIMIT;}
#else
#define bann_protectbias(a) 
#endif
#if defined(BANN_RNN)
#define bann_protect(a)     {if(a<-BIAS_LIMIT) a=-BIAS_LIMIT; else if(a>BIAS_LIMIT) a=BIAS_LIMIT;}
#else
#define bann_protect(a)     
#endif

// ---------------------------------------------------------------------

// ---------------------------------------------------------------------

#if defined(BANN_IMPLEMENTATION)

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <assert.h>

#if defined(BANN_DEBUG)
int sanitycheck(real v)
{
 if(_isnan(v))
  {
   printf("nan!\n");
   return 0;
  }
 else
  return 1;
}
real invalid()
{
 printf("out of range!\n");
 return 0;
}
real nn_getweight(bann*net,int l,int r,int c)
{
 if((l<0)||(l>=net->weightscount)||(r<0)||(r>=net->weights[l].ins)||(c<0)||(c>=net->weights[l].outs))
  return invalid();
 else
  return net->weights[l].w[c*net->weights[l].ins+r];
}
void nn_setweight(bann*net,int l,int r,int c,real v)
{
 if((l<0)||(l>=net->weightscount)||(r<0)||(r>=net->weights[l].ins)||(c<0)||(c>=net->weights[l].outs))
  invalid();
 else
  net->weights[l].w[c*net->weights[l].ins+r]=v;
}
real nn_getoutput(bann*net,int l,int c)
{
 if((l<0)||(l>=net->weightscount)||(c<0)||(c>=net->weights[l].outs))
  return invalid();
 else
  return net->weights[l].o[c];;
}
real nn_getdelta(bann*net,int l,int r)
{
 if((l<0)||(l>=net->weightscount)||(r<0)||(r>=net->weights[l].outs))
  return invalid();
 else
  return net->weights[l].d[r];
}
void nn_setoutput(bann*net,int l,int c,real v)
{
 if((l<0)||(l>=net->weightscount)||(c<0)||(c>=net->weights[l].outs))
  invalid();
 else
  net->weights[l].o[c]=v;
}
void nn_setdelta(bann*net,int l,int r,real v)
{
 if((l<0)||(l>=net->weightscount)||(r<0)||(r>=net->weights[l].outs))
  invalid();
 else
  net->weights[l].d[r]=v;
}
#else
#define sanitycheck(a)
#define nn_getweight(net,l,r,c) net->weights[l].w[c*net->weights[l].ins+r]
#define nn_getoutput(net,l,c) net->weights[l].o[c]
#define nn_getdelta(net,l,r) net->weights[l].d[r]
#define nn_setweight(net,l,r,c,v) net->weights[l].w[c*net->weights[l].ins+r]=v
#define nn_setoutput(net,l,c,v) net->weights[l].o[c]=v
#define nn_setdelta(net,l,r,v) net->weights[l].d[r]=v
#endif

// INTERNAL - random function
real bann_rand()
{
 return (rand()/(real)RAND_MAX)-0.5f;
}

// INTERNAL - sigmoid cache code
#define _sigmoid_dom_min     -32
#define _sigmoid_dom_max      32
#define _sigmoid_table_size 2048
real _sigmoid_interval;
real _sigmoid[_sigmoid_table_size];
real bann_sigmoid_direct(real a)
{
 return 1.0f / (1.0f + expf(-a));
}
real bann_softmax(real a)
{
 real ret;
 if(a>15)
  a=15;
 else
 if(a<-15)
  a=-15;
 ret=expf(a);
 return ret;
}
real bann_softmax_prime(real a)
{
 return a;
}
real bann_tanh(real a)
{
 return tanhf(a);
}
real bann_tanh_prime(real a)
{
 return 1-a*a;
}
real bann_linear(real f)
{
 return f;
}
real bann_relu(real f)
{
 if(f>0) return f;
 else    return 0;
}
real bann_relu_prime(real f)
{
 if(f>0) return 1;
 else    return 0;
}
real bann_sigmoid_prime(real f)
{
 return f*(1-f);
}
real bann_sigmoid(real f)
{
 if (f < _sigmoid_dom_min) 
  return _sigmoid[0];
 else
 if (f >= _sigmoid_dom_max) 
  return _sigmoid[_sigmoid_table_size-1];
 else
  {
   int j=(int)((f-_sigmoid_dom_min)*_sigmoid_interval+0.5f);  
   if(j>=_sigmoid_table_size) 
    return _sigmoid[_sigmoid_table_size-1];
   else
    return _sigmoid[j];
  }
}
void bann_buildsigmoidcache()
{
 if(_sigmoid_interval==0)
  {
   int  i;
   real f;
   _sigmoid_interval=(_sigmoid_table_size/(_sigmoid_dom_max-_sigmoid_dom_min));
   f=((_sigmoid_dom_max-_sigmoid_dom_min)/(real)_sigmoid_table_size);
   for (i = 0; i < _sigmoid_table_size; ++i) 
    {
     real a=_sigmoid_dom_min+f*i,b=a;
     if(a<-45)
      b=a=0;
     else
     if(a>45)
      b=a=1;
     else
      {
       a=(real)(1.0f/(1+exp(-a)));
       b=expf(b);
      }
     _sigmoid[i]=a;   
    }
  }
}

void bann_enableweightactivation(bann*net,int layernumber)
{ 
 switch(net->weights[layernumber].flags&0x7)
  {
  case bann_activation_sigmoid:
   net->weights[layernumber].activation=bann_sigmoid;
   net->weights[layernumber].prime=bann_sigmoid_prime;
   break;
  case bann_activation_linear:
   net->weights[layernumber].activation=bann_linear;
   net->weights[layernumber].prime=bann_linear;
   break;
  case bann_activation_relu:
   net->weights[layernumber].activation=bann_relu;
   net->weights[layernumber].prime=bann_relu_prime;
   break;
  case bann_activation_tanh:
   net->weights[layernumber].activation=bann_tanh;
   net->weights[layernumber].prime=bann_tanh_prime;
   break;
   case bann_activation_softmax: 
    net->weights[layernumber].activation=bann_softmax;
    net->weights[layernumber].prime=bann_softmax_prime;
    break;
  }
}

void bann_setweightactivation(bann*net,int layernumber,int actmethod)
{
 net->weights[layernumber].flags=(net->weights[layernumber].flags&0xFFFFFFF8)|(actmethod&0x7);
 bann_enableweightactivation(net,layernumber);
}

// INTERNAL - set weight info - using linear pointer
real*bann_setweight(bann*net,int layernumber,int ins,int outs,real*ptr,int actmethod,int fillwithrandom)
{
 int    i;
 weight*w=&net->weights[layernumber];
 
 w->ins=ins; 
 w->outs=outs;

 w->bias=ptr;
 if(fillwithrandom)  for(i=0;i<outs;i++)     w->bias[i]=bann_rand();
 ptr+=outs;

 w->w=ptr;
 if(fillwithrandom)  for(i=0;i<ins*outs;i++) w->w[i]=bann_rand();
 ptr+=ins*outs;

 w->flags=actmethod;
 bann_enableweightactivation(net,layernumber);
 return ptr;
}

real*bann_setweighttemp(bann*net,int layernumber,real*ptr)
{
 weight*w=&net->weights[layernumber];
 w->o=ptr;
 ptr+=w->outs;
 w->d=ptr;
 ptr+=w->outs;  
 return ptr;
}

#if defined(BANN_RNN)
void bannrnn_new(bannrnn*hlp,bann*net,int seqlen)
{
 int i;
 memset(hlp,0,sizeof(bannrnn));
 hlp->outseq=(real*)malloc(net->weights[net->weightscount-1].outs*seqlen*sizeof(hlp->outseq[0]));
 hlp->hidseq=(real*)malloc(net->weights[0].outs*seqlen*sizeof(hlp->hidseq[0]));
 hlp->hprev=(real*)malloc(net->rweights[0].ins*(seqlen+2)*sizeof(hlp->hprev[0]));
 hlp->seqlen=seqlen;

 hlp->_weights=(weight*)malloc((net->weightscount+1)*sizeof(weight));
 memset(hlp->_weights,0,(net->weightscount+1)*sizeof(weight));
 for(i=0;i<=net->weightscount;i++)
  {
   hlp->_adagrad[i]=(real*)malloc(net->weights[i].ins*net->weights[i].outs*sizeof(hlp->_weights[i].w[0]));
   hlp->_weights[i].ins=net->weights[i].ins;
   hlp->_weights[i].outs=net->weights[i].outs;
   hlp->_weights[i].w=(real*)malloc(net->weights[i].ins*net->weights[i].outs*sizeof(hlp->_weights[i].w[0]));
   hlp->_weights[i].o=(real*)malloc(net->weights[i].ins*net->weights[i].outs*sizeof(hlp->_weights[i].w[0]));
   hlp->_weights[i].d=(real*)malloc(net->weights[i].ins*net->weights[i].outs*sizeof(hlp->_weights[i].w[0]));
   hlp->_weights[i].bias=(real*)malloc(net->weights[i].outs*sizeof(hlp->_weights[i].bias[0]));   
  } 
 hlp->_rweights=&hlp->_weights[net->weightscount]; 
 hlp->weightscount=net->weightscount;
}

void bannrnn_delete(bannrnn*hlp)
{
 int i;
 for(i=0;i<=hlp->weightscount;i++)
  {
   free(hlp->_adagrad[i]);
   free(hlp->_weights[i].w);
   free(hlp->_weights[i].d);
   free(hlp->_weights[i].o);
   free(hlp->_weights[i].bias);   
  }
 free(hlp->_weights);

 free(hlp->outseq);
 free(hlp->hidseq);
 free(hlp->hprev);
}
void bannrnn_start(bannrnn*net,bann*N,int first)
{
 int i;
 if(first)
  memset(N->weights[net->weightscount].o,0,N->weights[N->weightscount].outs*sizeof(N->weights[N->weightscount].o[0]));
 if(net&&net->_weights)
  {
   for(i=0;i<=net->weightscount;i++)
    {     
     memset(net->_weights[i].w,0,net->_weights[i].ins*net->_weights[i].outs*sizeof(net->_weights[i].w[0]));
     memset(net->_weights[i].o,0,net->_weights[i].ins*net->_weights[i].outs*sizeof(net->_weights[i].w[0]));
     memset(net->_weights[i].d,0,net->_weights[i].ins*net->_weights[i].outs*sizeof(net->_weights[i].w[0]));
     memset(net->_weights[i].bias,0,net->_weights[i].outs*sizeof(net->_weights[i].bias[0]));     
    }
   memset(net->_rweights[0].o,0,net->_weights[net->weightscount-1].outs*sizeof(net->_rweights[0].o[0]));
  }
}
void bannrnn_updatenntraining(bannrnn*hlp,bann*net,real err,int seqlen)
{
 int l;
 err/=seqlen;
 for(l=0;l<=net->weightscount;l++)
  {
   int c,r;
   memset(hlp->_adagrad[l],0,net->weights[l].ins*net->weights[l].outs*sizeof(hlp->_adagrad[l][0]));
   for(c=0;c<net->weights[l].outs;c++)
    {
     real val;
     for(r=0;r<net->weights[l].ins;r++)
      {
       val=hlp->_weights[l].w[c*net->weights[l].ins+r];
       hlp->_adagrad[l][c*net->weights[l].ins+r]+=val*val;
       
       val=-net->opt.learningrate*val/sqrtf(hlp->_adagrad[l][c*net->weights[l].ins+r] + 1e-8);        

       if(val)
        {
         net->weights[l].w[c*net->weights[l].ins+r]+=val;
         if(net->weights[l].w[c*net->weights[l].ins+r]<-5)
          net->weights[l].w[c*net->weights[l].ins+r]=-5;
         else
         if(net->weights[l].w[c*net->weights[l].ins+r]>5)
          net->weights[l].w[c*net->weights[l].ins+r]=5;
        }
      }
      val=hlp->_weights[l].bias[c];
      val=-net->opt.learningrate*val/sqrtf(val*val + 1e-8);        
      net->weights[l].bias[c]-=val;
      if(net->weights[l].bias[c]<-5)
       net->weights[l].bias[c]=-5;
      else
      if(net->weights[l].bias[c]>5)
       net->weights[l].bias[c]=5;
     }
  }
 if(net->opt.optimize)
  bann_optimize(net,err);
}
int bannrnn_hprev(bannrnn*hlp,bann*N,int s,int mode)
{
 if((s>=0)&&(s<hlp->seqlen+1))
  {
   if(mode==bannrnn_push)    
    memcpy(hlp->hprev+(s+1)*N->rweights[0].ins,N->rweights[0].o,N->rweights[0].ins*sizeof(hlp->hprev[0]));
   else
    if(s==hlp->seqlen) // last one should be set as ready to go on
     memcpy(N->rweights[0].o,hlp->hprev+(s+1)*N->rweights[0].ins,N->rweights[0].ins*sizeof(hlp->hprev[0]));
    else // other ones are instead used as BEFORE hidden status
     memcpy(N->rweights[0].d,hlp->hprev+(s+1)*N->rweights[0].ins,N->rweights[0].ins*sizeof(hlp->hprev[0]));
   return 1;
  }
 else
  return 0;
}
int bannrnn_hidout(bannrnn*hlp,bann*N,int s,int mode)
{
 if((s>=0)&&(s<hlp->seqlen))
  {
   if(mode==bannrnn_push)
    {
     memcpy(hlp->hidseq+s*N->weights[0].outs,N->weights[0].o,N->weights[0].outs*sizeof(hlp->outseq[0]));         
     memcpy(hlp->outseq+s*N->weights[1].outs,N->weights[1].o,N->weights[1].outs*sizeof(hlp->hidseq[0]));                  
    }
   else
    {
     memcpy(N->weights[0].o,hlp->hidseq+s*N->weights[0].outs,N->weights[0].outs*sizeof(hlp->outseq[0]));         
     memcpy(N->weights[1].o,hlp->outseq+s*N->weights[1].outs,N->weights[1].outs*sizeof(hlp->hidseq[0]));                  
    }
   return 1;
  }
 else
  return 0;
}
real bannrnn_backward(bannrnn*hlp,bann*net,int s,const bann_io*input,const bann_io*output)
{
 /*
 {
    for(op = 0; op < nOutputNodes; op++){
      dy[op][0] = ps[op][t];
    }
    dy[targets[t]][0] -= 1;

    // calculate delwhy = (vocab_sizex1)x(nHiddenNodesx1)T
		// =(vocab_size x 1)x(1 x nHiddenNodes)= (vocab_size x nHiddenNodes)
    for(op = 0; op < nOutputNodes; op++){
      for(hi = 0; hi < nHiddenNodes; hi++){
        delwhy[op][hi] += dy[op][0]*hs[hi][t];
      }
      delwhy[op][nHiddenNodes] += dy[op][0];  // derivative of output bias
    }

    // backprop into h
    // dh = (vocab_size x nHiddenNodes)T x (vocab_size x 1)
		// = (nHiddenNodes x vocab_size)x(vocab_size x 1)
		// = (nHiddenNodes x 1)
    for(hi = 0; hi < nHiddenNodes; hi++){
      sum =0;
      for(op = 0; op < nOutputNodes; op++){
          sum += W_hidden_output[op][hi]*dy[op][0];
      }
      dh[hi][0] = sum + dhnext[hi][0];
    }

    //backprop through tanh nonlinearity
    for(hi = 0; hi <nHiddenNodes; hi++){
      dhraw[hi][0] = (1 - hs[hi][t]*hs[hi][t]) * dh[hi][0];
      delwhh[hi][nHiddenNodes] += dhraw[hi][0];
    }
    // delwxh = (nHiddenNodes x 1)x(vocab_size x 1)T
		// =(nHiddenNodes x 1)x(1x vocab_size )= (nHiddenNodes x vocab_size)
    // delwhh = (nHiddenNodes x 1)x(nHiddenNodes x 1)T
		// =(nHiddenNodes x1)x(1x nHiddenNodes)
		// = (nHiddenNodes x nHiddenNodes)
    // dhnext = (nHiddenNodes x nHiddenNodes)T x(nHiddenNodes x1)
		// = (nHiddenNodesx1)
    for(hi = 0; hi < nHiddenNodes; hi++){
      for(in = 0; in < nInputNodes; in++){
        delwxh[hi][in] += dhraw[hi][0]*xs[in][t];
      }
      if(t==0){
        for(op = 0; op < nHiddenNodes; op++){
          delwhh[hi][op] += dhraw[hi][0]* hprev[op][0];
        }
      }
      else{
        for(op = 0; op < nHiddenNodes; op++){
          delwhh[hi][op] += dhraw[hi][0]* hs[op][t-1];
        }
      }
      sum = 0;
      for(op = 0; op < nHiddenNodes; op++){
        sum += W_hidden_hidden[op][hi]*dhraw[op][0];
      }
      dhnext[hi][0]=sum;
    }
  }
 */
 int  c,r,l=net->weightscount-1,h=0,odd=0;
 real err=0; 
 real*toutput=hlp->outseq+s*net->weights[1].outs;
 real*thidden=hlp->hidseq+s*net->weights[0].outs;
 real*tthidden=hlp->hidseq+((s+1)-1)*net->weights[0].outs;
 // calc error deltas
 for(c=0;c<net->weights[l].outs;c++)
  {
   real error,val=0;
   switch(output->kind)
    {
     case iokind_standard:
      val=output->linear[c];
     break;
     case iokind_idx:
      if((h<output->idx->hotcnt)&&(c==output->idx->hotpos[h]))
       {
        val=1;
        h++;
       }
      break;
     case iokind_fidx:
      if((h<output->fidx->hotcnt)&&(c==output->fidx->hotpos[h]))
       {
        val=output->fidx->hotval[h];
        h++;
       }
      break;
    }  

   /*
   for(op = 0; op < nOutputNodes; op++){
      for(hi = 0; hi < nHiddenNodes; hi++){
        delwhy[op][hi] += dy[op][0]*hs[hi][t];
      }
      delwhy[op][nHiddenNodes] += dy[op][0];  // derivative of output bias
    }
   */
   toutput[c]-=val;
   error=toutput[c];
   for(r=0;r<net->weights[l].ins;r++)
    //hlp->delwhy
    hlp->_weights[1].w[c*net->weights[l].ins+r]+=error*thidden[r];

   // manca il bias
   hlp->_weights[1].bias[c]+=error;

   err+=0.5f*error*error;   
  }

 /*
 for(hi = 0; hi < nHiddenNodes; hi++){
      sum =0;
      for(op = 0; op < nOutputNodes; op++){
          sum += W_hidden_output[op][hi]*dy[op][0];
      }
      dh[hi][0] = sum + dhnext[hi][0];
    }
 */
 {
 for(r=0;r<net->weights[l].ins;r++)
  {
   real sum=0;
   for(c=0;c<net->weights[l].outs;c++)
    sum+=net->weights[l].w[c*net->weights[l].ins+r]*toutput[c];
   //hlp->dh[
   hlp->_weights[0].d[r]=sum+hlp->_rweights[0].o[r];//hlp->dhnext[r];
  }
 }

  /*
  for(hi = 0; hi <nHiddenNodes; hi++){
      dhraw[hi][0] = (1 - hs[hi][t]*hs[hi][t]) * dh[hi][0];
      delwhh[hi][nHiddenNodes] += dhraw[hi][0];
    }
  */
  for(r=0;r<net->weights[l].ins;r++)
  {
    //hlp->dhraw[
    hlp->_rweights[0].d[r] = (1 - thidden[r]*thidden[r]) * hlp->_weights[0].d[r];//hlp->dh[r];
    //hlp->delwhh
    hlp->_rweights[0].bias[r] += hlp->_rweights[0].d[r];//hlp->dhraw[r];
  }

  for(r=0;r<net->weights[0].outs;r++)
   {
    int  cc;
    /*
    for(in = 0; in < nInputNodes; in++){
     delwxh[hi][in] += dhraw[hi][0]*xs[in][t];
    */
    for(cc=0;cc<input->idx->hotcnt;cc++)
     {      
      int c=input->idx->hotpos[cc];
      hlp->_weights[0].w[r*net->weights[0].ins+c] += hlp->_rweights[0].d[r];
     }
   }
  for(r=0;r<net->rweights[0].outs;r++)
   {
    real sum=0;
    /*
    for(op = 0; op < nHiddenNodes; op++){
          delwhh[hi][op] += dhraw[hi][0]* hs[op][t-1];
    */ 
    for(c=0;c<net->rweights[0].ins;c++)
     {
      hlp->_rweights[0].w[r*net->rweights[0].ins+c] += hlp->_rweights[0].d[r] * tthidden[c];
      if(hlp->_rweights[0].w[r*net->rweights[0].ins+c]>10)
       odd++;
     }    
    /*
    sum = 0;
    for(op = 0; op < nHiddenNodes; op++){
      sum += W_hidden_hidden[op][hi]*dhraw[op][0];
    }
    dhnext[hi][0]=sum;
    */
    sum=0;
    for(c=0;c<net->rweights[0].ins;c++)
     sum += net->rweights[0].w[c*net->rweights[0].ins+r]*hlp->_rweights[0].d[c];//hlp->dhraw[c];
    //hlp->dhnext[r]=sum;
    hlp->_rweights[0].o[r]=sum;
    if(sum>4)
     odd++;
  }

/*
 for(;l>0;l--)
  for(r=0;r<net->weights[l].ins;r++)
   {
    real delta=0;
    for(c=0;c<net->weights[l].outs;c++)
     delta += nn_getdelta(net,l,c) * nn_getweight(net,l,r,c);
    delta*=net->weights[l].prime(nn_getoutput(net,l-1,r));
    sanitycheck(delta);
    nn_setdelta(net,l-1,r,delta);
   }
 // backpropagation
 for(l=0;l<net->weightscount;l++)
  {
   for(c=0;c<net->weights[l].outs;c++)
    {
     real delta=net->weights[l].d[c];
     hlp->_weights[l].bias[c]-=delta;
     sanitycheck(hlp->_weights[l].bias[c]);
     bann_protectbias(hlp->_weights[l].bias[c]);
     if(l==0)
     {
      switch(input->kind)
       {
        case standard:
         for(r=0;r<net->weights[l].ins;r++)
          {
           hlp->_weights[l].w[c*net->weights[l].ins+r]+=delta*input->linear[r];
           sanitycheck(hlp->_weights[l].w[c*net->weights[l].ins+r]);
           bann_protect(hlp->_weights[l].w[c*net->weights[l].ins+r]);
          }
         break;
        case idx:
         for(r=0;r<input->idx->hotcnt;r++)
          {      
           int rr=input->idx->hotpos[r];
           hlp->_weights[l].w[c*net->weights[l].ins+rr]+=delta;
           sanitycheck(hlp->_weights[l].w[c*net->weights[l].ins+rr]);
           bann_protect(hlp->_weights[l].w[c*net->weights[l].ins+rr]);
          }
         break;
        case fidx:
         for(r=0;r<input->fidx->hotcnt;r++)
          {      
           int rr=input->fidx->hotpos[r];
           hlp->_weights[l].w[c*net->weights[l].ins+rr]+=delta*input->fidx->hotval[r];
           sanitycheck(hlp->_weights[l].w[c*net->weights[l].ins+rr]);
           bann_protect(hlp->_weights[l].w[c*net->weights[l].ins+rr]);
          }
         break;
       }
      {
       real sum=0;
       for(r=0;r<net->rweights[0].ins;r++)
        {
         hlp->_rweights[0].w[c*net->rweights[0].ins+r]+=delta*net->rweights[0].d[r];
         sanitycheck(hlp->_rweights[0].w[c*net->rweights[0].ins+r]);
         bann_protect(hlp->_rweights[0].w[c*net->rweights[0].ins+r]);
         sum+=hlp->_rweights[0].w[c*net->rweights[0].ins+r]*delta;
        }
       hlp->_rweights[0].o[c]=sum;
      }
     }
     else
      {
       for(r=0;r<net->weights[l].ins;r++)
       {
        hlp->_weights[l].w[c*net->weights[l].ins+r]+=delta*net->weights[l-1].o[r];
        sanitycheck(hlp->_weights[l].w[c*net->weights[l].ins+r]);
        bann_protect(hlp->_weights[l].w[c*net->weights[l].ins+r]);
       }
      }
    }
  }*/
 return err;
}
#endif

// create the bANN structure
void bann_new(bann*net,int input,int hidden,int hiddencount,int output)
{
 int  l=0,h;
 real*ptr,startingratio=0.25f;
 memset(net,0,sizeof(bann));
 // allocate all the weights
 net->linearsize=( ((input+1)*hidden+hidden+hidden) + 
                   ((hidden+1)*output+output+output) + 
                   ((hidden+1)*hidden+hidden*2)*(hiddencount-1) ); 
#if defined(BANN_RNN)
 net->linearsize+=((hidden+1)*hidden+hidden*2);
#endif
 ptr=net->linear=(real*)malloc( net->linearsize * sizeof(net->linear[0]) );
 assert(net->linear!=NULL);
 // allocate the layers
#if defined(BANN_RNN)
 net->weightscount=(1+1+(hiddencount-1));
 net->weights=(weight*)malloc( (net->weightscount+1) * sizeof(net->weights[0]) );
 net->rweights=&net->weights[net->weightscount];
#else
 net->weightscount=(1+1+(hiddencount-1));
 net->weights=(weight*)malloc( net->weightscount * sizeof(net->weights[0]) );
#endif
 assert(net->weights!=NULL);
 // set weight size and ptr
 ptr=bann_setweight(net,l++,input,hidden,ptr,bann_activation_sigmoid,1); 
 for(h=0;h<net->weightscount-2;h++)
  ptr=bann_setweight(net,l++,hidden,hidden,ptr,bann_activation_sigmoid,1);
 ptr=bann_setweight(net,l++,hidden,output,ptr,bann_activation_sigmoid,1);
 // set weight input and delta memory
 for(l=0;l<net->weightscount;l++)
  ptr=bann_setweighttemp(net,l,ptr);
#if defined(BANN_RNN)
 ptr=bann_setweight(net,l,hidden,hidden,ptr,bann_activation_sigmoid,1); 
 ptr=bann_setweighttemp(net,l,ptr);
#endif
 assert((ptr-net->linear)==net->linearsize);
 // set learning rate in auto adjust mode
 if(input+output<8)
  startingratio=5.0f;
 else if(input+output<16)
  startingratio=2.0f;
 else if(input+output<32)
  startingratio=1.0f;
 else if(input+output<128)
  startingratio=0.5f;
 else
  startingratio=0.25f;
 bann_setlearningrate(net,startingratio,bann_optimizer_adaptive,input*output);
 // build sigmoid table
 bann_buildsigmoidcache(); 
}

// bann execution
real*bann_predict(bann*net,const bann_io*input)
{
 int  i,j,l,clipped=0;
 // from input layer to output one
 // v values are written on the next lever
 for(l=0;l<net->weightscount;l++)
  {   
   for(i=0;i<net->weights[l].outs;i++)
    {
     real sum=-net->weights[l].bias[i];
     if(l==0)
      {
       // use input instead of previous hidden state
       switch(input->kind)
        {
         case iokind_standard:
          for(j=0;j<net->weights[l].ins;j++)                     
           sum+=input->linear[j]*nn_getweight(net,l,j,i);       
          break;
         case iokind_idx:
          for(j=0;j<input->idx->hotcnt;j++)        
           sum+=nn_getweight(net,l,input->idx->hotpos[j],i);       
          break;
         case iokind_fidx:
          for(j=0;j<input->fidx->hotcnt;j++)        
           sum+=input->fidx->hotval[j]*nn_getweight(net,l,input->fidx->hotpos[j],i);       
          break;
        }
#if defined(BANN_RNN)
       for(j=0;j<net->rweights[0].ins;j++)         
        sum+=net->rweights[0].o[j] * net->rweights[0].w[i*net->rweights[0].ins+j];  
       sum+=-net->rweights[0].bias[i];
#endif
      }
     else
      {
       for(j=0;j<net->weights[l].ins;j++)
        sum+=net->weights[l-1].o[j]*nn_getweight(net,l,j,i);
      }
     sanitycheck(sum);
     nn_setoutput(net,l,i,net->weights[l].activation(sum));     
     sanitycheck(net->weights[l].o[i]);
     if((net->weights[l].o[i]==1.0f)||(net->weights[l].o[i]==-1.0f))
      clipped++;
    }
   if((net->weights[l].flags&7)==bann_activation_softmax)
    {
     real sum=0;
     for(i=0;i<net->weights[l].outs;i++)
      sum+=net->weights[l].o[i];
     if(sum==0)
      sum=0;
     else
      for(i=0;i<net->weights[l].outs;i++)
       {
        real val=net->weights[l].o[i]/sum;
        sanitycheck(val);
        net->weights[l].o[i]=val;
       }
     
    }
  }
 #if defined(BANN_RNN)
 for(j=0;j<net->rweights[0].ins;j++)         
  {
   net->rweights[0].d[j]=net->rweights[0].o[j];
   net->rweights[0].o[j]=net->weights[0].o[j];
  }
 #endif
  return net->weights[l-1].o;
}

void bann_setlearningrate(bann*net,real learningrate,int optimizemode,int speed)
{
 net->opt.learningrate=learningrate;
 net->opt.optimize=optimizemode;
 net->opt.speed=speed;
}

void bann_optimize(bann*net,real err)
{
 switch(net->opt.optimize)
 {
 case bann_optimizer_adaptivesteady:
 case bann_optimizer_adaptiveslow:
 case bann_optimizer_adaptive:
  {
   int  l=net->weightscount-1;
   net->opt.errsum+=err;net->opt.errcnt++;
   if(net->opt.errcnt>net->opt.speed)
    {
     real expand=1.0f,lowcut=0.25f;
     real minv=0.25f,maxv=1.75f;
     int  errlow=0;

     if(net->opt.optimize==bann_optimizer_adaptiveslow)
      {minv=0.95f;maxv=1.05f;}

     net->opt.errsum/=net->opt.errcnt;
     
     if(net->opt.errsum<0.5f) 
      if(net->opt.optimize==bann_optimizer_adaptivesteady)
       ;
      else
       errlow=1;
     if(errlow) 
      if(net->opt.speed>256) 
       net->opt.speed--;

     if(net->opt.errsum<lowcut)
      lowcut=net->opt.errsum;
     if(net->opt.preverrsum)
      if(net->opt.errsum>net->opt.preverrsum)
       expand=(net->opt.errsum/net->opt.preverrsum);
      else
      if(net->opt.errsum<net->opt.preverrsum) 
       expand=(net->opt.errsum/net->opt.preverrsum);

     net->opt.preverrsum=net->opt.errsum;
     net->opt.errsum=0;net->opt.errcnt=0;

     if(expand<minv) 
      {
       expand=minv;
       if(errlow) 
         net->opt.speed+=128;
      }
     else
      if(expand>maxv)
      {
       expand=maxv;
       if(errlow) 
        net->opt.speed+=128;
      }

     net->opt.learningrate*=expand;
     if(net->opt.learningrate<lowcut)
      net->opt.learningrate=lowcut;       
     else
     if(net->opt.learningrate>1.0f)
      net->opt.learningrate=1.0f;
    }
  }
 break;
 case bann_optimizer_decay:
  net->opt.errcnt++;
  if(net->opt.errcnt>net->opt.speed)
   {
    net->opt.learningrate*=0.95f;
    if(net->opt.learningrate<0.1f)
     net->opt.learningrate=0.1f;
    net->opt.errcnt=0;
   }
  break;
 }
}

// bann training
real bann_backward(bann*net,const bann_io*input,const bann_io*output)
{
 int  c,r,l=net->weightscount-1,h=0;
 real learningrate=net->opt.learningrate,err=0,bdelta=0; 
 // calc error deltas
 for(c=0;c<net->weights[l].outs;c++)
  {
   real error,derror,val=0;
   switch(output->kind)
    {
     case iokind_standard:
      val=output->linear[c];
     break;
     case iokind_idx:
      if((h<output->idx->hotcnt)&&(c==output->idx->hotpos[h]))
       {
        val=1;
        h++;
       }
      break;
     case iokind_fidx:
      if((h<output->fidx->hotcnt)&&(c==output->fidx->hotpos[h]))
       {
        val=output->fidx->hotval[h];
        h++;
       }
      break;
    }  
   error=(val-net->weights[l].o[c]);
   derror=error * net->weights[l].prime(net->weights[l].o[c]);
   sanitycheck(derror);
   nn_setdelta(net,l,c,derror);
   err+=0.5f*error*error;   
  }
 for(;l>0;l--)
  for(r=0;r<net->weights[l].ins;r++)
   {
    real delta=0;
    for(c=0;c<net->weights[l].outs;c++)
     delta += nn_getdelta(net,l,c) * nn_getweight(net,l,r,c);
    delta*=net->weights[l].prime(nn_getoutput(net,l-1,r));
    sanitycheck(delta);
    nn_setdelta(net,l-1,r,delta);
   }
 // backpropagation
 for(l=0;l<net->weightscount;l++)
  {
   for(c=0;c<net->weights[l].outs;c++)
    {
     real delta=net->weights[l].d[c];
     net->weights[l].bias[c]-=delta*learningrate;
     sanitycheck(net->weights[l].bias[c]);
     bann_protectbias(net->weights[l].bias[c]);
     if(l==0)
     {
      switch(input->kind)
       {
        case iokind_standard:
         for(r=0;r<net->weights[l].ins;r++)
          {
           net->weights[l].w[c*net->weights[l].ins+r]+=delta*learningrate*input->linear[r];
           sanitycheck(net->weights[l].w[c*net->weights[l].ins+r]);
           bann_protect(net->weights[l].w[c*net->weights[l].ins+r]);
          }
         break;
        case iokind_idx:
         for(r=0;r<input->idx->hotcnt;r++)
          {      
           int rr=input->idx->hotpos[r];
           net->weights[l].w[c*net->weights[l].ins+rr]+=delta*learningrate;
           sanitycheck(net->weights[l].w[c*net->weights[l].ins+rr]);
           bann_protect(net->weights[l].w[c*net->weights[l].ins+rr]);
          }
         break;
        case iokind_fidx:
         for(r=0;r<input->fidx->hotcnt;r++)
          {      
           int rr=input->fidx->hotpos[r];
           net->weights[l].w[c*net->weights[l].ins+rr]+=delta*learningrate*input->fidx->hotval[r];
           sanitycheck(net->weights[l].w[c*net->weights[l].ins+rr]);
           bann_protect(net->weights[l].w[c*net->weights[l].ins+rr]);
          }
         break;
       }
#if defined(BANN_RNN)
      for(r=0;r<net->rweights[0].ins;r++)
       {
        net->rweights[0].w[c*net->rweights[0].ins+r]+=delta*learningrate*net->rweights[0].d[r];
        sanitycheck(net->rweights[0].w[c*net->rweights[0].ins+r]);
        bann_protect(net->rweights[0].w[c*net->rweights[0].ins+r]);
       }
#endif
     }
     else
      {
       for(r=0;r<net->weights[l].ins;r++)
       {
        net->weights[l].w[c*net->weights[l].ins+r]+=delta*learningrate*net->weights[l-1].o[r];
        sanitycheck(net->weights[l].w[c*net->weights[l].ins+r]);
        bann_protect(net->weights[l].w[c*net->weights[l].ins+r]);
       }
      }
    }
  }
 if(net->opt.optimize)
  bann_optimize(net,err);
 return err;
}

real bann_train(bann*net,const bann_io*input,const bann_io*output)
{
 // forward
 bann_predict(net,input);
 // backward
 return bann_backward(net,input,output);
}
// nnet delete
void bann_delete(bann*net)
{
 free(net->weights);
 free(net->linear);
}

// bann write
int  bann_corewrite(bann*net,FILE*f,int way)
{
 int version=1,i,usedlinearsize=0;
  
 fwrite("BANN",4,1,f);
 fwrite(&version,sizeof(version),1,f);   
 
 fwrite(&way,sizeof(way),1,f);
  
 // linear size is the ALLOC size
 fwrite(&net->linearsize,sizeof(net->linearsize),1,f);  
 // but we want usedlinearsize that counts just real weights - not also outputs/delta
 for(i=0;i<net->weightscount;i++)
  usedlinearsize+=(net->weights[i].ins+1)*net->weights[i].outs;
 fwrite(&usedlinearsize,sizeof(usedlinearsize),1,f);  

 if(way==bann_8bit)
  {
   int  i;
   byte*blinear=(byte*)malloc(usedlinearsize);
   real maxf,minf,scale;
   maxf=minf=net->linear[0];
   for(i=1;i<usedlinearsize;i++)
    {
     if(net->linear[i]>maxf)      maxf=net->linear[i];
     if(net->linear[i]<minf)      minf=net->linear[i];
    }
   scale=(maxf-minf)/255;
   for(i=0;i<usedlinearsize;i++)
    {
     byte id=(byte)floor(0.5f+(net->linear[i]-minf)/(maxf-minf)*255);
     blinear[i]=id;
    }     
   fwrite(&minf,sizeof(minf),1,f);fwrite(&scale,sizeof(scale),1,f);
   fwrite(blinear,sizeof(blinear[0]),usedlinearsize,f);
   free(blinear);
  }
 else
  {
   fwrite(net->linear,sizeof(net->linear[0]),usedlinearsize,f);
   fwrite(&net->opt.optimize,sizeof(net->opt.optimize),1,f);
   fwrite(&net->opt.learningrate,sizeof(net->opt.learningrate),1,f);    
  }

 fwrite(&net->weightscount,sizeof(net->weightscount),1,f);
 for(i=0;i<net->weightscount;i++)
  {
   fwrite(&net->weights[i].ins,sizeof(net->weights[i].ins),1,f);
   fwrite(&net->weights[i].outs,sizeof(net->weights[i].outs),1,f);
   fwrite(&net->weights[i].flags,sizeof(net->weights[i].flags),1,f);
  }
 return 1;
}
int  bann_write(bann*net,const char*file,int way)
{
 FILE*f=fopen(file,"wb");
 if(f)
  {
   bann_corewrite(net,f,way);
   fclose(f);   
   return 1;
  }
 else
  return 0;
}

// bann read
int  bann_coreread(bann*net,FILE*f)
{
 int  version,i,ret=0,way;
 char magic[4];
 if((fread(magic,4,1,f)==1)&&(fread(&version,sizeof(version),1,f)==1)&&(memcmp(magic,"BANN",4)==0)&&(version==1))
  {
   real*ptr;
   int  usedlinearsize;
   memset(net,0,sizeof(bann));

   fread(&way,sizeof(way),1,f);
     
   // read net data
   fread(&net->linearsize,sizeof(net->linearsize),1,f);
   fread(&usedlinearsize,sizeof(usedlinearsize),1,f);
   ptr=net->linear=(real*)malloc( net->linearsize * sizeof(net->linear[0]) );   
   
   if(way==bann_8bit)
    {
     int  i;
     byte*blinear=(byte*)malloc(usedlinearsize);
     real base,scale;
     fread(&base,sizeof(base),1,f);fwrite(&scale,sizeof(scale),1,f);
     fread(blinear,sizeof(blinear[0]),usedlinearsize,f);
     for(i=0;i<usedlinearsize;i++)
      net->linear[i]=base+blinear[i]*scale;
     free(blinear);
    }
   else
    {
     fread(net->linear,sizeof(net->linear[0]),usedlinearsize,f);     
     fread(&net->opt.optimize,sizeof(net->opt.optimize),1,f);
     fread(&net->opt.learningrate,sizeof(net->opt.learningrate),1,f);
    }
   // read net weight structure and fill it
   fread(&net->weightscount,sizeof(net->weightscount),1,f);          
   net->weights=(weight*)malloc( net->weightscount * sizeof(net->weights[0]) );
   for(i=0;i<net->weightscount;i++)
    {
     int ins,outs,flags;
     fread(&ins,sizeof(ins),1,f);
     fread(&outs,sizeof(outs),1,f);
     fread(&flags,sizeof(flags),1,f);
     ptr=bann_setweight(net,i,ins,outs,ptr,flags,0);
    }
    // set weight input and delta memory
   for(i=0;i<net->weightscount;i++)
    ptr=bann_setweighttemp(net,i,ptr);

   // build sigmoid table
   bann_buildsigmoidcache();
   ret=1;
  }
 return ret;
}
int  bann_read(bann*net,const char*file)
{
 FILE*f=fopen(file,"rb");
 if(f)
  {
   int ret=bann_coreread(net,f);
   fclose(f);
   return ret;
  }
 else
  return 0;
}

#endif

 // ---------------------------------------------------------------------

 // ---------------------------------------------------------------------


#ifdef __cplusplus
}
#endif

#endif

//
// MIT License
// 
// Copyright (c) 2024 Marco Giorgini
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// 
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//