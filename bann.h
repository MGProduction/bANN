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

#include <stddef.h>

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
 int    loss;
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
  size_t hiddenstride;

 real **_adagrad;				// memory variable for adagrad

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

#define bann_loss_squared_error 0
#define bann_loss_cross_entropy 1
// Cross-entropy applies to softmax outputs. Returns zero for an unknown loss.
int bann_setloss(bann*net,int loss);

real*bann_predict(bann*net,const bann_io*input);
real bann_backward(bann*net,const bann_io*input,const bann_io*output);
real bann_train(bann*net,const bann_io*input,const bann_io*output);
void bann_optimize(bann*net,real err);
#if defined(BANN_RNN)
#define bannrnn_push 1
#define bannrnn_pull 0
// A zero seqlen in the helper indicates that initialization failed.
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
#include <stdint.h>
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
real bann_linear_prime(real f)
{
 (void)f;
 return 1;
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
   net->weights[layernumber].prime=bann_linear_prime;
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
    net->weights[layernumber].activation=bann_linear;
    net->weights[layernumber].prime=NULL;
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

static void bann_activation_gradient(const weight*w,real*delta)
{
 int c;
 if((w->flags&7)==bann_activation_softmax)
  {
   real dot=0;
   for(c=0;c<w->outs;c++)
    dot+=delta[c]*w->o[c];
   for(c=0;c<w->outs;c++)
    delta[c]=w->o[c]*(delta[c]-dot);
  }
 else
  for(c=0;c<w->outs;c++)
   delta[c]*=w->prime(w->o[c]);
}

#if defined(BANN_RNN)
void bannrnn_new(bannrnn*hlp,bann*net,int seqlen)
{
 int i;
 memset(hlp,0,sizeof(bannrnn));
 if(seqlen<=0)
  return;
 for(i=0;i<net->weightscount-1;i++)
  hlp->hiddenstride+=net->weights[i].outs;
 hlp->outseq=(real*)malloc((size_t)net->weights[net->weightscount-1].outs*seqlen*sizeof(real));
 hlp->hidseq=(real*)malloc((size_t)hlp->hiddenstride*seqlen*sizeof(real));
 hlp->hprev=(real*)calloc((size_t)net->rweights->ins*((size_t)seqlen+2),sizeof(real));
 hlp->weightscount=net->weightscount;
 hlp->_weights=(weight*)calloc(net->weightscount+1,sizeof(weight));
 hlp->_adagrad=(real**)calloc(net->weightscount+1,sizeof(hlp->_adagrad[0]));
 if(!hlp->outseq || !hlp->hidseq || !hlp->hprev || !hlp->_weights || !hlp->_adagrad)
  goto fail;
 for(i=0;i<=net->weightscount;i++)
  {
   hlp->_adagrad[i]=(real*)calloc((size_t)(net->weights[i].ins+1)*net->weights[i].outs,sizeof(real));
   hlp->_weights[i].ins=net->weights[i].ins;
   hlp->_weights[i].outs=net->weights[i].outs;
   hlp->_weights[i].w=(real*)malloc((size_t)net->weights[i].ins*net->weights[i].outs*sizeof(real));
   hlp->_weights[i].o=(real*)malloc(net->weights[i].outs*sizeof(real));
   hlp->_weights[i].d=(real*)malloc(net->weights[i].outs*sizeof(real));
   hlp->_weights[i].bias=(real*)malloc(net->weights[i].outs*sizeof(hlp->_weights[i].bias[0]));   
   if(!hlp->_adagrad[i] || !hlp->_weights[i].w || !hlp->_weights[i].o ||
      !hlp->_weights[i].d || !hlp->_weights[i].bias)
    goto fail;
  }
 hlp->_rweights=&hlp->_weights[net->weightscount]; 
 hlp->seqlen=seqlen;
 return;

fail:
 bannrnn_delete(hlp);
}

void bannrnn_delete(bannrnn*hlp)
{
 int i;
 for(i=0;i<=hlp->weightscount;i++)
  {
   if(hlp->_adagrad)
    free(hlp->_adagrad[i]);
   if(hlp->_weights)
    {
     free(hlp->_weights[i].w);
     free(hlp->_weights[i].d);
     free(hlp->_weights[i].o);
     free(hlp->_weights[i].bias);
    }
  }
 free(hlp->_weights);
 free(hlp->_adagrad);

 free(hlp->outseq);
 free(hlp->hidseq);
 free(hlp->hprev);
 memset(hlp,0,sizeof(bannrnn));
}
void bannrnn_start(bannrnn*net,bann*N,int first)
{
 int i;
 if(first)
  memset(N->rweights->o,0,N->rweights->outs*sizeof(real));
 if(net&&net->_weights)
  {
   for(i=0;i<=net->weightscount;i++)
    {     
     memset(net->_weights[i].w,0,net->_weights[i].ins*net->_weights[i].outs*sizeof(net->_weights[i].w[0]));
     memset(net->_weights[i].o,0,net->_weights[i].outs*sizeof(real));
     memset(net->_weights[i].d,0,net->_weights[i].outs*sizeof(real));
     memset(net->_weights[i].bias,0,net->_weights[i].outs*sizeof(net->_weights[i].bias[0]));     
    }
  }
}
void bannrnn_updatenntraining(bannrnn*hlp,bann*net,real err,int seqlen)
{
 int l;
 if(seqlen<=0 || seqlen>hlp->seqlen)
  return;
 err/=seqlen;
 for(l=0;l<=net->weightscount;l++)
  {
   int c,r;
   for(c=0;c<net->weights[l].outs;c++)
    {
     real val;
     for(r=0;r<net->weights[l].ins;r++)
      {
       val=hlp->_weights[l].w[c*net->weights[l].ins+r];
       if(val)
        {
         hlp->_adagrad[l][c*(net->weights[l].ins+1)+r]+=val*val;       
         val=-net->opt.learningrate*val/sqrtf(hlp->_adagrad[l][c*(net->weights[l].ins+1)+r] + 1e-8);                 
         //val=-net->opt.learningrate*val;

         net->weights[l].w[c*net->weights[l].ins+r]+=val;

         if(net->weights[l].w[c*net->weights[l].ins+r]<-5)
          net->weights[l].w[c*net->weights[l].ins+r]=-5;
         else
         if(net->weights[l].w[c*net->weights[l].ins+r]>5)
          net->weights[l].w[c*net->weights[l].ins+r]=5;
        }
      }
      val=hlp->_weights[l].bias[c];      
      if(val)
       {
        hlp->_adagrad[l][c*(net->weights[l].ins+1)+r]+=val*val;
        val=-net->opt.learningrate*val/sqrtf(hlp->_adagrad[l][c*(net->weights[l].ins+1)+r] + 1e-8);        
        //val=-net->opt.learningrate*val;

        net->weights[l].bias[c]-=val;

        if(net->weights[l].bias[c]<-5)
         net->weights[l].bias[c]=-5;
        else
        if(net->weights[l].bias[c]>5)
         net->weights[l].bias[c]=5;
       }
     }
  }
 if(net->opt.optimize)
  bann_optimize(net,err);
}
int bannrnn_hprev(bannrnn*hlp,bann*N,int s,int mode)
{
 if(hlp->hprev && s>=-1 && s<=hlp->seqlen)
  {
   if(mode==bannrnn_push)    
    memcpy(hlp->hprev+(s+1)*N->rweights[0].ins,N->rweights[0].o,N->rweights[0].ins*sizeof(hlp->hprev[0]));
   else
    if(s==hlp->seqlen) // restore the state for the next prediction
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
 int l;
 real*saved;
 if(s<0 || s>=hlp->seqlen)
  return 0;
 saved=hlp->hidseq+(size_t)s*hlp->hiddenstride;
 for(l=0;l<N->weightscount;l++)
  {
   weight*w=&N->weights[l];
   if(l==N->weightscount-1)
    saved=hlp->outseq+(size_t)s*w->outs;
   if(mode==bannrnn_push)
    memcpy(saved,w->o,w->outs*sizeof(real));
   else
    memcpy(w->o,saved,w->outs*sizeof(real));
   saved+=w->outs;
  }
 return 1;
}

real bannrnn_backward(bannrnn*hlp,bann*net,int s,const bann_io*input,const bann_io*output)
{
 int c,r,l=net->weightscount-1;
 real err=0,targetsum=0;
 weight*w=&net->weights[l];
 real*delta;
 const real*previous;
 int softmax=(w->flags&7)==bann_activation_softmax;

 if(!bannrnn_hidout(hlp,net,s,bannrnn_pull))
  return 0;
 delta=hlp->_weights[l].d;
 previous=s ? hlp->hidseq+(size_t)(s-1)*hlp->hiddenstride : hlp->hprev;
 memset(delta,0,w->outs*sizeof(real));
 switch(output->kind)
  {
   case iokind_standard:
    memcpy(delta,output->linear,w->outs*sizeof(real));
    break;
   case iokind_idx:
    for(c=0;c<output->idx->hotcnt;c++)
     delta[output->idx->hotpos[c]]+=1;
    break;
   case iokind_fidx:
    for(c=0;c<output->fidx->hotcnt;c++)
     delta[output->fidx->hotpos[c]]+=output->fidx->hotval[c];
    break;
  }
 if(softmax)
  for(c=0;c<w->outs;c++)
   targetsum+=delta[c];
 for(c=0;c<w->outs;c++)
  {
   if(softmax)
    {
     err-=delta[c]*logf(fmaxf(w->o[c],FLT_MIN));
     delta[c]=targetsum*w->o[c]-delta[c];
    }
   else
    {
     delta[c]=w->o[c]-delta[c];
     err+=0.5f*delta[c]*delta[c];
    }
  }
 if(!softmax)
  bann_activation_gradient(w,delta);

 for(;l>=0;l--)
  {
   weight*g=&hlp->_weights[l];
   w=&net->weights[l];
   for(c=0;c<w->outs;c++)
    {
     real*gradient=g->w+(size_t)c*w->ins;
     g->bias[c]+=g->d[c];
     if(l)
      for(r=0;r<w->ins;r++)
       gradient[r]+=g->d[c]*net->weights[l-1].o[r];
     else
      switch(input->kind)
       {
        case iokind_standard:
         for(r=0;r<w->ins;r++)
          gradient[r]+=g->d[c]*input->linear[r];
         break;
        case iokind_idx:
         for(r=0;r<input->idx->hotcnt;r++)
          gradient[input->idx->hotpos[r]]+=g->d[c];
         break;
        case iokind_fidx:
         for(r=0;r<input->fidx->hotcnt;r++)
          gradient[input->fidx->hotpos[r]]+=g->d[c]*input->fidx->hotval[r];
         break;
       }
    }
   if(l)
    {
     real*prior=hlp->_weights[l-1].d;
     for(r=0;r<w->ins;r++)
      {
       prior[r]=l==1 ? hlp->_rweights->o[r] : 0;
       for(c=0;c<w->outs;c++)
        prior[r]+=w->w[c*w->ins+r]*g->d[c];
      }
     bann_activation_gradient(&net->weights[l-1],prior);
    }
  }
 w=net->rweights;
 delta=hlp->_weights[0].d;
 for(c=0;c<w->outs;c++)
  {
   hlp->_rweights->bias[c]+=delta[c];
   for(r=0;r<w->ins;r++)
    hlp->_rweights->w[c*w->ins+r]+=delta[c]*previous[r];
  }
 for(r=0;r<w->ins;r++)
  {
   real sum=0;
   for(c=0;c<w->outs;c++)
    sum+=w->w[c*w->ins+r]*delta[c];
   hlp->_rweights->o[r]=sum;
  }
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
#if defined(BANN_RNN)
 ptr=bann_setweight(net,l,hidden,hidden,ptr,bann_activation_sigmoid,1);
#endif
 // set weight output and delta memory
 for(l=0;l<net->weightscount;l++)
  ptr=bann_setweighttemp(net,l,ptr);
#if defined(BANN_RNN)
 ptr=bann_setweighttemp(net,l,ptr);
 memset(net->rweights->o,0,hidden*sizeof(real));
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
     real max=net->weights[l].o[0];
     for(i=1;i<net->weights[l].outs;i++)
      if(net->weights[l].o[i]>max)
       max=net->weights[l].o[i];
     for(i=0;i<net->weights[l].outs;i++)
      net->weights[l].o[i]=expf(net->weights[l].o[i]-max);
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

int bann_setloss(bann*net,int loss)
{
 if(loss!=bann_loss_squared_error && loss!=bann_loss_cross_entropy)
  return 0;
 net->opt.loss=loss;
 return 1;
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
 int  c,r,l=net->weightscount-1;
 real learningrate=net->opt.learningrate,err=0;
 int cross_entropy=net->opt.loss==bann_loss_cross_entropy &&
                   (net->weights[l].flags&7)==bann_activation_softmax;
 // Store target values, then transform the output errors into logit deltas.
 memset(net->weights[l].d,0,net->weights[l].outs*sizeof(real));
 switch(output->kind)
  {
   case iokind_standard:
    memcpy(net->weights[l].d,output->linear,net->weights[l].outs*sizeof(real));
    break;
   case iokind_idx:
    for(c=0;c<output->idx->hotcnt;c++)
     net->weights[l].d[output->idx->hotpos[c]]=1;
    break;
   case iokind_fidx:
    for(c=0;c<output->fidx->hotcnt;c++)
     net->weights[l].d[output->fidx->hotpos[c]]+=output->fidx->hotval[c];
    break;
  }
 if(cross_entropy)
  {
   real targetsum=0;
   for(c=0;c<net->weights[l].outs;c++)
    targetsum+=net->weights[l].d[c];
   for(c=0;c<net->weights[l].outs;c++)
    {
     real target=net->weights[l].d[c],probability=net->weights[l].o[c];
     err-=target*logf(fmaxf(probability,FLT_MIN));
     nn_setdelta(net,l,c,target-targetsum*probability);
    }
  }
 else
  {
   for(c=0;c<net->weights[l].outs;c++)
    {
     real error=net->weights[l].d[c]-net->weights[l].o[c];
     nn_setdelta(net,l,c,error);
     err+=0.5f*error*error;
    }
   bann_activation_gradient(&net->weights[l],net->weights[l].d);
  }
 for(;l>0;l--)
  {
   for(r=0;r<net->weights[l].ins;r++)
    {
     real delta=0;
     for(c=0;c<net->weights[l].outs;c++)
      delta+=nn_getdelta(net,l,c)*nn_getweight(net,l,r,c);
     sanitycheck(delta);
     nn_setdelta(net,l-1,r,delta);
    }
   bann_activation_gradient(&net->weights[l-1],net->weights[l-1].d);
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
      net->rweights->bias[c]-=delta*learningrate;
      bann_protectbias(net->rweights->bias[c]);
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

// Version 2 includes the recurrent matrix as the final stored layer.
#if defined(BANN_RNN)
#define BANN_FILE_VERSION 2
#define BANN_RECURRENT_LAYERS 1
#else
#define BANN_FILE_VERSION 1
#define BANN_RECURRENT_LAYERS 0
#endif

int bann_corewrite(bann*net,FILE*f,int way)
{
 int version=BANN_FILE_VERSION,i,usedlinearsize=0;
 int layercount=net->weightscount+BANN_RECURRENT_LAYERS;

 for(i=0;i<layercount;i++)
  usedlinearsize+=(net->weights[i].ins+1)*net->weights[i].outs;
 if((way!=bann_float && way!=bann_8bit) ||
    fwrite("BANN",4,1,f)!=1 ||
    fwrite(&version,sizeof(version),1,f)!=1 ||
    fwrite(&way,sizeof(way),1,f)!=1 ||
    fwrite(&net->linearsize,sizeof(net->linearsize),1,f)!=1 ||
    fwrite(&usedlinearsize,sizeof(usedlinearsize),1,f)!=1)
  return 0;

 if(way==bann_8bit)
  {
   real maxf=net->linear[0],minf=maxf,scale;
   for(i=1;i<usedlinearsize;i++)
    {
     if(net->linear[i]>maxf) maxf=net->linear[i];
     if(net->linear[i]<minf) minf=net->linear[i];
    }
   scale=(maxf-minf)/255;
   if(!isfinite(scale) ||
      fwrite(&minf,sizeof(minf),1,f)!=1 ||
      fwrite(&scale,sizeof(scale),1,f)!=1)
    return 0;
   for(i=0;i<usedlinearsize;i++)
    {
     if(!isfinite(net->linear[i]) ||
        fputc(scale ? (int)floor(0.5+(net->linear[i]-minf)/(double)(maxf-minf)*255) : 0,f)==EOF)
      return 0;
    }
  }
 else if(fwrite(net->linear,sizeof(real),usedlinearsize,f)!=(size_t)usedlinearsize ||
         fwrite(&net->opt.optimize,sizeof(net->opt.optimize),1,f)!=1 ||
         fwrite(&net->opt.learningrate,sizeof(net->opt.learningrate),1,f)!=1)
  return 0;

 if(fwrite(&layercount,sizeof(layercount),1,f)!=1)
  return 0;
 for(i=0;i<layercount;i++)
  {
   weight*w=&net->weights[i];
   if(fwrite(&w->ins,sizeof(w->ins),1,f)!=1 ||
      fwrite(&w->outs,sizeof(w->outs),1,f)!=1 ||
      fwrite(&w->flags,sizeof(w->flags),1,f)!=1)
    return 0;
  }
 return 1;
}

int bann_write(bann*net,const char*file,int way)
{
 FILE*f=fopen(file,"wb");
 int ok;
 if(!f)
  return 0;
 ok=bann_corewrite(net,f,way);
 return fclose(f)==0 && ok;
}

int bann_coreread(bann*net,FILE*f)
{
 bann loaded={0};
 int version,i,way,usedlinearsize,layercount;
 uint64_t used=0,total=0;
 char magic[4];

 if(fread(magic,4,1,f)!=1 || memcmp(magic,"BANN",4)!=0 ||
    fread(&version,sizeof(version),1,f)!=1 || version!=BANN_FILE_VERSION ||
    fread(&way,sizeof(way),1,f)!=1 || (way!=bann_float && way!=bann_8bit) ||
    fread(&loaded.linearsize,sizeof(loaded.linearsize),1,f)!=1 ||
    fread(&usedlinearsize,sizeof(usedlinearsize),1,f)!=1 ||
    usedlinearsize<=0 || loaded.linearsize<usedlinearsize ||
    (size_t)loaded.linearsize>SIZE_MAX/sizeof(real))
  return 0;
 loaded.linear=(real*)malloc((size_t)loaded.linearsize*sizeof(real));
 if(!loaded.linear)
  return 0;

 if(way==bann_8bit)
  {
   real base,scale;
   if(fread(&base,sizeof(base),1,f)!=1 || fread(&scale,sizeof(scale),1,f)!=1 ||
      !isfinite(base) || !isfinite(scale) || scale<0)
    goto fail;
   for(i=0;i<usedlinearsize;i++)
    {
     int value=fgetc(f);
     if(value==EOF)
      goto fail;
     loaded.linear[i]=base+value*scale;
    }
  }
 else if(fread(loaded.linear,sizeof(real),usedlinearsize,f)!=(size_t)usedlinearsize ||
         fread(&loaded.opt.optimize,sizeof(loaded.opt.optimize),1,f)!=1 ||
         fread(&loaded.opt.learningrate,sizeof(loaded.opt.learningrate),1,f)!=1)
  goto fail;

 if(fread(&layercount,sizeof(layercount),1,f)!=1 ||
    layercount<2+BANN_RECURRENT_LAYERS || layercount>usedlinearsize ||
    (size_t)layercount>SIZE_MAX/sizeof(weight))
  goto fail;
 loaded.weightscount=layercount-BANN_RECURRENT_LAYERS;
 loaded.weights=(weight*)malloc((size_t)layercount*sizeof(weight));
 if(!loaded.weights)
  goto fail;

 for(i=0;i<layercount;i++)
  {
   int ins,outs,flags;
   uint64_t count;
   if(fread(&ins,sizeof(ins),1,f)!=1 || fread(&outs,sizeof(outs),1,f)!=1 ||
      fread(&flags,sizeof(flags),1,f)!=1 || ins<=0 || outs<=0 ||
      (flags&7)>bann_activation_softmax)
    goto fail;
   if(i && i<loaded.weightscount && loaded.weights[i-1].outs!=ins)
    goto fail;
#if defined(BANN_RNN)
   if(i==loaded.weightscount && (ins!=loaded.weights[0].outs || outs!=ins))
    goto fail;
#endif
   count=((uint64_t)ins+1)*outs;
   if(used+count>(uint64_t)usedlinearsize ||
      total+count+2*(uint64_t)outs>(uint64_t)loaded.linearsize)
    goto fail;
   bann_setweight(&loaded,i,ins,outs,loaded.linear+(size_t)used,flags,0);
   used+=count;
   total+=count+2*(uint64_t)outs;
  }
 if(used!=(uint64_t)usedlinearsize || total!=(uint64_t)loaded.linearsize)
  goto fail;
 for(i=0;i<layercount;i++)
  {
   bann_setweighttemp(&loaded,i,loaded.linear+used);
   used+=2*(uint64_t)loaded.weights[i].outs;
  }
#if defined(BANN_RNN)
 loaded.rweights=loaded.weights+loaded.weightscount;
 memset(loaded.rweights->o,0,loaded.rweights->outs*sizeof(real));
#endif
 bann_buildsigmoidcache();
 *net=loaded;
 return 1;

fail:
 bann_delete(&loaded);
 return 0;
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
