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
//                - cached) signmoid activation / reLU
//                - single memory allocation for weights
//                - basic autooptimization of learning rate
//                - write / save (normal&quantized)
//                - predict normal&quantized          (*** not yet available ***)
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
typedef unsigned char byte;

enum {
 standard=0,
 idx=1,
 fidx=2,
}bann_iokind;

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

// ---------------------------------------------------------------------

typedef real (*bact)(real f);

#if defined(BANN_USEIDX)
typedef struct{
 int*b;
 int c,m;
}bannidx;
#endif

// matrix between layers
typedef struct{
 int  ins,outs; 
 real*bias;       
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
 int    errcnt;
}bopt;

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

#define bann_autooptimize_off          0
#define bann_autooptimize_auto         1
#define bann_autooptimize_progressive  2

void bann_setlearningrate(bann*net,real learningrate,int optimizemode);

real*bann_predict(bann*net,const bann_io*input);
real bann_train(bann*net,const bann_io*input,const bann_io*output);
#if defined(BANN_USEIDX)
real*bann_predictidx(bann*net,const bannidx*input);
real bann_trainidx(bann*net,const bannidx*input,const bannidx*output);
#endif
#if defined(BANN_RNN)
void bann_startrnn(bann*net);
#endif

int  bann_write(bann*net,const char*file,int way);
int  bann_read(bann*net,const char*file);

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
  return 0;
 else
  return 1;
}
real invalid()
{
 printf("error!\n");
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
 if((l<0)||(l>=net->weightscount)||(r<0)||(r>=net->weights[l].ins))
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
 if((l<0)||(l>=net->weightscount)||(r<0)||(r>=net->weights[l].ins))
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
 return expf(a);
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
    net->weights[layernumber].prime=bann_sigmoid_prime;
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
 if(fillwithrandom)
  w->bias[0]=bann_rand();
 ptr++;
 w->w=ptr;
 if(fillwithrandom)
  for(i=0;i<ins*outs;i++)
   w->w[i]=bann_rand();
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
 ptr+=w->ins;  
 return ptr;
}

#if defined(BANN_RNN)
void bann_startrnn(bann*net)
{
 memset(net->weights[net->weightscount].o,0,net->weights[net->weightscount].outs*sizeof(net->weights[net->weightscount].o[0]));
}
#endif

// create the bANN structure
void bann_new(bann*net,int input,int hidden,int hiddencount,int output)
{
 int  l=0,h;
 real*ptr;
 memset(net,0,sizeof(bann));
 // allocate all the weights
 net->linearsize=((input*hidden+1+hidden+input) + (hidden*output+1+output+hidden) + (hidden*hidden+1+hidden*2)*(hiddencount-1) ); 
#if defined(BANN_RNN)
 net->linearsize+=(input*hidden+1+hidden+input);
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
 ptr=bann_setweight(net,l,input,hidden,ptr,bann_activation_sigmoid,1); 
 ptr=bann_setweighttemp(net,l,ptr);
#endif
 assert((ptr-net->linear)==net->linearsize);
 // set learning rate in auto adjust mode
 bann_setlearningrate(net,1.0f,bann_autooptimize_auto);
 // build sigmoid table
 bann_buildsigmoidcache(); 
}

// bann execution
real*bann_predict(bann*net,const bann_io*input)
{
 int  i,j,l;
 // from input layer to output one
 // v values are written on the next lever
 for(l=0;l<net->weightscount;l++)
  {   
   for(i=0;i<net->weights[l].outs;i++)
    {
     real sum=net->weights[l].bias[0];
     if(l==0)
      {
       // use input instead of previous hidden state
       switch(input->kind)
        {
         case standard:
          for(j=0;j<net->weights[l].ins;j++)                     
           sum+=input->linear[j]*nn_getweight(net,l,j,i);       
          break;
         case idx:
          for(j=0;j<input->idx->hotcnt;j++)        
           sum+=nn_getweight(net,l,input->idx->hotpos[j],i);       
          break;
         case fidx:
          for(j=0;j<input->fidx->hotcnt;j++)        
           sum+=input->fidx->hotval[j]*nn_getweight(net,l,input->fidx->hotpos[j],i);       
          break;
        }
      }
     else
      {
       for(j=0;j<net->weights[l].ins;j++)
        sum+=net->weights[l-1].o[j]*nn_getweight(net,l,j,i);
      }
     nn_setoutput(net,l,i,net->weights[l].activation(sum));     
    }
   if((net->weights[l].flags&7)==bann_activation_softmax)
    {
     real sum=0;
     for(i=0;i<net->weights[l].outs;i++)
      sum+=net->weights[l].o[i];
     for(i=0;i<net->weights[l].outs;i++)
      net->weights[l].o[i]=net->weights[l].o[i]/sum;
    }
  }
  return net->weights[l-1].o;
}
#if defined(BANN_USEIDX)
real*bann_predictidx(bann*net,const bannidx*input)
{
 int  i,j,l;
 // from input layer to output one
 // v values are written on the next lever
#if defined(BANN_RNN)
 int rnn=net->weightscount;
 for(j=0;j<net->weights[rnn].ins;j++)
   net->weights[rnn].o[j]=net->weights[0].o[j];
#endif
 for(l=0;l<net->weightscount;l++)
  {   
   for(i=0;i<net->weights[l].outs;i++)
    {
     real sum=net->weights[l].bias[0];     
     // use input instead of previous hidden state
     if(l==0)      
      {       
       for(j=0;j<input->c;j++)        
        sum+=nn_getweight(net,l,input->b[j],i);       
#if defined(BANN_RNN)
       sum+=net->weights[rnn].bias[0];
       for(j=0;j<net->weights[l].ins;j++)
        sum+=net->weights[rnn].o[j]*nn_getweight(net,rnn,j,i);
#endif
      }
     else
      for(j=0;j<net->weights[l].ins;j++)
       sum+=net->weights[l-1].o[j]*nn_getweight(net,l,j,i);
     nn_setoutput(net,l,i,net->weights[l].activation(sum));     
    }
   if((net->weights[l].flags&7)==bann_activation_softmax)
    {
     real sum=0;
     for(i=0;i<net->weights[l].outs;i++)
      sum+=net->weights[l].o[i];
     for(i=0;i<net->weights[l].outs;i++)
      net->weights[l].o[i]=net->weights[l].o[i]/sum;
    }
  }
  return net->weights[l-1].o;
}
#endif
void bann_setlearningrate(bann*net,real learningrate,int optimizemode)
{
 net->opt.learningrate=learningrate;
 net->opt.optimize=optimizemode;
}

void bann_optimize(bann*net,real err)
{
 switch(net->opt.optimize)
 {
 case bann_autooptimize_auto:
  {
   int  l=net->weightscount-1;
   net->opt.errsum+=err;net->opt.errcnt++;
   if(net->opt.errcnt>64)
    {
     real expand=1.0f;
     net->opt.errsum/=net->opt.errcnt;
     if(net->opt.preverrsum)
      if(net->opt.errsum>net->opt.preverrsum)
       expand=(net->opt.preverrsum/net->opt.errsum);
      else
      if(net->opt.errsum<net->opt.preverrsum) 
       expand=(net->opt.preverrsum/net->opt.errsum);
     if(expand<0.75f) 
      expand=0.75f;
     else
      if(expand>1.25f)
       expand=1.25f;
     net->opt.learningrate*=expand;
     if(net->opt.learningrate<0.1f)
      net->opt.learningrate=0.1f;
     else
     if(net->opt.learningrate>1.0f)
      net->opt.learningrate=1.0f;
     net->opt.preverrsum=net->opt.errsum;
     net->opt.errsum=0;
     net->opt.errcnt=0;
    }
  }
 break;
 case bann_autooptimize_progressive:
  net->opt.errcnt++;
  if(net->opt.errcnt>16*1024)
   {
    net->opt.learningrate*=0.95f;
    if(net->opt.learningrate<0.1f)
     net->opt.learningrate=0.1f;
    net->opt.errcnt=0;
   }
  break;
 }
}

#if defined(BANN_PROTECTEXPLODING)
#define bann_protectbias(a) {if(a<-10) a=-10; else if(a>10) a=10;}
#define bann_protect(a)     {if(a<-15) a=-15; else if(a>15) a=15;}
#else
#define bann_protectbias(a) 
#define bann_protect(a)     
#endif
// bann training
real bann_train(bann*net,const bann_io*input,const bann_io*output)
{
 int  c,r,l=net->weightscount-1;
 real learningrate=net->opt.learningrate,err=0;
 // forward
 bann_predict(net,input);
// backpropagation
 // output to hidden
 for(r=0;r<net->weights[l].ins;r++)
  {
   real delta=0.0f;
   int  h=0;
   for(c=0;c<net->weights[l].outs;c++)
    {
     real error,derror;

     switch(output->kind)
     {
     case standard:
      error=(net->weights[l].o[c]-output->linear[c]);
      break;
      case idx:
       if((h<output->idx->hotcnt)&&(c==output->idx->hotpos[h]))
        {
         error=(net->weights[l].o[c]-1);
         h++;
        }
       else
        error=(net->weights[l].o[c]-0);
       break;
      case fidx:
       if((h<output->fidx->hotcnt)&&(c==output->fidx->hotpos[h]))
        {
         error=(net->weights[l].o[c]-output->fidx->hotval[h]);
         h++;
        }
       else
        error=(net->weights[l].o[c]-0);
       break;
     }
     
     derror=error * net->weights[l].prime(net->weights[l].o[c]);

     err+=0.5f*error*error;
     
     delta += derror * nn_getweight(net,l,r,c);
     
     net->weights[l].w[c*net->weights[l].ins+r] -= learningrate * derror * net->weights[l-1].o[r];           
     bann_protect(net->weights[l].w[c*net->weights[l].ins+r])

     sanitycheck(nn_getweight(net,l,r,c));
    }
   nn_setdelta(net,l,r,delta);   
   sanitycheck(net->weights[l].d[r]);   
  }
 net->weights[l].bias[0] -= learningrate * nn_getdelta(net,l,0);
 bann_protectbias(net->weights[l].bias[0])
 // hidden till input 
 while(l--)
  {   
   for(c=0;c<net->weights[l].outs;c++)
    {
     real delta=learningrate * nn_getdelta(net,l+1,c) * net->weights[l].prime(nn_getoutput(net,l,c)),ndelta=0;     
     if(l==0)
      {
       switch(input->kind)
        {
         case standard:
          for(r=0;r<net->weights[l].ins;r++)
           {      
            ndelta += nn_getdelta(net,l+1,c) * nn_getweight(net,l,r,c);
            net->weights[l].w[c*net->weights[l].ins+r] -= delta * input->linear[r];
            bann_protect(net->weights[l].w[c*net->weights[l].ins+r])
           }
          break;
          case idx:
           for(r=0;r<input->idx->hotcnt;r++)
            {      
             int rr=input->idx->hotpos[r];
             ndelta += nn_getdelta(net,l+1,c) * nn_getweight(net,l,rr,c);
             net->weights[l].w[c*net->weights[l].ins+rr] -= delta;
             bann_protect(net->weights[l].w[c*net->weights[l].ins+rr])
            }
          break;
          case fidx:
           for(r=0;r<input->fidx->hotcnt;r++)
            {      
             int rr=input->fidx->hotpos[r];
             ndelta += nn_getdelta(net,l+1,c) * nn_getweight(net,l,rr,c);
             net->weights[l].w[c*net->weights[l].ins+rr] -= delta*input->fidx->hotval[r];
             bann_protect(net->weights[l].w[c*net->weights[l].ins+rr])
            }
          break;
        }
      }
     else
      {
       for(r=0;r<net->weights[l].ins;r++)
        {      
         ndelta += nn_getdelta(net,l+1,c) * nn_getweight(net,l,r,c);
         net->weights[l].w[c*net->weights[l].ins+r] -= delta * net->weights[l-1].o[r];
         bann_protect(net->weights[l].w[c*net->weights[l].ins+r])
        }
      }
     nn_setdelta(net,l,c,ndelta);
    }   
   net->weights[l].bias[0] -= learningrate * nn_getdelta(net,l,0);
   bann_protectbias(net->weights[l].bias[0])
  }
 if(net->opt.optimize)
  bann_optimize(net,err);
 return err;
}
#if defined(BANN_USEIDX)
real bann_trainidx(bann*net,const bannidx*input,const bannidx*output)
{
 int  c,r,l=net->weightscount-1,b=0;
#if defined(BANN_RNN)
 int  rnn=net->weightscount;
#endif
 real learningrate=net->opt.learningrate,err=0;
 // forward
 bann_predictidx(net,input);
// backpropagation
 // output to hidden
 for(r=0;r<net->weights[l].ins;r++)
  {
   real delta=0.0f;
   for(c=0;c<net->weights[l].outs;c++)
    {
     real error,derror;
     if((b<output->c)&&(c==output->b[b]))
      {error=(net->weights[l].o[c]-1);b++;}
     else
      error=(net->weights[l].o[c]-0);

     err+=0.5f*error;

     derror=error * net->weights[l].prime(net->weights[l].o[c]);
     
     delta += derror * nn_getweight(net,l,r,c);
     
     net->weights[l].w[c*net->weights[l].ins+r] -= learningrate * derror * net->weights[l-1].o[r];           
     bann_protect(net->weights[l].w[c*net->weights[l].ins+r])

     sanitycheck(nn_getweight(net,l,r,c));
    }
   nn_setdelta(net,l,r,delta);   
   sanitycheck(net->weights[l].d[r]);

   net->weights[l].bias[0] -= learningrate * nn_getdelta(net,l,c);
   bann_protectbias(net->weights[l].bias[0])
  }
 // hidden till input 
 while(l--)
  {   
   for(c=0;c<net->weights[l].outs;c++)
    {
     real delta=learningrate * nn_getdelta(net,l+1,c) * net->weights[l].prime(nn_getoutput(net,l,c)),ndelta=0;     
     if(l==0)
      {
       for(r=0;r<input->c;r++)
        {
         net->weights[l].w[c*net->weights[l].ins+input->b[r]] -= delta;
         bann_protect(net->weights[l].w[c*net->weights[l].ins+input->b[r]])
        }
#if defined(BANN_RNN)
       for(r=0;r<input->c;r++)
        {
         net->weights[rnn].w[c*net->weights[rnn].ins+input->b[r]] -= delta;
         bann_protect(net->weights[l].w[c*net->weights[l].ins+input->b[r]])
        }
#endif
      }
     else
      {
       for(r=0;r<net->weights[l].ins;r++)
        {
        ndelta += nn_getdelta(net,l+1,c) * nn_getweight(net,l,r,c);
        net->weights[l].w[c*net->weights[l].ins+r] -= delta * net->weights[l-1].o[r];
        bann_protect(net->weights[l].w[c*net->weights[l].ins+r])
        }
      }     

     nn_setdelta(net,l,c,ndelta);

     net->weights[l].bias[0] -= learningrate * nn_getdelta(net,l,c);
#if defined(BANN_RNN)
     net->weights[rnn].bias[0] -= learningrate * nn_getdelta(net,l,c);
#endif
     bann_protectbias(net->weights[l].bias[0])
    }   
  }
 if(net->opt.optimize)
  bann_optimize(net,err);

 return err;
}
#endif
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
  usedlinearsize+=net->weights[i].ins*net->weights[i].outs+1;
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