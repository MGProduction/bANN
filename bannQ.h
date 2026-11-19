//
// bANNQ - v0.01 - bANN quantized neural network single-file header-only implementation
//                without any dependency apart from standard libs
//
//                by Marco Giorgini (@marcogiorgini) 2024 -
//
//                Features:
//                - this library uses the quantized version of the neural netwrok
//                  trainged and saved by bANN
//
//   COMPILING & LINKING
//      In one C/C++ file that #includes this file, do this:
//         #define BANNQ_IMPLEMENTATION
//      before the #include. That will create the implementation in that file.
//
// LICENSE
//
//   See end of file.
//

#ifndef _H_BANNQ_H
#define _H_BANNQ_H

#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------------------

typedef float         real;
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

 real base,scale; 
 byte*bias;       
 byte*w;          // weight values (quantized)

 real*o;          // output values

 bact activation; // activation function
 int  flags;
}weightQ;

// nnet struct
typedef struct{
 weightQ*weights;
 weightQ*rweights;
 int     weightscount;
 int     linearsize;
 real   *linear;
 byte   *blinear;
}bannQ;

int  bannQ_read(bannQ*net,const char*file);
real*bannQ_predict(bannQ*net,bann_io*input);
void bannQ_delete(bannQ*net);
void bannQ_reset(bannQ*net);

// ---------------------------------------------------------------------

#if defined(BANNQ_IMPLEMENTATION)

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>

#define bannQ_activation_sigmoid   0
#define bannQ_activation_linear    1
#define bannQ_activation_relu      2
#define bannQ_activation_tanh      3
#define bannQ_activation_softmax   4

#define bannQ_8bit                 1

// INTERNAL - sigmoid cache code
#define _Qsigmoid_dom_min     -32
#define _Qsigmoid_dom_max      32
#define _Qsigmoid_table_size 2048
real _Qsigmoid_interval;
real _Qsigmoid[_Qsigmoid_table_size];
real bannQ_tanh(real a)
{
 return tanhf(a);
}
real bannQ_linear(real f)
{
 return f;
}
real bannQ_relu(real f)
{
 if(f>0) return f;
 else    return 0;
}
real bannQ_sigmoid(real f)
{
 if (f < _Qsigmoid_dom_min) 
  return _Qsigmoid[0];
 else
 if (f >= _Qsigmoid_dom_max) 
  return _Qsigmoid[_Qsigmoid_table_size-1];
 else
  {
   int j=(int)((f-_Qsigmoid_dom_min)*_Qsigmoid_interval+0.5f);  
   if(j>=_Qsigmoid_table_size) 
    return _Qsigmoid[_Qsigmoid_table_size-1];
   else
    return _Qsigmoid[j];
  }
}
void bannQ_buildsigmoidcache()
{
 if(_Qsigmoid_interval==0)
  {
   int  i;
   real f;
   _Qsigmoid_interval=(_Qsigmoid_table_size/(_Qsigmoid_dom_max-_Qsigmoid_dom_min));
   f=((_Qsigmoid_dom_max-_Qsigmoid_dom_min)/(real)_Qsigmoid_table_size);
   for (i = 0; i < _Qsigmoid_table_size; ++i) 
    {
     real a=_Qsigmoid_dom_min+f*i,b=a;
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
     _Qsigmoid[i]=a;   
    }
  }
}

void bannQ_enableweightactivation(bannQ*net,int layernumber)
{ 
 switch(net->weights[layernumber].flags&0x7)
  {
  case bannQ_activation_sigmoid:
   net->weights[layernumber].activation=bannQ_sigmoid;
   break;
  case bannQ_activation_linear:
  case bannQ_activation_softmax:
   net->weights[layernumber].activation=bannQ_linear;
   break;
  case bannQ_activation_relu:
   net->weights[layernumber].activation=bannQ_relu;
   break;
  case bannQ_activation_tanh:
   net->weights[layernumber].activation=bannQ_tanh;
   break;
  }
}

// INTERNAL - set weight info - using linear pointer
byte*bannQ_setweight(bannQ*net,int layernumber,int ins,int outs,byte*ptr,real base,real scale,int actmethod)
{
 weightQ*w=&net->weights[layernumber];
 w->base=base;w->scale=scale;
 w->ins=ins; 
 w->outs=outs;
 
 w->bias=ptr;
 ptr+=outs;

 w->w=ptr;
 ptr+=ins*outs;

 w->flags=actmethod;
 bannQ_enableweightactivation(net,layernumber);
 return ptr;
}

int bannQ_coreread(bannQ*net,FILE*f)
{
 bannQ loaded={0};
 int version,way,storedsize,usedlinearsize,layercount,i;
 uint64_t used=0,outputs=0;
 real base,scale;
 char magic[4];

 if(fread(magic,4,1,f)!=1 || memcmp(magic,"BANN",4)!=0 ||
    fread(&version,sizeof(version),1,f)!=1 || (version!=1 && version!=2) ||
    fread(&way,sizeof(way),1,f)!=1 || way!=bannQ_8bit ||
    fread(&storedsize,sizeof(storedsize),1,f)!=1 ||
    fread(&usedlinearsize,sizeof(usedlinearsize),1,f)!=1 ||
    usedlinearsize<=0 || storedsize<=usedlinearsize ||
    fread(&base,sizeof(base),1,f)!=1 || !isfinite(base) ||
    fread(&scale,sizeof(scale),1,f)!=1 || !isfinite(scale) || scale<0)
  return 0;

 loaded.blinear=(byte*)malloc((size_t)usedlinearsize);
 if(!loaded.blinear ||
    fread(loaded.blinear,1,usedlinearsize,f)!=(size_t)usedlinearsize ||
    fread(&layercount,sizeof(layercount),1,f)!=1 ||
    layercount<version+1 || layercount>usedlinearsize ||
    (size_t)layercount>SIZE_MAX/sizeof(weightQ))
  goto fail;
 loaded.weightscount=layercount-(version==2);
 loaded.weights=(weightQ*)malloc((size_t)layercount*sizeof(weightQ));
 if(!loaded.weights)
  goto fail;

 for(i=0;i<layercount;i++)
  {
   int ins,outs,flags;
   uint64_t count;
   if(fread(&ins,sizeof(ins),1,f)!=1 || fread(&outs,sizeof(outs),1,f)!=1 ||
      fread(&flags,sizeof(flags),1,f)!=1 || ins<=0 || outs<=0 ||
      (flags&7)>bannQ_activation_softmax)
    goto fail;
   if(i && i<loaded.weightscount && loaded.weights[i-1].outs!=ins)
    goto fail;
   if(i==loaded.weightscount && (ins!=loaded.weights[0].outs || outs!=ins))
    goto fail;
   count=((uint64_t)ins+1)*outs;
   if(used+count>(uint64_t)usedlinearsize ||
      used+count+2*(outputs+outs)>(uint64_t)storedsize)
    goto fail;
   bannQ_setweight(&loaded,i,ins,outs,loaded.blinear+used,base,scale,flags);
   used+=count;
   outputs+=outs;
  }
 if(used!=(uint64_t)usedlinearsize || used+2*outputs!=(uint64_t)storedsize ||
    outputs>SIZE_MAX/sizeof(real))
  goto fail;

 loaded.linearsize=(int)outputs;
 loaded.linear=(real*)calloc((size_t)outputs,sizeof(real));
 if(!loaded.linear)
  goto fail;
 outputs=0;
 for(i=0;i<layercount;i++)
  {
   loaded.weights[i].o=loaded.linear+outputs;
   outputs+=loaded.weights[i].outs;
  }
 if(version==2)
  loaded.rweights=&loaded.weights[loaded.weightscount];

 bannQ_buildsigmoidcache();
 *net=loaded;
 return 1;

fail:
 bannQ_delete(&loaded);
 return 0;
}
int  bannQ_read(bannQ*net,const char*file)
{
 FILE*f=fopen(file,"rb");
 if(f)
  {
   int ret=bannQ_coreread(net,f);
   fclose(f);
   return ret;
  }
 else
  return 0;
}

// Decode one row at a time, keeping only the activations in float.
real*bannQ_predict(bannQ*net,bann_io*input)
{
 int i,j,l;
 for(l=0;l<net->weightscount;l++)
  {
   weightQ*w=&net->weights[l];
   for(i=0;i<w->outs;i++)
    {
     const byte*row=w->w+(size_t)i*w->ins;
     real sum=-(w->base+w->bias[i]*w->scale);
     if(l==0)
      {
       switch(input->kind)
        {
         case iokind_standard:
          for(j=0;j<w->ins;j++)
           sum+=input->linear[j]*(w->base+row[j]*w->scale);
          break;
         case iokind_idx:
          for(j=0;j<input->idx->hotcnt;j++)
           sum+=w->base+row[input->idx->hotpos[j]]*w->scale;
          break;
         case iokind_fidx:
          for(j=0;j<input->fidx->hotcnt;j++)
           sum+=input->fidx->hotval[j]*(w->base+row[input->fidx->hotpos[j]]*w->scale);
          break;
        }
       if(net->rweights)
        {
         weightQ*recurrent=net->rweights;
         const byte*recurrent_row=recurrent->w+(size_t)i*recurrent->ins;
         for(j=0;j<recurrent->ins;j++)
          sum+=recurrent->o[j]*(recurrent->base+recurrent_row[j]*recurrent->scale);
         sum-=recurrent->base+recurrent->bias[i]*recurrent->scale;
        }
      }
     else
      {
       for(j=0;j<w->ins;j++)
        sum+=net->weights[l-1].o[j]*(w->base+row[j]*w->scale);
      }
     w->o[i]=w->activation(sum);
    }
   if((w->flags&7)==bannQ_activation_softmax)
    {
     real max=w->o[0],sum=0;
     for(i=1;i<w->outs;i++)
      if(w->o[i]>max)
       max=w->o[i];
     for(i=0;i<w->outs;i++)
      {
       w->o[i]=expf(w->o[i]-max);
       sum+=w->o[i];
      }
     for(i=0;i<w->outs;i++)
      w->o[i]/=sum;
    }
  }
 if(net->rweights)
  memcpy(net->rweights->o,net->weights[0].o,net->rweights->outs*sizeof(real));
 return net->weights[l-1].o;
}

void bannQ_reset(bannQ*net)
{
 if(net->rweights)
  memset(net->rweights->o,0,net->rweights->outs*sizeof(real));
}
// nnet delete
void bannQ_delete(bannQ*net)
{
 free(net->weights);
 free(net->linear);
 free(net->blinear);
}

#endif

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