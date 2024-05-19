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
 int     weightscount;
 int     linearsize;
 real   *linear;
 byte   *blinear;
}bannQ;

int  bannQ_read(bannQ*net,const char*file);
real*bannQ_predict(bannQ*net,bann_io*input);
void bannQ_delete(bannQ*net);

// ---------------------------------------------------------------------

#if defined(BANNQ_IMPLEMENTATION)

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>

#define bannQ_activation_sigmoid   0
#define bannQ_activation_linear    1
#define bannQ_activation_relu      2
#define bannQ_activation_tanh      3

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

real*bannQ_setweighttemp(bannQ*net,int layernumber,real*ptr)
{
 weightQ*w=&net->weights[layernumber];
 w->o=ptr;
 ptr+=w->outs;
 ptr+=w->ins;  
 return ptr;
}

// bann read
int  bannQ_coreread(bannQ*net,FILE*f)
{
 int  version,ret=0,way;
 char magic[4];
 if((fread(magic,4,1,f)==1)&&(fread(&version,sizeof(version),1,f)==1)&&(memcmp(magic,"BANN",4)==0)&&(version==1))
  {
   real*ptr,base,scale;
   byte*bptr;
   int  i,usedlinearsize;

   memset(net,0,sizeof(bannQ));

   fread(&way,sizeof(way),1,f);
   if(way!=bannQ_8bit)
    return 0;

   fread(&net->linearsize,sizeof(net->linearsize),1,f);
   fread(&usedlinearsize,sizeof(usedlinearsize),1,f);

   ptr=net->linear=(real*)malloc( (net->linearsize-usedlinearsize) * sizeof(net->linear[0]) );
   bptr=net->blinear=(byte*)malloc( usedlinearsize * sizeof(net->blinear[0]) );

   fread(&base,sizeof(base),1,f);fread(&scale,sizeof(scale),1,f);
   fread(net->blinear,sizeof(net->blinear[0]),usedlinearsize,f);

   // read net weight structure and fill it
   fread(&net->weightscount,sizeof(net->weightscount),1,f);          
   net->weights=(weightQ*)malloc( net->weightscount * sizeof(net->weights[0]) );
   for(i=0;i<net->weightscount;i++)
    {
     int ins,outs,flags;
     fread(&ins,sizeof(ins),1,f);
     fread(&outs,sizeof(outs),1,f);
     fread(&flags,sizeof(flags),1,f);
     bptr=bannQ_setweight(net,i,ins,outs,bptr,base,scale,flags);
    }
   for(i=0;i<net->weightscount;i++)
    ptr=bannQ_setweighttemp(net,i,ptr);
   // build sigmoid table
   bannQ_buildsigmoidcache();
   ret=1;
  }
 return ret;
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

// bann execution
#define nnQ_getweight(net,l,r,c) (net->weights[l].base+(net->weights[l].w[c*net->weights[l].ins+r]*net->weights[l].scale))
#define nnQ_setoutput(net,l,c,v) net->weights[l].o[c]=v
real*bannQ_predict(bannQ*net,bann_io*input)
{
 int  i,j,l;
 // from input layer to output one
 // v values are written on the next lever
 for(l=0;l<net->weightscount;l++)
  {
   real bias=net->weights[l].base+(net->weights[l].bias[0]*net->weights[l].scale);
   for(i=0;i<net->weights[l].outs;i++)
    {
     real sum=0.0f;
     if(l==0)      
      {
       // use input instead of previous hidden state
       switch(input->kind)
        {
         case standard:
          for(j=0;j<net->weights[l].ins;j++)             
           sum+=input->linear[j]*nnQ_getweight(net,l,j,i);       
         break;
         case idx:
          for(j=0;j<input->idx->hotcnt;j++)        
           sum+=nnQ_getweight(net,l,input->idx->hotpos[j],i);       
          break;
         case fidx:
          for(j=0;j<input->fidx->hotcnt;j++)        
           sum+=input->fidx->hotval[j]*nnQ_getweight(net,l,input->fidx->hotpos[j],i);       
          break;
        }
      }
     else
      {
       for(j=0;j<net->weights[l].ins;j++)
        sum+=net->weights[l-1].o[j]*nnQ_getweight(net,l,j,i);
      }
     nnQ_setoutput(net,l,i,net->weights[l].activation(sum+bias));
    }
  }
  return net->weights[l-1].o;
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