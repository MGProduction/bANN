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
real*bannQ_predict(bannQ*net,const real*input);
void bannQ_delete(bannQ*net);

// ---------------------------------------------------------------------

#if defined(BANNQ_IMPLEMENTATION)

// bann read
int  bannQ_coreread(bannQ*net,FILE*f)
{
 int  version,i,ret=0,way;
 char magic[4];
 if((fread(magic,4,1,f)==1)&&(fread(&version,sizeof(version),1,f)==1)&&(memcmp(magic,"BANN",4)==0)&&(version==1))
  {
   real*ptr,base,scale;
   byte*blinear;
   int  i;

   memset(net,0,sizeof(bann));

   fread(&way,sizeof(way),1,f);
   if(way!=bann_8bit)
    return 0;

   fread(&net->linearsize,sizeof(net->linearsize),1,f);
   blinear=(byte*)malloc(net->linearsize);
   fread(&base,sizeof(base),1,f);fwrite(&scale,sizeof(scale),1,f);
   fread(blinear,sizeof(blinear[0]),net->linearsize,f);
   for(i=0;i<net->linearsize;i++)
    net->linear[i]=base+blinear[i]*scale;
   free(blinear);
   

   ptr=net->linear=(real*)malloc( net->linearsize * sizeof(net->linear[0]) );

   fread(net->linear,sizeof(net->linear[0]),net->linearsize,f);     

   // read net weight structure and fill it
   fread(&net->weightscount,sizeof(net->weightscount),1,f);          
   net->weights=(weightQ*)malloc( net->weightscount * sizeof(net->weights[0]) );
   for(i=0;i<net->weightscount;i++)
    {
     int ins,outs,flags;
     fread(&ins,sizeof(ins),1,f);
     fread(&outs,sizeof(outs),1,f);
     fread(&flags,sizeof(flags),1,f);
     ptr=bannQ_setweight(net,i,ins,outs,ptr,flags,0);
    }
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
real*bannQ_predict(bannQ*net,const real*input)
{
 int  i,j,l;
 // from input layer to output one
 // v values are written on the next lever
 for(l=0;l<net->weightscount;l++)
  for(i=0;i<net->weights[l].outs;i++)
   {
    real sum=0.0f;
    for(j=0;j<net->weights[l].ins;j++)
     if(l==0)      
      // use input instead of previous hidden state
      sum+=input[j]*nn_getweight(net,l,j,i);       
     else
      sum+=net->weights[l-1].o[j]*nn_getweight(net,l,j,i);
    nn_setoutput(net,l,i,net->weights[l].activation(sum+net->weights[l].bias[l]));
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