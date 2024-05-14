#include <stdio.h>
#include <stdlib.h>

#define BANN_IMPLEMENTATION
#include "bann.h"

#define BANNQ_IMPLEMENTATION
#include "bannQ.h"

 real input[4][2] = {{0, 0}, 
                           {0, 1}, 
                           {1, 0}, 
                           {1, 1}};
 real output[4] = {0, 1, 1, 0};

int main()
{
 bann     N; 
 bannQ    NQ; 
 int      xor_maxepoch=512;
 int      e=0,v,cnt=0;
 real     err=0;
 bann_io  in,out;

 bann_new(&N,2,2,1,1);

 in.kind=standard;
 out.kind=standard;

 for(e=0;e<xor_maxepoch;e++)
  for(v=0;v<4;v++)
   {

    in.linear=&input[v][0];
    out.linear=&output[v];
    err+=bann_train(&N,&in,&out);
    
    cnt++;
    
    printf("e:%d v:%d err: %.2f \r",e,v,err/cnt);

   }
 printf("\n");
 for(v=0;v<4;v++)
  {
   in.linear=&input[v][0];
   bann_predict(&N,&in);   
   printf("value %d: %1.f / %1.f (%.2f)\n",v+1,output[v],N.weights[1].o[0],N.weights[1].o[0]);
  }
 bann_write(&N,"bin/xor.bann",bann_float);
 bann_write(&N,"bin/xor.bannQ",bann_8bit); 
 bann_delete(&N); 

 if(bann_read(&N,"bin/xor.bann"))
  {
   for(v=0;v<4;v++)
    {
     in.linear=&input[v][0];
     bann_predict(&N,&in);   
     printf("value %d: %1.f / %1.f (%.2f)\n",v+1,output[v],N.weights[1].o[0],N.weights[1].o[0]);
    }
   bann_delete(&N); 
  }

 if(bannQ_read(&NQ,"bin/xor.bannQ"))
  {
   for(v=0;v<4;v++)
    {
     in.linear=&input[v][0];
     bannQ_predict(&NQ,input[v]);   
     printf("value %d: %1.f / %1.f (%.2f)\n",v+1,output[v],NQ.weights[1].o[0],NQ.weights[1].o[0]);
    }
   bannQ_delete(&NQ); 
  }

 return 1;
}
