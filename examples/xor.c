#include <stdio.h>
#include <stdlib.h>

#define BANN_IMPLEMENTATION
#include "bann.h"

const real input[4][2] = {{0, 0}, 
                           {0, 1}, 
                           {1, 0}, 
                           {1, 1}};
const real output[4] = {0, 1, 1, 0};

int main()
{
 bann N; 
 int      xor_maxepoch=256;
 int      e=0,v,cnt=0;
 real     err=0;

 bann_new(&N,2,2,1,1);
 bann_setweightactivation(&N,0,bann_activation_tanh);

 for(e=0;e<xor_maxepoch;e++)
  for(v=0;v<4;v++)
   {
    real dif;

    bann_train(&N,&input[v][0],&output[v]);
    
    dif=0.5f*(N.weights[1].o[0]-output[v])*(N.weights[1].o[0]-output[v]);
    err+=dif; 
    cnt++;
    
    printf("e:%d v:%d err: %.2f \r",e,v,err/cnt);

   }
 printf("\n");

 for(v=0;v<4;v++)
  {
   bann_predict(&N,input[v]);   
   printf("value %d: %1.f / %1.f\n",v+1,N.weights[1].o[0],output[v]);
  }
 bann_write(&N,"bin/xor.bann",bann_float);
 bann_write(&N,"bin/xor.bannQ",bann_8bit);
 bann_delete(&N); 

 return 1;
}
