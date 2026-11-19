#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define BANN_IMPLEMENTATION
#include "bann.h"
#define BANNQ_IMPLEMENTATION
#include "bannQ.h"

enum { COOKING, WEATHER, SOFTWARE, CLASSES };
enum { FEATURES=512, HIDDEN=24, EPOCHS=100 };

typedef struct {
 const char *text;
 int label;
} sample;

static const char *labels[]={"cooking","weather","software"};
static const sample training[]={
 {"Bake the cake in a preheated oven",COOKING},
 {"Chop the onions and add them to the soup",COOKING},
 {"Stir the sauce over low heat",COOKING},
 {"Mix the flour and butter into a dough",COOKING},
 {"Roast the vegetables with olive oil",COOKING},
 {"Boil the pasta and drain the water",COOKING},
 {"Season the chicken with salt and pepper",COOKING},
 {"Fry the eggs in a pan with butter",COOKING},
 {"Bake the bread until the crust is golden",COOKING},
 {"Simmer the tomato sauce with garlic",COOKING},
 {"Heavy rain is expected this afternoon",WEATHER},
 {"The forecast calls for strong winds",WEATHER},
 {"Snow will fall overnight in the mountains",WEATHER},
 {"Clouds are moving in from the coast",WEATHER},
 {"Temperatures will drop below freezing",WEATHER},
 {"Thunderstorms may arrive before dawn",WEATHER},
 {"Sunny skies and warm weather tomorrow",WEATHER},
 {"A cold front will bring rain on Friday",WEATHER},
 {"Wind and snow are expected by evening",WEATHER},
 {"The rain should clear by the weekend",WEATHER},
 {"Compile the code and run the tests",SOFTWARE},
 {"Fix the error in the compiler output",SOFTWARE},
 {"Commit the patch to the git repository",SOFTWARE},
 {"Install the package before building",SOFTWARE},
 {"Run the unit tests after the update",SOFTWARE},
 {"Debug the crash in the release build",SOFTWARE},
 {"Check the build logs for warnings",SOFTWARE},
 {"The test suite found a new bug",SOFTWARE},
 {"Update the code and rebuild the binary",SOFTWARE},
 {"The build failed with a linker error",SOFTWARE},
};
static const sample test[]={
 {"Roast the chicken in the oven with salt",COOKING},
 {"Stir the butter into the dough",COOKING},
 {"Add garlic to the tomato soup",COOKING},
 {"The evening forecast calls for rain",WEATHER},
 {"Snow and strong winds are expected overnight",WEATHER},
 {"Warm temperatures and sunny skies tomorrow",WEATHER},
 {"Rebuild the package and run the test suite",SOFTWARE},
 {"The compiler found an error in the patch",SOFTWARE},
 {"Commit the fix after the tests pass",SOFTWARE},
};

static unsigned char lower_ascii(unsigned char c)
{
 return c>='A' && c<='Z' ? (unsigned char)(c-'A'+'a') : c;
}

/* Hash overlapping trigrams and combine counts for bins that collide. */
static void encode(const char *text,bann_fidx *input)
{
 input->hotcnt=0;
 for(;text[0] && text[1] && text[2];text++)
  {
   uint32_t hash=2166136261u;
   int i,bin;
   for(i=0;i<3;i++)
    hash=(hash^lower_ascii((unsigned char)text[i]))*16777619u;
   bin=(int)(hash%FEATURES);
   for(i=0;i<input->hotcnt;i++)
    if(input->hotpos[i]==bin)
     break;
   if(i==input->hotcnt)
    {
     input->hotpos[i]=bin;
     input->hotval[i]=0;
     input->hotcnt++;
    }
   input->hotval[i]+=1;
  }
}

static int best_class(const real *scores)
{
 int i,best=0;
 for(i=1;i<CLASSES;i++)
  if(scores[i]>scores[best])
   best=i;
 return best;
}

int main(int argc,char **argv)
{
 const char *model=argc>1 ? argv[1] : "text-classifier.bannQ";
 enum { TRAIN_COUNT=sizeof(training)/sizeof(training[0]),
        TEST_COUNT=sizeof(test)/sizeof(test[0]) };
 int positions[FEATURES],order[TRAIN_COUNT],label,i,epoch;
 int correct_float=0,correct_quantized=0,agreement=0;
 real values[FEATURES],loss=0,max_difference=0;
 bann_fidx features={positions,values,0};
 bann_idx target={&label,1};
 bann_io input,output;
 bann net;
 bannQ quantized;

 input.kind=iokind_fidx;
 input.fidx=&features;
 output.kind=iokind_idx;
 output.idx=&target;
 srand(1);
 bann_new(&net,FEATURES,HIDDEN,1,CLASSES);
 bann_setweightactivation(&net,0,bann_activation_tanh);
 bann_setweightactivation(&net,1,bann_activation_softmax);
 bann_setloss(&net,bann_loss_cross_entropy);
 bann_setlearningrate(&net,0.03f,bann_optimizer_off,0);

 for(i=0;i<TRAIN_COUNT;i++)
  order[i]=i;
 for(epoch=0;epoch<EPOCHS;epoch++)
  {
   loss=0;
   for(i=TRAIN_COUNT-1;i>0;i--)
    {
     int j=rand()%(i+1),saved=order[i];
     order[i]=order[j];
     order[j]=saved;
    }
   for(i=0;i<TRAIN_COUNT;i++)
    {
     const sample *item=&training[order[i]];
     encode(item->text,&features);
     label=item->label;
     loss+=bann_train(&net,&input,&output);
    }
  }
 if(!bann_write(&net,model,bann_8bit) || !bannQ_read(&quantized,model))
  {
   fprintf(stderr,"Could not save or load %s\n",model);
   bann_delete(&net);
   return 1;
  }

 printf("Final training loss: %.4f\n",loss/TRAIN_COUNT);
 for(i=0;i<TEST_COUNT;i++)
  {
   const real *full_scores,*quantized_scores;
   int c,full_class,quantized_class;
   encode(test[i].text,&features);
   full_scores=bann_predict(&net,&input);
   quantized_scores=bannQ_predict(&quantized,&input);
   full_class=best_class(full_scores);
   quantized_class=best_class(quantized_scores);
   correct_float+=full_class==test[i].label;
   correct_quantized+=quantized_class==test[i].label;
   agreement+=full_class==quantized_class;
   for(c=0;c<CLASSES;c++)
    max_difference=fmaxf(max_difference,fabsf(full_scores[c]-quantized_scores[c]));
   printf("%s | expected: %s, float: %s, 8-bit: %s\n",test[i].text,
          labels[test[i].label],labels[full_class],labels[quantized_class]);
  }
 printf("Accuracy: float %d/%d, 8-bit %d/%d. Agreement: %d/%d.\n",
        correct_float,TEST_COUNT,correct_quantized,TEST_COUNT,agreement,TEST_COUNT);
 printf("Largest probability difference: %.6f\n",max_difference);
 bannQ_delete(&quantized);
 bann_delete(&net);
 return 0;
}
