#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BANN_IMPLEMENTATION
#include "bann.h"

enum { MAX_WORDS=128, MAX_TOKENS=12, RADIUS=2, EMBEDDING=16, EPOCHS=250 };

/* Each row is a sentence. Context windows stop at sentence boundaries. */
static char corpus[][72]={
 "the violin plays a melody in the orchestra",
 "the cello plays a melody in the orchestra",
 "the flute plays a melody in the orchestra",
 "she practices the violin before rehearsal",
 "she practices the cello before rehearsal",
 "she practices the flute before rehearsal",
 "the violin sounds clear in the concert hall",
 "the cello sounds deep in the concert hall",
 "the flute sounds bright in the concert hall",
 "the tiger hunts for prey in the forest",
 "the leopard hunts for prey in the forest",
 "the jaguar hunts for prey in the forest",
 "we saw a tiger beside the river",
 "we saw a leopard beside the river",
 "we saw a jaguar beside the river",
 "the tiger hides among the trees",
 "the leopard hides among the trees",
 "the jaguar hides among the trees",
 "mars orbits the sun in our solar system",
 "venus orbits the sun in our solar system",
 "jupiter orbits the sun in our solar system",
 "we observed mars through the telescope",
 "we observed venus through the telescope",
 "we observed jupiter through the telescope",
 "mars is a planet visible at night",
 "venus is a planet visible at night",
 "jupiter is a planet visible at night",
};

static int word_index(const char *word,const char **vocabulary,int size)
{
 int i;
 for(i=0;i<size;i++)
  if(strcmp(word,vocabulary[i])==0)
   return i;
 return -1;
}

/* The input feature's column holds its embedding, one value per hidden unit. */
static real cosine(const weight *layer,int first,int second)
{
 int unit;
 real dot=0,norm_first=0,norm_second=0;
 for(unit=0;unit<layer->outs;unit++)
  {
   const real *row=layer->w+unit*layer->ins;
   dot+=row[first]*row[second];
   norm_first+=row[first]*row[first];
   norm_second+=row[second]*row[second];
  }
 return norm_first>0 && norm_second>0 ? dot/sqrtf(norm_first*norm_second) : 0;
}

static void show_neighbors(const weight *layer,const char **vocabulary,int size,const char *word)
{
 real similarity[MAX_WORDS];
 int i,rank,query=word_index(word,vocabulary,size);
 for(i=0;i<size;i++)
  similarity[i]=i==query ? -2 : cosine(layer,query,i);
 printf("%s:",word);
 for(rank=0;rank<3;rank++)
  {
   int best=0;
   for(i=1;i<size;i++)
    if(similarity[i]>similarity[best])
     best=i;
   printf(" %s (%.3f)",vocabulary[best],similarity[best]);
   similarity[best]=-2;
  }
 putchar('\n');
}

int main(void)
{
 enum { SENTENCES=sizeof(corpus)/sizeof(corpus[0]) };
 const char *vocabulary[MAX_WORDS];
 int sentence_ids[SENTENCES][MAX_TOKENS],lengths[SENTENCES]={0};
 int vocabulary_size=0,sentence,epoch,center,label,positions[2*RADIUS];
 real values[2*RADIUS],loss=0,initial_loss=0;
 bann_fidx context={positions,values,0};
 bann_idx target={&label,1};
 bann_io input,output;
 bann net;

 for(sentence=0;sentence<SENTENCES;sentence++)
  {
   char *word=strtok(corpus[sentence]," ");
   for(;word;word=strtok(NULL," "))
    {
     int id=word_index(word,vocabulary,vocabulary_size);
     if(lengths[sentence]==MAX_TOKENS || (id<0 && vocabulary_size==MAX_WORDS))
      {
       fprintf(stderr,"The embedded corpus exceeds the example's capacity.\n");
       return 1;
      }
     if(id<0)
      {
       id=vocabulary_size++;
       vocabulary[id]=word;
      }
     sentence_ids[sentence][lengths[sentence]++]=id;
    }
  }

 input.kind=iokind_fidx;
 input.fidx=&context;
 output.kind=iokind_idx;
 output.idx=&target;
 srand(1);
 bann_new(&net,vocabulary_size,EMBEDDING,1,vocabulary_size);
 bann_setweightactivation(&net,0,bann_activation_linear);
 bann_setweightactivation(&net,1,bann_activation_softmax);
 bann_setloss(&net,bann_loss_cross_entropy);
 bann_setlearningrate(&net,0.05f,bann_optimizer_off,0);

 for(epoch=0;epoch<EPOCHS;epoch++)
  {
   int examples=0;
   loss=0;
   for(sentence=0;sentence<SENTENCES;sentence++)
    for(center=0;center<lengths[sentence];center++)
     {
      int offset,i;
      context.hotcnt=0;
      label=sentence_ids[sentence][center];
      for(offset=-RADIUS;offset<=RADIUS;offset++)
       if(offset && center+offset>=0 && center+offset<lengths[sentence])
        positions[context.hotcnt++]=sentence_ids[sentence][center+offset];
      /* Repeated words contribute once per occurrence to the average. */
      for(i=0;i<context.hotcnt;i++)
       values[i]=1.0f/context.hotcnt;
      loss+=bann_train(&net,&input,&output);
      examples++;
     }
   loss/=examples;
   if(epoch==0)
    initial_loss=loss;
  }

 printf("%d words, %d dimensions. Mean training loss: %.3f -> %.3f\n",
        vocabulary_size,EMBEDDING,initial_loss,loss);
 puts("Nearest words by cosine similarity:");
 show_neighbors(&net.weights[0],vocabulary,vocabulary_size,"violin");
 show_neighbors(&net.weights[0],vocabulary,vocabulary_size,"tiger");
 show_neighbors(&net.weights[0],vocabulary,vocabulary_size,"mars");
 bann_delete(&net);
 return 0;
}
