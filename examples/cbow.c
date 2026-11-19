#include <errno.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define BANN_IMPLEMENTATION
#include "bann.h"

enum { RADIUS=2, DEFAULT_WORDS=1000, DEFAULT_DIMENSIONS=32, DEFAULT_EPOCHS=10 };

static const char demo[]=
"the violin plays a melody in the orchestra\n"
"the cello plays a melody in the orchestra\n"
"the flute plays a melody in the orchestra\n"
"she practices the violin before rehearsal\n"
"she practices the cello before rehearsal\n"
"she practices the flute before rehearsal\n"
"the violin sounds clear in the concert hall\n"
"the cello sounds deep in the concert hall\n"
"the flute sounds bright in the concert hall\n"
"the tiger hunts for prey in the forest\n"
"the leopard hunts for prey in the forest\n"
"the jaguar hunts for prey in the forest\n"
"we saw a tiger beside the river\n"
"we saw a leopard beside the river\n"
"we saw a jaguar beside the river\n"
"the tiger hides among the trees\n"
"the leopard hides among the trees\n"
"the jaguar hides among the trees\n"
"mars orbits the sun in our solar system\n"
"venus orbits the sun in our solar system\n"
"jupiter orbits the sun in our solar system\n"
"we observed mars through the telescope\n"
"we observed venus through the telescope\n"
"we observed jupiter through the telescope\n"
"mars is a planet visible at night\n"
"venus is a planet visible at night\n"
"jupiter is a planet visible at night\n";

typedef struct {
 char *word;
 size_t sentence;
 int id;
} token;

typedef struct {
 token **occurrences;
 size_t count;
} word_group;

static char *read_corpus(const char *path)
{
 FILE *file=fopen(path,"rb");
 char *text=NULL;
 long length;
 if(!file)
  return NULL;
 if(fseek(file,0,SEEK_END)==0 && (length=ftell(file))>=0 &&
    fseek(file,0,SEEK_SET)==0 && (uintmax_t)length<SIZE_MAX)
  {
   text=(char*)malloc((size_t)length+1);
   if(text && fread(text,1,(size_t)length,file)==(size_t)length)
    text[length]='\0';
   else
    {
     free(text);
     text=NULL;
    }
  }
 fclose(file);
 return text;
}

static unsigned char lower_ascii(unsigned char c)
{
 return c>='A' && c<='Z' ? (unsigned char)(c-'A'+'a') : c;
}

static int is_word_byte(unsigned char c)
{
 c=lower_ascii(c);
 return (c>='a' && c<='z') || (c>='0' && c<='9') || c>=128;
}

static token *tokenize(char *text,size_t *count)
{
 token *tokens=NULL;
 size_t capacity=0,sentence=0;
 char *start=NULL,*cursor;
 *count=0;
 for(cursor=text;;cursor++)
  {
   unsigned char c=(unsigned char)*cursor;
   if(is_word_byte(c))
    {
     if(!start)
      start=cursor;
     *cursor=(char)lower_ascii(c);
     continue;
    }
   if(start)
    {
     if(*count==capacity)
      {
       size_t next=capacity ? capacity*2 : 1024;
       token *grown;
       if(next<capacity || next>SIZE_MAX/sizeof(*tokens))
        break;
       grown=(token*)realloc(tokens,next*sizeof(*tokens));
       if(!grown)
        break;
       tokens=grown;
       capacity=next;
      }
     *cursor='\0';
     tokens[*count].word=start;
     tokens[*count].sentence=sentence;
     tokens[*count].id=-1;
     (*count)++;
     start=NULL;
    }
   if(!c)
    return tokens;
   if(c=='.' || c=='?' || c=='!' || c=='\r' || c=='\n')
    sentence++;
  }
 free(tokens);
 *count=0;
 return NULL;
}

static int compare_tokens(const void *a,const void *b)
{
 const token *first=*(token *const*)a,*second=*(token *const*)b;
 int order=strcmp(first->word,second->word);
 if(order)
  return order;
 return (first>second)-(first<second);
}

static int compare_frequency(const void *a,const void *b)
{
 const word_group *first=(const word_group*)a,*second=(const word_group*)b;
 if(first->count!=second->count)
  return first->count>second->count ? -1 : 1;
 return strcmp(first->occurrences[0]->word,second->occurrences[0]->word);
}

static int compare_appearance(const void *a,const void *b)
{
 const word_group *first=(const word_group*)a,*second=(const word_group*)b;
 return (first->occurrences[0]>second->occurrences[0])-
        (first->occurrences[0]<second->occurrences[0]);
}

static const char **build_vocabulary(token *tokens,size_t count,int limit,int use_demo,
                                    int *size,size_t *retained)
{
 token **ordered;
 word_group *groups;
 const char **vocabulary=NULL;
 size_t i,group_count=0;
 *size=0;
 *retained=0;
 if(count>SIZE_MAX/sizeof(*ordered) || count>SIZE_MAX/sizeof(*groups))
  return NULL;
 ordered=(token**)malloc(count*sizeof(*ordered));
 groups=(word_group*)malloc(count*sizeof(*groups));
 if(!ordered || !groups)
  goto done;
 for(i=0;i<count;i++)
  ordered[i]=&tokens[i];
 qsort(ordered,count,sizeof(*ordered),compare_tokens);
 for(i=0;i<count;)
  {
   size_t end=i+1;
   while(end<count && strcmp(ordered[i]->word,ordered[end]->word)==0)
    end++;
   if(use_demo || end-i>=2)
    {
     groups[group_count].occurrences=ordered+i;
     groups[group_count].count=end-i;
     group_count++;
    }
   i=end;
  }
 if(group_count>(size_t)limit)
  {
   qsort(groups,group_count,sizeof(*groups),compare_frequency);
   group_count=(size_t)limit;
  }
 if(!group_count)
  goto done;
 qsort(groups,group_count,sizeof(*groups),use_demo ? compare_appearance : compare_frequency);
 vocabulary=(const char**)malloc(group_count*sizeof(*vocabulary));
 if(!vocabulary)
  goto done;
 for(i=0;i<group_count;i++)
  {
   size_t j;
   vocabulary[i]=groups[i].occurrences[0]->word;
   for(j=0;j<groups[i].count;j++)
    groups[i].occurrences[j]->id=(int)i;
   *retained+=groups[i].count;
  }
 *size=(int)group_count;
done:
 free(groups);
 free(ordered);
 return vocabulary;
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

static void show_neighbors(const weight *layer,const char **vocabulary,int size,const char *word,int limit)
{
 int best[5],i,query,found=0;
 real scores[5];
 for(query=0;query<size;query++)
  {
   size_t j=0;
   while(word[j] && lower_ascii((unsigned char)word[j])==(unsigned char)vocabulary[query][j])
    j++;
   if(!word[j] && !vocabulary[query][j])
    break;
  }
 if(query==size)
  {
   printf("%s: outside the retained vocabulary\n",word);
   return;
  }
 for(i=0;i<size;i++)
  {
   int rank,pos;
   real score;
   if(i==query)
    continue;
   score=cosine(layer,query,i);
   for(rank=0;rank<found && score<=scores[rank];rank++)
    ;
   if(rank==limit)
    continue;
   if(found<limit)
    found++;
   for(pos=found-1;pos>rank;pos--)
    {
     scores[pos]=scores[pos-1];
     best[pos]=best[pos-1];
    }
   scores[rank]=score;
   best[rank]=i;
  }
 printf("%s:",word);
 for(i=0;i<found;i++)
  printf(" %s (%.3f)",vocabulary[best[i]],scores[i]);
 putchar('\n');
}

static int parse_positive(const char *arg,int *value)
{
 char *end;
 long parsed;
 errno=0;
 parsed=strtol(arg,&end,10);
 if(errno || end==arg || *end || parsed<=0 || parsed>INT_MAX)
  return 0;
 *value=(int)parsed;
 return 1;
}

int main(int argc,char **argv)
{
 int use_demo=argc==1,limit=DEFAULT_WORDS,dimensions=DEFAULT_DIMENSIONS,epochs=DEFAULT_EPOCHS;
 int vocabulary_size,epoch,label,positions[2*RADIUS],status=1;
 size_t count,retained,center;
 uint64_t network_size;
 char *text;
 token *tokens;
 const char **vocabulary;
 real values[2*RADIUS],initial_loss=0;
 bann_fidx context={positions,values,0};
 bann_idx target={&label,1};
 bann_io input,output;
 bann net;

 if((argc>2 && !parse_positive(argv[2],&limit)) ||
    (argc>3 && !parse_positive(argv[3],&dimensions)) ||
    (argc>4 && !parse_positive(argv[4],&epochs)))
  {
   fprintf(stderr,"Usage: %s [corpus [vocabulary [dimensions [epochs [word ...]]]]]\n",argv[0]);
   return 1;
  }
 if(use_demo)
  {
   dimensions=16;
   epochs=250;
   text=(char*)malloc(sizeof(demo));
   if(text)
    memcpy(text,demo,sizeof(demo));
  }
 else
  text=read_corpus(argv[1]);
 if(!text)
  {
   fprintf(stderr,"Could not read the corpus.\n");
   return 1;
  }
 tokens=tokenize(strncmp(text,"\xef\xbb\xbf",3)==0 ? text+3 : text,&count);
 if(!tokens)
  {
   fprintf(stderr,"The corpus contains no words or could not be tokenized.\n");
   free(text);
   return 1;
  }
 vocabulary=build_vocabulary(tokens,count,limit,use_demo,&vocabulary_size,&retained);
 if(vocabulary_size<2)
  {
   fprintf(stderr,"Need at least two retained words. File mode keeps words seen at least twice.\n");
   goto cleanup;
  }
 network_size=2*(uint64_t)vocabulary_size*dimensions+3*((uint64_t)vocabulary_size+dimensions);
 /* bann_new also evaluates the hidden-to-hidden size and input * output. */
 if((uint64_t)vocabulary_size*vocabulary_size>INT_MAX ||
    (uint64_t)dimensions*((uint64_t)dimensions+3)>INT_MAX ||
    network_size>INT_MAX || network_size>SIZE_MAX/sizeof(real))
  {
   fprintf(stderr,"The requested network exceeds bANN's size limits.\n");
   goto cleanup;
  }

 input.kind=iokind_fidx;
 input.fidx=&context;
 output.kind=iokind_idx;
 output.idx=&target;
 srand(1);
 bann_new(&net,vocabulary_size,dimensions,1,vocabulary_size);
 bann_setweightactivation(&net,0,bann_activation_linear);
 bann_setweightactivation(&net,1,bann_activation_softmax);
 bann_setloss(&net,bann_loss_cross_entropy);
 bann_setlearningrate(&net,0.05f,bann_optimizer_off,0);
 printf("%d words, %d dimensions, %zu/%zu tokens retained.\n",
        vocabulary_size,dimensions,retained,count);

 for(epoch=0;epoch<epochs;epoch++)
  {
   size_t examples=0;
   double loss=0;
   clock_t started=clock();
   for(center=0;center<count;center++)
    {
     int offset,i;
     label=tokens[center].id;
     if(label<0)
      continue;
     context.hotcnt=0;
     for(offset=-RADIUS;offset<=RADIUS;offset++)
      {
       size_t neighbor;
       if(!offset || (offset<0 && center<(size_t)-offset))
        continue;
       neighbor=offset<0 ? center-(size_t)-offset : center+(size_t)offset;
       if(neighbor<count && tokens[neighbor].sentence==tokens[center].sentence &&
          tokens[neighbor].id>=0)
        positions[context.hotcnt++]=tokens[neighbor].id;
      }
     if(!context.hotcnt)
      continue;
     /* Repeated words contribute once per occurrence to the average. */
     for(i=0;i<context.hotcnt;i++)
      values[i]=1.0f/context.hotcnt;
     loss+=bann_train(&net,&input,&output);
     examples++;
    }
   if(!examples)
    {
     fprintf(stderr,"No training examples have a retained context word.\n");
     bann_delete(&net);
     goto cleanup;
    }
   loss/=examples;
   if(epoch==0)
    initial_loss=(real)loss;
   if(!use_demo)
    printf("Epoch %d/%d: loss %.4f, %zu examples, %.2f s CPU\n",
           epoch+1,epochs,loss,examples,(double)(clock()-started)/CLOCKS_PER_SEC);
   if(epoch==epochs-1)
    printf("Mean training loss: %.3f -> %.3f\n",initial_loss,loss);
  }

 puts("Nearest words by cosine similarity:");
 if(use_demo)
  {
   show_neighbors(&net.weights[0],vocabulary,vocabulary_size,"violin",3);
   show_neighbors(&net.weights[0],vocabulary,vocabulary_size,"tiger",3);
   show_neighbors(&net.weights[0],vocabulary,vocabulary_size,"mars",3);
  }
 else if(argc>5)
  {
   int i;
   for(i=5;i<argc;i++)
    show_neighbors(&net.weights[0],vocabulary,vocabulary_size,argv[i],5);
  }
 else
  {
   int i;
   for(i=0;i<vocabulary_size && i<3;i++)
    show_neighbors(&net.weights[0],vocabulary,vocabulary_size,vocabulary[i],5);
  }
 bann_delete(&net);
 status=0;
cleanup:
 free(vocabulary);
 free(tokens);
 free(text);
 return status;
}
