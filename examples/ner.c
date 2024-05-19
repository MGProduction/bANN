#include <stdlib.h>
#include <stdio.h>

#define BANN_IMPLEMENTATION
#define BANN_DEBUG
#include "bann.h"

// ----------------------------------
// NER tagger configuration
// ----------------------------------

#define WINDOW_SIZE         8
#define HIDDEN_LAYER       256
#define HIDDEN_LAYER_COUNT  1
#define MAX_EPOCHS        128
#define MSE_STOP     0.001f

#define HEADTAIL_LIMIT      6

//#define POSTAGGER
//#define CHARSONLY

// ----------------------------------

// ----------------------------------
// helpers
// ----------------------------------

const char*getwrd(const char*sz,char*wrd)
{
 int i=0;
 while(*sz&&(*sz!=' ')&&(*sz!='\t')&&(*sz!='\r')&&(*sz!='\n'))
  wrd[i++]=*sz++;
 wrd[i]=0;
 while((*sz==' ')||(*sz=='\t')) sz++;
 return sz;
}

const char*morph(const char*w,int way)
{
 static char ww[256];
 int         i=0,j=0;
 while(w[i])
 {
  char ch='_';
  if((w[i]>='A')&&(w[i]<='Z'))
   ch='A';
  else
   if((w[i]>='a')&&(w[i]<='z'))
    ch='a';
   else
    if((w[i]>='0')&&(w[i]<='9'))
     ch='0';
    else
     ch=w[i];
  if(((way==1)&&(j>1)&&(ww[j-1]==ch)&&(ww[j-2]==ch)))
   ;
  else   
   if((way==0)&&j&&(ww[j-1]==ch))
    ;
   else   
    ww[j++]=ch;
  i++;
 }
 ww[j]=0;
 return ww;
}

int chars_compare(const void*a,const void*b)
{
 return (*(char*)a)-(*(char*)b);
}

int bann_gethighestoutput(bann*net)
{
 int  i,best=-1,hm=net->weights[net->weightscount-1].outs;
 real top,*v=net->weights[net->weightscount-1].o;
 for(i=0;i<hm;i++)
  if((best==-1)||(v[i]>top))
  {
   best=i;
   top=net->weights[net->weightscount-1].o[i];
  }
 return best;
}

// ----------------------------------


// ----------------------------------
// basic dictionary implementation
// ----------------------------------
typedef struct dict_entry_s
{
 char *key;
 int   cnt;
} dict_entry_s;

typedef struct dict_s
{
 int c,m;
 dict_entry_s *entry;
} dict_s, *dict_t;

int dict_find(dict_t dict, const char *key)
{
 int i;
 for (i = 0; i < dict->c; i++)
  if (!strcmp(dict->entry[i].key, key))
   return i;
 return -1;
}

int dict_add(dict_t dict, const char *key, int cnt)
{
 int idx = dict_find(dict, key);
 if (idx != -1) 
 {
     dict->entry[idx].cnt += cnt;
     return idx;
 }
 if (dict->c == dict->m)
 {
     dict->m *= 2;
     dict->entry = realloc(dict->entry, dict->m * sizeof(dict_entry_s));
 }
 dict->entry[dict->c].key = _strdup(key);
 dict->entry[dict->c].cnt = cnt;
 dict->c++;
 return dict->c-1;
}

dict_t dict_new(void)
{
 dict_t d = malloc(sizeof(dict_s));
 d->c=0;
 d->m=10;
 d->entry=(dict_entry_s*)malloc(d->m*sizeof(dict_entry_s));
 return d;
}

void dict_free(dict_t dict)
{
 int i;
 for (i = 0; i < dict->c; i++)
  free(dict->entry[i].key);
 free(dict->entry);
 free(dict);
}

void dict_emit(const char*head,dict_t d)
{
 int i;
 printf("%s: ",head);
 for(i=0;i<d->c;i++)
  {
   if(i) printf(", ");
   printf(d->entry[i].key);
  }
 printf("\n");
}
// ----------------------------------

// ----------------------------------
// basic corpus implementation
// ----------------------------------

#define MAX_CHARS 256

typedef struct{
 bann   N;
 dict_t cdict;
 int    ichars,charseq,corpusseq,maxlen,window_size,tagcnt;
 byte   usedchars[MAX_CHARS];
}NERtagger;

void NERtagger_new(NERtagger*ner,dict_t dict,dict_t nerdict)
{
 int  e,sum=0,avg=0,top=0;
 char chars[MAX_CHARS];

 memset(ner,0,sizeof(NERtagger));
 ner->cdict=dict_new();
 memset(ner->usedchars,0,sizeof(ner->usedchars));
 
 for(e=0;e<dict->c;e++)
  {
   if(dict->entry[e].cnt>top)
    top=dict->entry[e].cnt;
   sum+=dict->entry[e].cnt;
  }
 avg=(sum-top)/(dict->c-1);
 avg=max(2,avg/2);
 
 ner->usedchars[2]=1;
 for(e=0;e<dict->c;e++)
  {
   const char*wrd=dict->entry[e].key;
   int len=(int)strlen(wrd);
   if(len>ner->maxlen)
    ner->maxlen=len;
   len=0;
   while(wrd[len])
    {
     byte ch=(byte)wrd[len++];
     if((ch>='A')&&(ch<='Z'))
      ch=(ch-'A')+'a';
     ner->usedchars[ch]=1;
    }
   if(dict->entry[e].cnt>avg)
    dict_add(ner->cdict,dict->entry[e].key,dict->entry[e].cnt);
   else
    {
     dict_add(ner->cdict,morph(dict->entry[e].key,1),dict->entry[e].cnt);
     dict_add(ner->cdict,morph(dict->entry[e].key,0),dict->entry[e].cnt);
    }
  }
 for(e=0;e<MAX_CHARS;e++)
  if(ner->usedchars[e])
   {
    ner->usedchars[e]=ner->ichars;
    chars[ner->ichars++]=e;
   }
  else
   ner->usedchars[e]=MAX_CHARS-1;
 chars[ner->ichars]=0;
 qsort(chars,ner->ichars,sizeof(chars[0]),chars_compare);

 ner->window_size=WINDOW_SIZE;
 ner->tagcnt=nerdict->c;
#if defined(CHARSONLY)
 ner->charseq=(ner->ichars*(HEADTAIL_LIMIT*2)+1+1);
 ner->corpusseq=ner->charseq*(ner->window_size*2+1);
 bann_new(&ner->N,ner->corpusseq,HIDDEN_LAYER,HIDDEN_LAYER_COUNT,nerdict->c);
#else
 if(HEADTAIL_LIMIT)
  ner->charseq=(ner->ichars*(HEADTAIL_LIMIT*2));
 else
  ner->charseq=(ner->ichars*ner->maxlen);
 ner->corpusseq=(ner->cdict->c+ner->tagcnt)*(ner->window_size*2+1);
 bann_new(&ner->N,(ner->corpusseq+ner->charseq),HIDDEN_LAYER,1,nerdict->c);
#endif

 printf("creating bANN with %d x %d x %d cells\n",(ner->corpusseq+ner->charseq),HIDDEN_LAYER,nerdict->c); 
}

void NERtagger_delete(NERtagger*ner)
{
 bann_delete(&ner->N);
}

// ----------------------------------
// basic corpus implementation
// ----------------------------------

typedef struct{
 int fit,fitR;
 int pos;
 int ner;
 int nerT;
}word;

typedef struct{
 word*b;
 int  c,m;

 dict_t dict;
 dict_t posdict;
 dict_t nerdict;

}corpus;

void corpus_new(corpus*w)
{
 w->c=0;w->m=256;
 w->b=(word*)malloc(w->m*sizeof(w->b[0]));
 w->dict=dict_new();
 w->posdict=dict_new();
 w->nerdict=dict_new();
 #if !defined(POSTAGGER)
 dict_add(w->nerdict,"O",0);
 #endif
}

void corpus_delete(corpus*w)
{
 dict_free(w->dict);
 dict_free(w->posdict);
 dict_free(w->nerdict);
 free(w->b);
}
void corpus_add(corpus*w,word*ww)
{
 if(w->c+1==w->m)
 {
  w->m*=2;
  w->b=(word*)realloc(w->b,w->m*sizeof(w->b[0]));
 }
 w->b[w->c++]=*ww;
}

int corpus_read(corpus*w,const char*name)
{
 FILE*f;
 printf("reading %s...\n",name);
 f=fopen(name,"rb");
 if(f)
  {
   while(!feof(f))
   {
    char       line[1024],wrd[256],ner[256],tmp[256],pos[256];
    const char*next;
    int        i;
    word       ww;
    memset(&ww,0xFF,sizeof(ww));

    fgets(line,sizeof(line),f);
    
    i=(int)strlen(line);
    while(i&&((line[i-1]=='\r')||(line[i-1]=='\n')||(line[i-1]==' ')))
     i--;
    line[i]=0;

    if(*line==0)
     corpus_add(w,&ww);
    else
     {
      next=getwrd(line,wrd);
      ww.fit=dict_add(w->dict,wrd,1);

#if defined(POSTAGGER)
      next=getwrd(next,ner);      
      ww.ner=dict_add(w->nerdict,ner,1);      

      next=getwrd(next,tmp);
      next=getwrd(next,pos);
      ww.pos=dict_add(w->posdict,pos,1);
#else
      next=getwrd(next,pos);
      ww.pos=dict_add(w->posdict,pos,1);

      next=getwrd(next,tmp);
      next=getwrd(next,ner);      
      ww.ner=dict_add(w->nerdict,ner,1);
#endif

      corpus_add(w,&ww);
     }
   }
   fclose(f);
   return 1;
  }
 else
  {
   printf("Error: can't find %s\n",name);
   return 0;
  }
}

void corpus_remap(corpus*wrd,dict_t cdict,corpus*remap)
{
 int e,err=0;
 for(e=0;e<wrd->c;e++)
 {
  if(wrd->b[e].fit!=-1)
   {
    const char*fit=wrd->dict->entry[wrd->b[e].fit].key;
    int id=dict_find(cdict,fit);
    if(id!=-1)
     wrd->b[e].fitR=id;
    else
     {
      id=dict_find(cdict,morph(fit,1));
      if(id==-1)
       id=dict_find(cdict,morph(fit,0));
      wrd->b[e].fitR=id;
     }
   }
  else
   wrd->b[e].fitR=-1;
  if(remap)
   {
    if(wrd->b[e].pos!=-1)
     {
      const char*spos=wrd->posdict->entry[wrd->b[e].pos].key;
      int id=dict_find(remap->posdict,spos);
      if(id!=-1)
       wrd->b[e].pos=id;
      else
       err++;
     }
    if(wrd->b[e].ner!=-1)
     {
      const char*sner=wrd->nerdict->entry[wrd->b[e].ner].key;
      int id=dict_find(remap->nerdict,sner);
      if(id!=-1)
       wrd->b[e].ner=id;
      else
       err++;
     }
   }
 }
 if(remap)
  {
   int n;
   dict_free(wrd->posdict);
   dict_free(wrd->nerdict);
   wrd->posdict=dict_new();
   for(n=0;n<remap->posdict->c;n++)
    dict_add(wrd->posdict,remap->posdict->entry[n].key,remap->posdict->entry[n].cnt);
   wrd->nerdict=dict_new();
   for(n=0;n<remap->nerdict->c;n++)
    dict_add(wrd->nerdict,remap->nerdict->entry[n].key,remap->nerdict->entry[n].cnt);
  }
}

void corpus_setinputlayer(NERtagger*ner,corpus*wrd,int w,bann_io*in)
{
 int        ww,p,odd=0;
 in->idx->hotcnt=0;
#if defined(CHARSONLY)
 for(p=0,ww=-ner->window_size;ww<=ner->window_size;ww++)
  {
   int base=p*ner->charseq;
   if((w+ww>=0)&&(w+ww<wrd->c)&&(wrd->b[w+ww].fitR!=-1))
    {
     const char*fit=wrd->dict->entry[wrd->b[w].fit].key;
     int        c,cc,id,up=0;
     
     for(c=0;fit[c];c++)
      {
       byte ch=(byte)fit[c];
       if((ch>='A')&&(ch<='Z'))
        up++;
      }     
     if(up==1)
      in->idx->hotpos[in->idx->hotcnt++]=base;
     if(up>1)
      in->idx->hotpos[in->idx->hotcnt++]=base+1;
     base+=2;

     for(c=0;fit[c];c++)
      if(c<HEADTAIL_LIMIT)
       {
        byte ch=(byte)fit[c];
        if((ch>='A')&&(ch<='Z'))
         ch=(ch-'A')+'a';
        id=ner->usedchars[ch];
        if(id!=255)
         in->idx->hotpos[in->idx->hotcnt++]=base+c*ner->ichars+id;
        else
         odd++;
       }
      else
       break;
     c=(int)strlen(fit);cc=0;
     while(c--)
      {
       byte ch=(byte)fit[c];
       if((ch>='A')&&(ch<='Z'))
        ch=(ch-'A')+'a';
       id=ner->usedchars[ch];
       if(id!=255)
        in->idx->hotpos[in->idx->hotcnt++]=base+(HEADTAIL_LIMIT+cc*ner->ichars)+id;
       else
        odd++;
       cc++;
       if(cc==HEADTAIL_LIMIT)
        break;
      }
    }
   p++;
  }
#else
 // add word chars
 if(ner->ichars)
  {
   int        c,cc=0;
   const char*fit=wrd->dict->entry[wrd->b[w].fit].key;
   if(HEADTAIL_LIMIT)
    {
     for(c=0;fit[c];c++)
      if(c<HEADTAIL_LIMIT)
       {
        int id=ner->usedchars[(byte)fit[c]];
        if(id!=255)
         in->idx->hotpos[in->idx->hotcnt++]=c*ner->ichars+id;
        else
         odd++;
       }
      else
       break;
     c=(int)strlen(fit);
     while(c--)
      {
       int id=ner->usedchars[(byte)fit[c]];
       if(id!=255)
        in->idx->hotpos[in->idx->hotcnt++]=(HEADTAIL_LIMIT+cc*ner->ichars)+id;
       else
        odd++;
       cc++;
       if(cc==HEADTAIL_LIMIT)
        break;
      }
    }
   else
    {
     for(c=0;fit[c];c++)
      if(c<ner->maxlen)
       {
        int id=ner->usedchars[(byte)fit[c]];
        if(id!=255)
         in->idx->hotpos[in->idx->hotcnt++]=c*ner->ichars+id;
        else
         odd++;
       }
      else
       break;
    }
  }
 // add context window        
 if(ner->window_size)
  for(p=0,ww=-ner->window_size;ww<=ner->window_size;ww++)
   //if(ww)
    {
     if((w+ww>=0)&&(w+ww<wrd->c)&&(wrd->b[w+ww].fitR!=-1))
      {
       int id=wrd->b[w+ww].fitR;
       int slotid=p*(ner->cdict->c+ner->tagcnt)+ner->charseq;
       if((slotid+id>=0)&&(slotid+id<ner->N.weights[0].ins))
       {
        in->idx->hotpos[in->idx->hotcnt++]=slotid+id;
        if(ww<0)
         if(wrd->b[w+ww].nerT!=-1)
          in->idx->hotpos[in->idx->hotcnt++]=slotid+ner->cdict->c+wrd->b[w+ww].nerT;    
         else
          odd++;
       }
       else
        odd++;
      }
     p++;
    }
#endif
}

void ner_qual(int*good,int*add,int*miss,int hm,real*acc,real*prec,real*recall,real*F1)
{
 real gprec=0,grecall=0,gacc=0;
 int  i=0,ggood=0,gmiss=0,gadd=0,cnt=0,tgood=0,tadd=0,tmiss=0;
#if defined(POSTAGGER)
 for(i=0;i<hm;i++)
#else
 for(i=1;i<hm;i++) // skip O
#endif
  {
   ggood=good[i];
   gmiss=miss[i];
   gadd=add[i];
   tgood+=good[i];
   tmiss+=miss[i];
   tadd+=add[i];
  }
 gprec=tgood*100.0f/(tgood+tadd);
 grecall=tgood*100.0f/(tgood+tmiss);
 gacc=(float)(tgood)*100.0f/(float)(tgood+tmiss+tadd);
 if(acc) *acc=gacc;
 if(prec) *prec=gprec;
 if(recall) *recall=grecall;
 if(F1) *F1=2*gprec*grecall/(gprec+grecall);
}

void corpus_predict(NERtagger*ner,bann_io*in,corpus*wrdA,const char*name)
{
 bann*N=&ner->N;
 real acc,prec,recall,F1;
 int  goodN[256],missN[256],addN[256],w,cnt=0,hmNER=N->weights[N->weightscount-1].outs; 
 
 memset(goodN,0,hmNER*sizeof(goodN[0]));
 memset(missN,0,hmNER*sizeof(missN[0]));
 memset(addN,0,hmNER*sizeof(addN[0]));

 for(w=0;w<wrdA->c;w++)
  if(wrdA->b[w].fitR!=-1)
   {
    corpus_setinputlayer(ner,wrdA,w,in);           
    if(in->idx->hotcnt)
     {
      int        sel,correct=wrdA->b[w].ner;
      const char*fit=wrdA->dict->entry[wrdA->b[w].fit].key;
      const char*fitR=ner->cdict->entry[wrdA->b[w].fitR].key;
      const char*ner=wrdA->nerdict->entry[correct].key;
      const char*selner="";
      
      bann_predict(N,in);

      sel=bann_gethighestoutput(N);
      selner=wrdA->nerdict->entry[sel].key;
      if(strcmp(ner,selner)==0)
       goodN[sel]++;
      else
       {
        addN[sel]++;
        missN[correct]++;
       }       
      wrdA->b[w].nerT=sel;
      cnt++;
      if((cnt&0xFF)==0xFF)
      {
       ner_qual(goodN,addN,missN,hmNER,&acc,&prec,&recall,&F1);
       printf("[%s] F1: %.2f P: %.2f R: %.2f  \r",name,F1,prec,recall);
      }
     }
   }
 ner_qual(goodN,addN,missN,hmNER,&acc,&prec,&recall,&F1);
 printf("[%s] F1: %.2f P: %.2f R: %.2f  \n",name,F1,prec,recall);
}

// ----------------------------------

void shuffle(int*idx,int size)
{
 int i;
 for(i=0;i<size;i++)
  {
   int a=rand()%size;
   int b=rand()%size;
   if(a!=b)
   {
    int t=idx[a];
    idx[a]=idx[b];
    idx[b]=t;
   }
  }
}

// ----------------------------------


int main(int argc,const char*argv[])
{
 NERtagger NER;
 corpus    train,testA,testB; 
 int       e,inw[256],o[2],*index;   
 real      lasterr=100;
 bann_idx  a,b;
 bann_io   in,out;

 corpus_new(&train);
 if(corpus_read(&train,"bin/data/ner/eng.train"))
  {
   dict_emit("NER",train.nerdict);
   dict_emit("POS",train.posdict);
   NERtagger_new(&NER,train.dict,train.nerdict);
   corpus_remap(&train,NER.cdict,NULL);
  }
 
 corpus_new(&testA);
 if(corpus_read(&testA,"bin/data/ner/eng.testa"))
  {
   corpus_remap(&testA,NER.cdict,&train);
  }

 corpus_new(&testB);
 if(corpus_read(&testB,"bin/data/ner/eng.testb"))
  {
   corpus_remap(&testB,NER.cdict,&train);
  }
     
 in.kind=idx;in.idx=&a;
 a.hotpos=&inw[0];

 out.kind=idx;out.idx=&b;
 b.hotpos=&o[0];

 index=(int*)malloc(train.c*sizeof(int));
 for(e=0;e<train.c;e++)
  index[e]=e;

 for(e=0;e<MAX_EPOCHS;e++)
  {
   real err=0;
   int  cnt=0,ww,w,goodN[256],addN[256],missN[256],hmNER=NER.N.weights[NER.N.weightscount-1].outs;
   real acc,prec,recall,F1;

   memset(goodN,0,hmNER*sizeof(goodN[0]));
   memset(missN,0,hmNER*sizeof(missN[0]));
   memset(addN,0,hmNER*sizeof(addN[0]));

   shuffle(index,train.c);
   for(ww=0;ww<train.c;ww++)
    {
     w=index[ww];
     if(train.b[w].fitR!=-1)
      {
       corpus_setinputlayer(&NER,&train,w,&in);
              
       if(in.idx->hotcnt)
        {
         int        sel,correct=train.b[w].ner;
         const char*fit=train.dict->entry[train.b[w].fit].key;
         const char*ner=train.nerdict->entry[correct].key;
         const char*selner="";
         
         out.idx->hotcnt=1;
         out.idx->hotpos[0]=train.b[w].ner;
         //bann_predict(N,in);
         err+=bann_train(&NER.N,&in,&out);

         sel=bann_gethighestoutput(&NER.N);
         selner=train.nerdict->entry[sel].key;
         if(strcmp(ner,selner)==0)
          goodN[sel]++;
         else
          {
           addN[sel]++;
           missN[correct]++;
          }       
         train.b[w].nerT=sel;
         cnt++;
         if((cnt&0xFF)==0xFF)
         {
          ner_qual(goodN,addN,missN,hmNER,&acc,&prec,&recall,&F1);
          printf("[%s.%d] F1: %.2f P: %.2f R: %.2f MSE: %.2f  \r","eng.train",e+1,F1,prec,recall,err/cnt);
         }
        }
        
      }
    }
   ner_qual(goodN,addN,missN,hmNER,&acc,&prec,&recall,&F1);
   err/=cnt;
   printf("[%s.%d] F1: %.2f P: %.2f R: %.2f MSE: %.2f  \n","eng.train",e+1,F1,prec,recall,err);

   if((err<0.5f)||(e+1==MAX_EPOCHS))
    {
     corpus_predict(&NER,&in,&testA,"eng.testa");
     corpus_predict(&NER,&in,&testB,"eng.testb");    
     if(err<lasterr)
     {
      bann_write(&NER.N,"bin/ner.bann",0);
      bann_write(&NER.N,"bin/ner.bannQ",1);
      lasterr=err;
      if(err<MSE_STOP)
       break;
     }
    }
  }

 NERtagger_delete(&NER);

 corpus_delete(&train);
 corpus_delete(&testA);
 corpus_delete(&testB);

 return 1;
}