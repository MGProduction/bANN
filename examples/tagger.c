#include <stdlib.h>
#include <stdio.h>
#include <ctype.h>

#define BANN_IMPLEMENTATION
#define BANN_DEBUG
#include "bann.h"

// ----------------------------------
// NER/POS tagger configuration
// ----------------------------------

#define WINDOW_SIZE         5
#define HIDDEN_LAYER       96
#define HIDDEN_LAYER_COUNT  1
#define MAX_EPOCHS        128
#define MSE_STOP       0.001f

// ----------------------------------

// ----------------------------------
// helpers
// ----------------------------------

void history_add(bann*N,const char*file,const char*name,const char*quality)
{
 FILE*f=fopen(file,"ab");
 if(f)
 {
  fprintf(f,"[%s]: %d x %d x %d: %s\r\n",name,N->weights[0].ins,N->weights[0].outs,N->weights[N->weightscount-1].outs,quality);
  fclose(f);
 } 
}

const char*getwrd(const char*sz,char*wrd)
{
 int i=0;
 while(*sz&&(*sz!=' ')&&(*sz!='\t')&&(*sz!='\r')&&(*sz!='\n'))
  wrd[i++]=*sz++;
 wrd[i]=0;
 while((*sz==' ')||(*sz=='\t')) sz++;
 return sz;
}

int ansi_char(const char*s,int*l)
{
 if(l) *l=1;
 return (unsigned char)*s;
}
int ansi_char_is(const char*s,char req,int*l)
{
 int ch=ansi_char(s,l);
 switch(req)
  {
   case 'a':
   return isalpha(ch);
   case 'l':
   return islower(ch);
   case 'u':
   return isupper(ch);
   case 'd':
   return isdigit(ch);
   case 'p':
   return ispunct(ch);
   case 's':
   return isspace(ch);
  }
 return 0;
}
#define char_get(sz,pl)     ansi_char(sz,pl)
#define char_isalpha(sz,pl) char_is(sz,'a',pl)
#define char_isalnum(sz,pl) char_is(sz,'n',pl)
#define char_isupper(sz,pl) char_is(sz,'u',pl)
#define char_islower(sz,pl) char_is(sz,'l',pl)
#define char_isdigit(sz,pl) char_is(sz,'d',pl)
#define char_ispunct(sz,pl) char_is(sz,'p',pl)
#define char_isspace(sz,pl) char_is(sz,'s',pl)
#define char_iscr(sz,pl)    char_is(sz,'r',pl)
#define char_is(s,req,l)    ansi_char_is(s,req,l)
int str_charinset(const char*lemma, const char*set)
{
 int check =char_get(lemma, NULL);
 while (set&&*set)
 {
  int adv,code = char_get(set,&adv);
  if (code == check)
   return 1;
  else
   set += adv;
 }
 return 0;
}
int cpos_normalize(const char*wrd,char*morph)
{
 char m[64]={0};
 int  n;
 if((_memicmp(wrd,"http:/",6)==0)||(_memicmp(wrd,"https:/",7)==0)||(_memicmp(wrd,"www.",4)==0))
  strcpy(m,"$URL$");
 else
  if(char_isdigit(wrd,&n))
   {
    int l=n,dots=0;
    while(wrd[l])
     if(char_isdigit(wrd+l,&n))
      l+=n;
     else
      if(((wrd[l]=='.')||(wrd[l]==','))&&char_isdigit(wrd+l+1,&n))
       {l+=1;dots++;}
      else
       break;
    if(wrd[l]==0)
     if(dots)
      strcpy(m,"$NUMDOTS$");
     else
     switch(l)
       {
      case 1:
       strcpy(m,"$D$");
       break;
      case 2:
       strcpy(m,"$DD$");
       break;
      case 3:
       strcpy(m,"$DDD$");
       break;
      case 4:
       strcpy(m,"$DDDD$");
       break;
      default:
        strcpy(m,"$NUM$");
       }     
   }
 if(morph)
  strcpy(morph,m);
 return *m;
}
int cpos_morph(const char*wrd,char*morph,int end)
{
 int i=0,j=0,fend=0;
 while(wrd[i])
 {
  int a,c=char_get(wrd+i,&a),code='_';
  if(char_isdigit(wrd+i,NULL))
   code='0';
  else
  if(char_isupper(wrd+i,NULL))
   code='A';
  else
   if(char_islower(wrd+i,NULL))
    code='a';
   else
    if(char_ispunct(wrd+i,NULL))
     if(str_charinset(wrd+i,".,:;+-=?!()[]/\\&%$£\"'@"))
      code=c;
     else
      code='^';
    else
     code='_';
  if(j<24)
   {
    if(end<0)
    {
     if(j&&(morph[j-1]==code))
      ;
     else
      morph[j++]=code;
    }
    else
     if((j>4)&&(morph[j-1]==code)&&(morph[j-2]==code))
      ;
     else
      morph[j++]=code;
   }
  i+=a;  
  if(end&&(strlen(wrd+i)==end))
   {
    fend=1;
    break;
   }
 }
 morph[j]=0;
 if(end==3)
  if(fend)
   {
    morph[j++]='+';
    strcpy(morph+j,wrd+i);
   }
  else
   return 0;
 else
 if(end==2)
  if(fend)
   {
    morph[j++]='+';
    strcpy(morph+j,wrd+i);
   }
  else
   return 0;
 else
 if(end==1)
  if(fend)
   {
    morph[j++]='+';
    strcpy(morph+j,wrd+i);
   }
  else
   return 0;
 return j;
}

int chars_compare(const void*a,const void*b)
{
 return (*(char*)a)-(*(char*)b);
}

int bann_getPOSNER(bann*net,int split,int*pos,int*ner)
{
 int  i,best=-1,hm=net->weights[net->weightscount-1].outs;
 real top,*v=net->weights[net->weightscount-1].o;
 best=-1;
 for(i=0;i<split;i++)
  if((best==-1)||(v[i]>top))
  {
   best=i;
   top=net->weights[net->weightscount-1].o[i];
  }
 if(pos) *pos=best;
 best=-1;
 for(;i<hm;i++)
  if((best==-1)||(v[i]>top))
  {
   best=i;
   top=net->weights[net->weightscount-1].o[i];
  }
 if(ner) *ner=best-split;
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
 int    ichars,charseq,corpusseq,maxlen,window_size,poscnt,nercnt;
 byte   usedchars[MAX_CHARS];
}NERtagger;

void NERtagger_new(NERtagger*ner,dict_t dict,dict_t posdict,dict_t nerdict)
{
 int  e,sum=0,avg=0,top=0;
 char chars[MAX_CHARS];
 dict_t tdict=dict_new();

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
   char wrd[256],tmp[256];
   int  len;
   strcpy(wrd,dict->entry[e].key);
   len=(int)strlen(wrd);   
   if(len>ner->maxlen)
    ner->maxlen=len;
   len=0;
   _strlwr(wrd);
   if(cpos_normalize(wrd,tmp))
    strcpy(wrd,tmp);
   while(wrd[len])
    {
     byte ch=(byte)wrd[len++];
     ner->usedchars[ch]=1;
    }
   if(dict->entry[e].cnt>avg)
    dict_add(ner->cdict,dict->entry[e].key,dict->entry[e].cnt);
   else
    {
     char tmp[256];
     if(cpos_morph(dict->entry[e].key,tmp,3))
      dict_add(tdict,tmp,dict->entry[e].cnt);
     if(cpos_morph(dict->entry[e].key,tmp,2))
      dict_add(tdict,tmp,dict->entry[e].cnt);
     if(cpos_morph(dict->entry[e].key,tmp,1))
      dict_add(tdict,tmp,dict->entry[e].cnt);
     if(cpos_morph(dict->entry[e].key,tmp,0))
      dict_add(tdict,tmp,dict->entry[e].cnt);
     if(cpos_morph(dict->entry[e].key,tmp,-1))
      dict_add(tdict,tmp,dict->entry[e].cnt);
    }
  }

 sum=top=0;
 for(e=0;e<tdict->c;e++)
  {
   if(tdict->entry[e].cnt>top)
    top=tdict->entry[e].cnt;
   sum+=tdict->entry[e].cnt;
  }
 avg=(sum-top)/(tdict->c-1);
 avg=max(2,avg/2);

 for(e=0;e<tdict->c;e++) 
  if(tdict->entry[e].cnt>avg)
    dict_add(ner->cdict,tdict->entry[e].key,tdict->entry[e].cnt);

 dict_free(tdict);


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
 ner->poscnt=posdict->c;
 ner->nercnt=nerdict->c;
 ner->corpusseq=(ner->cdict->c+ner->poscnt+ner->nercnt+2)*(ner->window_size*2+1);

 bann_new(&ner->N,ner->corpusseq,HIDDEN_LAYER,HIDDEN_LAYER_COUNT,ner->poscnt+ner->nercnt);
 bann_setlearningrate(&ner->N,1.0f,bann_optimizer_decay,64*1024);

 if(HIDDEN_LAYER_COUNT>1)
  printf("creating bANN with %d x %d (%d) x %d cells\n",(ner->corpusseq+ner->charseq),HIDDEN_LAYER,HIDDEN_LAYER_COUNT,ner->poscnt+ner->nercnt); 
 else
  printf("creating bANN with %d x %d x %d cells\n",(ner->corpusseq+ner->charseq),HIDDEN_LAYER,ner->poscnt+ner->nercnt); 
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
 int posT;
 int ner;
 int nerT;
}word;
typedef struct{
 word*b;
 int  c,m;
}words;
typedef struct{
 int from,to;
}clause;
typedef struct{
 clause*b;
 int    c,m;
}clauses;
typedef struct{
 words   wrd;
 clauses cl;

 dict_t  dict;
 dict_t  posdict;
 dict_t  nerdict;

}corpus;

void corpus_new(corpus*w)
{
 w->wrd.c=0;w->wrd.m=256;
 w->wrd.b=(word*)malloc(w->wrd.m*sizeof(w->wrd.b[0]));
 
 w->cl.c=0;w->cl.m=256;
 w->cl.b=(clause*)malloc(w->cl.m*sizeof(w->cl.b[0]));

 w->dict=dict_new();
 
 w->posdict=dict_new();
 dict_add(w->posdict,"$BOS$",0);
 dict_add(w->posdict,"$BOS$",0);

 w->nerdict=dict_new();
 dict_add(w->nerdict,"O",0);
}

void corpus_delete(corpus*w)
{
 dict_free(w->dict);
 dict_free(w->posdict);
 dict_free(w->nerdict);
 free(w->wrd.b);
 free(w->cl.b);
}
void corpus_addword(corpus*w,word*ww)
{
 if(w->wrd.c+1==w->wrd.m)
 {
  w->wrd.m*=2;
  w->wrd.b=(word*)realloc(w->wrd.b,w->wrd.m*sizeof(w->wrd.b[0]));
 }
 w->wrd.b[w->wrd.c++]=*ww;
}
void corpus_addclause(corpus*w,int from,int to)
{
 clause c;
 c.from=from;c.to=to;
 if(w->cl.c+1==w->cl.m)
 {
  w->cl.m*=2;
  w->cl.b=(clause*)realloc(w->cl.b,w->cl.m*sizeof(w->cl.b[0]));
 }
 w->cl.b[w->cl.c++]=c;
}
int corpus_read(corpus*w,const char*name)
{
 FILE*f;
 printf("reading %s...\n",name);
 f=fopen(name,"rb");
 if(f)
  {
   int from=0;
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
    {
     if(w->wrd.c-from)
      {
       corpus_addclause(w,from,w->wrd.c);
       from=w->wrd.c;
      }
    }
    else
     {
      next=getwrd(line,wrd);
      ww.fit=dict_add(w->dict,wrd,1);

      next=getwrd(next,pos);
      ww.pos=dict_add(w->posdict,pos,1);

      next=getwrd(next,tmp);
      next=getwrd(next,ner);      
      ww.ner=dict_add(w->nerdict,ner,1);

      corpus_addword(w,&ww);
     }
   }
   if(w->wrd.c-from)
    {
     corpus_addclause(w,from,w->wrd.c);
     from=w->wrd.c;
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

void corpus_remap(corpus*crp,dict_t cdict,corpus*remap)
{
 int e,err=0;
 for(e=0;e<crp->wrd.c;e++)
 {
  if(crp->wrd.b[e].fit!=-1)
   {
    char fit[256],tmp[256];
    int  id;
    strcpy(fit,crp->dict->entry[crp->wrd.b[e].fit].key);
    _strlwr(fit);
    if(cpos_normalize(fit,tmp))
     strcpy(fit,tmp);
    id=dict_find(cdict,fit);
    if(id!=-1)
     crp->wrd.b[e].fitR=id;
    else
     {
      char tmp[256];
      if(cpos_morph(fit,tmp,3)&&((id=dict_find(cdict,tmp))!=-1))
       crp->wrd.b[e].fitR=id;
      else
      if(cpos_morph(fit,tmp,2)&&((id=dict_find(cdict,tmp))!=-1))
       crp->wrd.b[e].fitR=id;
      else
       if(cpos_morph(fit,tmp,1)&&((id=dict_find(cdict,tmp))!=-1))
        crp->wrd.b[e].fitR=id;
       else
        if(cpos_morph(fit,tmp,0)&&((id=dict_find(cdict,tmp))!=-1))
         crp->wrd.b[e].fitR=id;
        else
         if(cpos_morph(fit,tmp,-1)&&((id=dict_find(cdict,tmp))!=-1))
          crp->wrd.b[e].fitR=id;
         else
          crp->wrd.b[e].fitR=-1;
     }
   }
  else
   crp->wrd.b[e].fitR=-1;
  if(remap)
   {
    if(crp->wrd.b[e].pos!=-1)
     {
      const char*spos=crp->posdict->entry[crp->wrd.b[e].pos].key;
      int id=dict_find(remap->posdict,spos);
      if(id!=-1)
       crp->wrd.b[e].pos=id;
      else
       err++;
     }
    if(crp->wrd.b[e].ner!=-1)
     {
      const char*sner=crp->nerdict->entry[crp->wrd.b[e].ner].key;
      int id=dict_find(remap->nerdict,sner);
      if(id!=-1)
       crp->wrd.b[e].ner=id;
      else
       err++;
     }
   }
 }
 if(remap)
  {
   int n;
   dict_free(crp->posdict);
   dict_free(crp->nerdict);
   crp->posdict=dict_new();
   for(n=0;n<remap->posdict->c;n++)
    dict_add(crp->posdict,remap->posdict->entry[n].key,remap->posdict->entry[n].cnt);
   crp->nerdict=dict_new();
   for(n=0;n<remap->nerdict->c;n++)
    dict_add(crp->nerdict,remap->nerdict->entry[n].key,remap->nerdict->entry[n].cnt);
  }
}

void corpus_setinputlayer(NERtagger*ner,corpus*wrd,int cl,int w,bann_io*in)
{
 int    ww,odd=0,pos=0;
 clause*lc=&wrd->cl.b[cl];
 in->idx->hotcnt=0;
 // add context window CASE
 for(ww=-ner->window_size;ww<=ner->window_size;ww++)
  {
   if((w+ww>=lc->from)&&(w+ww<lc->to)&&(wrd->wrd.b[w+ww].fit!=-1))
    {
     const char*fit=wrd->dict->entry[wrd->wrd.b[w].fit].key;
     int        c,up=0,fup=0;
     for(c=0;fit[c];c++)
      {
       byte ch=(byte)fit[c];
       if((ch>='A')&&(ch<='Z'))
        {
         up++;
         if(c==0)
          fup=1;
        }
      }     
     if(fup==1) // Up
      in->idx->hotpos[in->idx->hotcnt++]=pos;
     if(up>1)   // UP
      in->idx->hotpos[in->idx->hotcnt++]=pos+1;
    }
   pos+=2;
  }
 // add context window POS
 for(ww=-ner->window_size;ww<=ner->window_size;ww++)
  {
   if(ww<0)
    if((w+ww>=lc->from)&&(w+ww<lc->to))
     if(wrd->wrd.b[w+ww].posT!=-1)
      in->idx->hotpos[in->idx->hotcnt++]=pos+wrd->wrd.b[w+ww].posT;    
   pos+=ner->poscnt;
   if(ww<0)
    if((w+ww>=lc->from)&&(w+ww<lc->to))
     if(wrd->wrd.b[w+ww].nerT!=-1)
      in->idx->hotpos[in->idx->hotcnt++]=pos+wrd->wrd.b[w+ww].nerT;    
   pos+=ner->nercnt;
  }
 // add context window WORDS / MORPH
 for(ww=-ner->window_size;ww<=ner->window_size;ww++)
  {
   if((w+ww>=lc->from)&&(w+ww<lc->to)&&(wrd->wrd.b[w+ww].fitR!=-1))
    {
     int id=wrd->wrd.b[w+ww].fitR;
     in->idx->hotpos[in->idx->hotcnt++]=pos+id;    
    }
   pos+=ner->cdict->c;
  }
}

void ner_qual(int*good,int*add,int*miss,int from,int hm,real*acc,real*prec,real*recall,real*F1)
{
 real gprec=0,grecall=0,gacc=0;
 int  i=0,ggood=0,gmiss=0,gadd=0,cnt=0,tgood=0,tadd=0,tmiss=0;
 for(i=from;i<hm;i++) // skip O
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

void corpus_predict(NERtagger*ner,bann_io*in,corpus*wrdA,const char*name,const char*outname)
{
 bann*N=&ner->N;
 real acc,prec,recall,F1,F1pos;
 int  cl,goodN[256],missN[256],addN[256],w,cnt=0,hmNER=N->weights[N->weightscount-1].outs; 
 FILE*f=NULL;
 char what[256];

 if(outname)
  f=fopen(outname,"wb");
 
 memset(goodN,0,hmNER*sizeof(goodN[0]));
 memset(missN,0,hmNER*sizeof(missN[0]));
 memset(addN,0,hmNER*sizeof(addN[0]));

 for(cl=0;cl<wrdA->cl.c;cl++)
  {
   clause*lc=&wrdA->cl.b[cl];
   for(w=lc->from;w<lc->to;w++)
    {
     corpus_setinputlayer(ner,wrdA,cl,w,in);           
     if(in->idx->hotcnt)
      {
       int        selPOS,correctPOS=wrdA->wrd.b[w].pos;
       int        selNER,correctNER=wrdA->wrd.b[w].ner;
       const char*fit=wrdA->dict->entry[wrdA->wrd.b[w].fit].key;
       const char*szpos=wrdA->posdict->entry[correctPOS].key;
       const char*szner=wrdA->nerdict->entry[correctNER].key;
       
       bann_predict(N,in);

       bann_getPOSNER(N,ner->poscnt,&selPOS,&selNER);

       if(selNER==correctNER)
        goodN[selNER]++;
       else
        {
         addN[selNER]++;
         missN[correctNER]++;
        }       

       if(selPOS==correctPOS)
        goodN[ner->nercnt+selPOS]++;
       else
        {
         addN[ner->nercnt+selPOS]++;
         missN[ner->nercnt+correctPOS]++;
        }   

       if(f)
        if((selPOS==correctPOS)&&(selNER==correctNER))
         fprintf(f,"%s\t%s\t%s\n",fit,szpos,szner);
        else
         fprintf(f,"%s\t%s\t%s\t%s\t%s\n",fit,szpos,szner,wrdA->posdict->entry[selPOS].key,wrdA->nerdict->entry[selNER].key);
       
       wrdA->wrd.b[w].posT=selPOS;
       wrdA->wrd.b[w].nerT=selNER;

       cnt++;
       if((cnt&0xFF)==0xFF)
       {
        ner_qual(goodN,addN,missN,1,ner->nercnt,&acc,&prec,&recall,&F1);
        ner_qual(goodN,addN,missN,ner->nercnt,ner->nercnt+ner->poscnt,NULL,NULL,NULL,&F1pos);
        printf("[%s] NER: F1: %.2f P: %.2f R: %.2f - POS: %.2f \r",name,F1,prec,recall,F1pos);
       }
      }
    }
  }
 ner_qual(goodN,addN,missN,1,ner->nercnt,&acc,&prec,&recall,&F1);
 ner_qual(goodN,addN,missN,ner->nercnt,ner->nercnt+ner->poscnt,NULL,NULL,NULL,&F1pos);
 sprintf(what,"NER: F1: %.2f P: %.2f R: %.2f - POS: %.2f ",F1,prec,recall,F1pos);
 printf("[%s] %s \n",name,what);
 history_add(&ner->N,"bin/bann.log",name,what);

 if(f)
  fclose(f);
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
   // ---- dict_emit("NER",train.nerdict);
   // ---- dict_emit("POS",train.posdict);
   NERtagger_new(&NER,train.dict,train.posdict,train.nerdict);
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

 index=(int*)malloc(train.cl.c*sizeof(int));
 for(e=0;e<train.cl.c;e++)
  index[e]=e;

 for(e=0;e<MAX_EPOCHS;e++)
  {
   real err=0;
   int  cnt=0,cl,w,goodN[256],addN[256],missN[256],hmNER=NER.N.weights[NER.N.weightscount-1].outs;
   real acc,prec,recall,F1,F1pos;

   memset(goodN,0,hmNER*sizeof(goodN[0]));
   memset(missN,0,hmNER*sizeof(missN[0]));
   memset(addN,0,hmNER*sizeof(addN[0]));

   shuffle(index,train.cl.c);
   for(cl=0;cl<train.cl.c;cl++)
    {
     int    shuffledcl=index[cl];
     clause*lc=&train.cl.b[shuffledcl];
     if(lc->from>=lc->to)
      cnt=0;
     for(w=lc->from;w<lc->to;w++)
      {       
       corpus_setinputlayer(&NER,&train,shuffledcl,w,&in);               
       if(in.idx->hotcnt)
        {
         int        selPOS,correctPOS=train.wrd.b[w].pos;
         int        selNER,correctNER=train.wrd.b[w].ner;
         const char*fit=train.dict->entry[train.wrd.b[w].fit].key;
         const char*szpos=train.posdict->entry[correctPOS].key;
         const char*szner=train.nerdict->entry[correctNER].key;
         
         out.idx->hotcnt=2;
         out.idx->hotpos[0]=train.wrd.b[w].pos;
         out.idx->hotpos[1]=NER.poscnt+train.wrd.b[w].ner;
         //bann_predict(N,in);
         err+=bann_train(&NER.N,&in,&out);

         bann_getPOSNER(&NER.N,NER.poscnt,&selPOS,&selNER);

         if(selNER==correctNER)
          goodN[selNER]++;
         else
          {
           addN[selNER]++;
           missN[correctNER]++;
          }       

         if(selPOS==correctPOS)
          goodN[NER.nercnt+selPOS]++;
         else
          {
           addN[NER.nercnt+selPOS]++;
           missN[NER.nercnt+correctPOS]++;
          }       
         
         train.wrd.b[w].posT=selPOS;
         train.wrd.b[w].nerT=selNER;

         cnt++;
         if((cnt&0xFF)==0xFF)
         {
          ner_qual(goodN,addN,missN,1,NER.nercnt,&acc,&prec,&recall,&F1);
          ner_qual(goodN,addN,missN,NER.nercnt,NER.nercnt+NER.poscnt,NULL,NULL,NULL,&F1pos);
          printf("[%s.%d] NER: F1: %.2f P: %.2f R: %.2f - POS: %.2f - MSE: %.2f  \r","eng.train",e+1,F1,prec,recall,F1pos,err/cnt);
         }
        }
      }
    }
   ner_qual(goodN,addN,missN,1,NER.nercnt,&acc,&prec,&recall,&F1);
   ner_qual(goodN,addN,missN,NER.nercnt,NER.nercnt+NER.poscnt,NULL,NULL,NULL,&F1pos);
   printf("[%s.%d] NER: F1: %.2f P: %.2f R: %.2f - POS: %.2f - MSE: %.2f  \n","eng.train",e+1,F1,prec,recall,F1pos,err/cnt);
   err/=cnt;

   if((err<0.05f)||(e+1==MAX_EPOCHS))
    {
     corpus_predict(&NER,&in,&testA,"eng.testa","bin/eng.testa.log");
     corpus_predict(&NER,&in,&testB,"eng.testb","bin/eng.testb.log");    
     if(err<lasterr)
     {
      bann_write(&NER.N,"bin/ner.bann",0);
      bann_write(&NER.N,"bin/ner.bannQ",1);
      lasterr=err;
      if(err<MSE_STOP)
       break;
     }
     if(NER.N.opt.optimize==bann_optimizer_decay)
      NER.N.opt.optimize=bann_optimizer_adaptive;
    }
  }

 NERtagger_delete(&NER);

 free(index);

 corpus_delete(&train);
 corpus_delete(&testA);
 corpus_delete(&testB);

 return 1;
}