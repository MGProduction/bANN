#include <stdlib.h>
#include <stdio.h>
#include <ctype.h>

#define BANN_IMPLEMENTATION
#define BANN_DEBUG
//#define BANN_RNN
#include "bann.h"

// ----------------------------------
// NER/POS tagger configuration
// ----------------------------------

#define WINDOW_SIZE         5
#define HIDDEN_LAYER       64
#define HIDDEN_LAYER_COUNT  1
#if defined(BANN_RNN)
#define MAX_EPOCHS         64
#else
#define MAX_EPOCHS         32
#endif
#define MSE_STOP       0.001f

#define MAX_ENDLENGTH       3

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
int cpos_safemorph(const char*wrd,char*morph)
{
 int  i=0,m=0,j,mask=0;
 char codes[]="0Aa_^+",ch=0;
 while(wrd[i])
  {
   int a,c=char_get(wrd+i,&a);
   if(char_isdigit(wrd+i,NULL))
    {mask|=1;if(ch==0) ch='0';}
   else
   if(char_isupper(wrd+i,NULL))
    {mask|=2;if(ch==0) ch='A';}
   else
    if(char_islower(wrd+i,NULL))
     {mask|=4;if(ch==0) ch='a';}
    else
     if(char_ispunct(wrd+i,NULL))
      if(str_charinset(wrd+i,".,:;+-=?!()[]/\\&%$£\"'@"))
       {mask|=32;if(ch==0) ch=c;}
      else
       {mask|=16;if(ch==0) ch=c;}
     else
      {mask|=8;if(ch==0) ch=c;}
   i+=a;
  }
 if(mask)
  {
   morph[m++]='$';
   if(ch)
    morph[m++]=ch;
   for(i=0,j=1;codes[i];i++,j*=2)
    if(mask&j)
     morph[m++]=codes[i];
  }
 morph[m]=0;
 return m;
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
     if((j>3)&&(morph[j-1]==code)&&(morph[j-2]==code))
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

int bann_getPOSCHUNKNER(bann*net,int*start,int*sel,int rcnt)
{
 int  i,best=-1,r;
 real top,*v=net->weights[net->weightscount-1].o;
 for(r=0;r<rcnt;r++)
  {
   best=-1;
   for(i=start[r];i<start[r+1];i++)
    if((best==-1)||(v[i]>top))
    {
     best=i;
     top=net->weights[net->weightscount-1].o[i];
    }
   if(sel) sel[r]=best-start[r];
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

int dictentry_compare(const void*a,const void*b)
{
 dict_entry_s*A=(dict_entry_s*)a;
 dict_entry_s*B=(dict_entry_s*)b;
 return B->cnt-A->cnt;
}

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
 int    ichars,charseq,corpusseq,maxlen,window_size,poscnt,chunkcnt,nercnt;
 byte   usedchars[MAX_CHARS];
}NERtagger;

void NERtagger_new(NERtagger*ner,dict_t dict,dict_t*tagdict)
{
 int  e,sum=0,avg=0,top=0,avg2,ttop;
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
 avg2=max(2,avg/4);
 ttop=top*80/100;

 if(0)
  if(dict->c>6*1024)
   avg2=dict->entry[6*1024].cnt;
 
 ner->usedchars[2]=1;
 for(e=0;e<dict->c;e++) 
  {
   char wrd[256],tmp[256];
   int  len,n=0;
   strcpy(wrd,dict->entry[e].key);
   len=(int)strlen(wrd);   
   if(len>ner->maxlen)
    ner->maxlen=len;
   len=0;
   _strlwr(wrd);
   if(cpos_normalize(wrd,tmp))
    {strcpy(wrd,tmp);n=1;}
   while(wrd[len])
    {
     byte ch=(byte)wrd[len++];
     ner->usedchars[ch]=1;
    }
   if(((dict->entry[e].cnt>avg2)/*&&char_isalpha(wrd,NULL)*/)||n)
    dict_add(ner->cdict,wrd,dict->entry[e].cnt);

   if(n==0)
    if(dict->entry[e].cnt<ttop)
     {
      char tmp[256];
      if(1)
       {
        if(cpos_morph(dict->entry[e].key,tmp,3))
         dict_add(tdict,tmp,1);
        if(cpos_morph(dict->entry[e].key,tmp,2))
         dict_add(tdict,tmp,1);
        if(cpos_morph(dict->entry[e].key,tmp,1))
         dict_add(tdict,tmp,1);
        if(cpos_morph(dict->entry[e].key,tmp,0))
         dict_add(tdict,tmp,1);
        if(cpos_morph(dict->entry[e].key,tmp,-1))
         dict_add(tdict,tmp,1);
        if(cpos_safemorph(wrd,tmp))
         dict_add(tdict,tmp,1);
       }
      else
       {
        int  end,id=-1;
        for(end=MAX_ENDLENGTH;end>=-1;end--)
         if(cpos_morph(wrd,tmp,end))
          dict_add(tdict,tmp,1);
        if(cpos_safemorph(wrd,tmp))
         dict_add(tdict,tmp,1);
       }
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

 qsort(tdict->entry,tdict->c,sizeof(tdict->entry[0]),dictentry_compare);

 if(tdict->c>1*1024)
  avg=tdict->entry[1*1024].cnt;

 for(e=0;e<tdict->c;e++) 
  if(tdict->entry[e].cnt>avg)
   dict_add(ner->cdict,tdict->entry[e].key,tdict->entry[e].cnt);
  else
   break;

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
 
 ner->poscnt=tagdict[0]->c;
 ner->chunkcnt=tagdict[1]->c;
 ner->nercnt=tagdict[2]->c;
 
 ner->corpusseq=((ner->cdict->c+1)+ner->poscnt+ner->chunkcnt+ner->nercnt+3)*(ner->window_size*2+1);

 bann_new(&ner->N,ner->corpusseq,HIDDEN_LAYER,HIDDEN_LAYER_COUNT,ner->poscnt+ner->chunkcnt+ner->nercnt);
#if defined(BANN_RNN)
 bann_setlearningrate(&ner->N,0.5f,bann_optimizer_adaptiveslow,96*1024);
#else
 bann_setlearningrate(&ner->N,1.0f,bann_optimizer_decay,96*1024);
#endif

 if(HIDDEN_LAYER_COUNT>1)
  printf("creating bANN with %d x %d (%d) x %d cells\n",ner->corpusseq,HIDDEN_LAYER,HIDDEN_LAYER_COUNT,ner->poscnt+ner->chunkcnt+ner->nercnt); 
 else
  printf("creating bANN with %d x %d x %d cells\n",ner->corpusseq,HIDDEN_LAYER,ner->poscnt+ner->chunkcnt+ner->nercnt); 
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
 int tag[3],tagT[3];
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
 dict_t  tagdict[3];

}corpus;

void corpus_new(corpus*w)
{
 int i;

 w->wrd.c=0;w->wrd.m=256;
 w->wrd.b=(word*)malloc(w->wrd.m*sizeof(w->wrd.b[0]));
 
 w->cl.c=0;w->cl.m=256;
 w->cl.b=(clause*)malloc(w->cl.m*sizeof(w->cl.b[0]));

 w->dict=dict_new();
 
 for(i=0;i<3;i++)
  w->tagdict[i]=dict_new();

 dict_add(w->tagdict[2],"O",0); // added to be sure it's the FIRST
}

void corpus_delete(corpus*w)
{
 int i;
 dict_free(w->dict);
 for(i=0;i<3;i++)
  dict_free(w->tagdict[i]);
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
    char       line[1024],wrd[256],tag[256];
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

      for(i=0;i<3;i++)
       {
        next=getwrd(next,tag);
        ww.tag[i]=dict_add(w->tagdict[i],tag,1);
       }

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
      int  end=5,id=-1;
      if(1)
       {
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
        if(cpos_safemorph(fit,tmp)&&((id=dict_find(cdict,tmp))!=-1))
         crp->wrd.b[e].fitR=id;
        else
         crp->wrd.b[e].fitR=-1;
       }
      else
       {
       for(end=MAX_ENDLENGTH;end>=-1;end--)
        if(cpos_morph(fit,tmp,end)&&((id=dict_find(cdict,tmp))!=-1))
         break;
       if(id!=-1)
        crp->wrd.b[e].fitR=id;
       else
        if(cpos_safemorph(fit,tmp)&&((id=dict_find(cdict,tmp))!=-1))
         crp->wrd.b[e].fitR=id;
        else
         crp->wrd.b[e].fitR=-1;
       }
     }
   }
  else
   crp->wrd.b[e].fitR=-1;
  if(remap)
   {
    int i;
    for(i=0;i<3;i++)
     if(crp->wrd.b[e].tag[i]!=-1)
      {
       const char*spos=crp->tagdict[i]->entry[crp->wrd.b[e].tag[i]].key;
       int id=dict_find(remap->tagdict[i],spos);
       if(id!=-1)
        crp->wrd.b[e].tag[i]=id;
       else
        err++;
      }
   }
 }
 if(remap)
  {
   int n,i;
   for(i=0;i<3;i++)
    {
     dict_free(crp->tagdict[i]);
     crp->tagdict[i]=dict_new();
     for(n=0;n<remap->tagdict[i]->c;n++)
      dict_add(crp->tagdict[i],remap->tagdict[i]->entry[n].key,remap->tagdict[i]->entry[n].cnt);
    }
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
     int        c,up=0,fup=0,low=0,dig=0;
     for(c=0;fit[c];c++)
      {
       byte ch=(byte)fit[c];
       if((ch>='A')&&(ch<='Z'))
        {
         up++;
         if(c==0)
          fup=1;
        }
       else
        if((ch>='a')&&(ch<='z'))
         low++;
        else
         if((ch>='0')&&(ch<='9'))
          dig++;
      }     
     if(fup==1) // Up
      in->idx->hotpos[in->idx->hotcnt++]=pos;
     if(up>1)   // UP
      in->idx->hotpos[in->idx->hotcnt++]=pos+1;
     if(dig)   // UP
      in->idx->hotpos[in->idx->hotcnt++]=pos+2;
    }
   pos+=3;
  }
 // add context window POS
 for(ww=-ner->window_size;ww<=ner->window_size;ww++)
  {
   if(ww<0)
    if((w+ww>=lc->from)&&(w+ww<lc->to))
     if(wrd->wrd.b[w+ww].tagT[0]!=-1)
      in->idx->hotpos[in->idx->hotcnt++]=pos+wrd->wrd.b[w+ww].tagT[0];    
   pos+=ner->poscnt;
   if(ww<0)
    if((w+ww>=lc->from)&&(w+ww<lc->to))
     if(wrd->wrd.b[w+ww].tagT[1]!=-1)
      in->idx->hotpos[in->idx->hotcnt++]=pos+wrd->wrd.b[w+ww].tagT[1];    
   pos+=ner->chunkcnt;
   if(ww<0)
    if((w+ww>=lc->from)&&(w+ww<lc->to))
     if(wrd->wrd.b[w+ww].tagT[2]!=-1)
      in->idx->hotpos[in->idx->hotcnt++]=pos+wrd->wrd.b[w+ww].tagT[2];    
   pos+=ner->nercnt;
  }
 // add context window WORDS / MORPH
 for(ww=-ner->window_size;ww<=ner->window_size;ww++)
  {
   int id=0;
   if((w+ww>=lc->from)&&(w+ww<lc->to)&&(wrd->wrd.b[w+ww].fitR!=-1))
    id=wrd->wrd.b[w+ww].fitR+1;
   in->idx->hotpos[in->idx->hotcnt++]=pos+id;    
   pos+=ner->cdict->c;
  }
}

void ner_qual(int*good,int*add,int*miss,int*start,real*F1,int rcnt)
{
 int r;
 for(r=0;r<rcnt;r++)
  {
   real gprec=0,grecall=0,gacc=0;
   int  i=0,ggood=0,gmiss=0,gadd=0,cnt=0,tgood=0,tadd=0,tmiss=0;
   for(i=start[r]+(r==2);i<start[r+1];i++) // skip O
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
   //if(acc) *acc=gacc;
   //if(prec) *prec=gprec;
   //if(recall) *recall=grecall;
   if(F1) F1[r]=2*gprec*grecall/(gprec+grecall);
  }
}

real corpus_predict(NERtagger*ner,bann_io*in,corpus*wrdA,const char*name,int trainingepochs,const char*outname)
{
 bann*N=&ner->N;
 real F1[3];
 int  cl,goodN[256],missN[256],addN[256],w,cnt=0,hmNER=N->weights[N->weightscount-1].outs,good=0,bad=0; 
 FILE*f=NULL;
 int  start[4];
 char what[256];
 char tagname[3][8]={"POS","CHUNK","NER"};
 dict_t errorlist;
 
 start[0]=0;
 start[1]=start[0]+ner->poscnt;
 start[2]=start[1]+ner->chunkcnt;
 start[3]=start[2]+ner->nercnt; 

 if(outname)
 {
  f=fopen(outname,"wb");
  errorlist=dict_new();
 }
 
 memset(goodN,0,hmNER*sizeof(goodN[0]));
 memset(missN,0,hmNER*sizeof(missN[0]));
 memset(addN,0,hmNER*sizeof(addN[0]));

 for(cl=0;cl<wrdA->cl.c;cl++)
  {
   clause*lc=&wrdA->cl.b[cl];
#if defined(BANN_RNN)
   bann_startrnn(&ner->N);
#endif
   for(w=lc->from;w<lc->to;w++)
    {
     corpus_setinputlayer(ner,wrdA,cl,w,in);           
     if(in->idx->hotcnt)
      {
       int        g,ii,sel[3],correct[3];
       const char*fit=wrdA->dict->entry[wrdA->wrd.b[w].fit].key;

       for(ii=0;ii<3;ii++)
        correct[ii]=wrdA->wrd.b[w].tag[ii];
                           
       bann_predict(N,in);

       bann_getPOSCHUNKNER(&ner->N,&start[0],&sel[0],3);

       for(g=ii=0;ii<3;ii++)
        if(sel[ii]==correct[ii])
         {goodN[start[ii]+sel[ii]]++;g++;}
        else
         {
          addN[start[ii]+sel[ii]]++;
          missN[start[ii]+correct[ii]]++;
         }   

       if(g==ii)
        good++;
       else
        bad++;
                
       for(ii=0;ii<3;ii++)
        wrdA->wrd.b[w].tagT[ii]=sel[ii];

       if(f)
        if((sel[0]==correct[0])&&(sel[1]==correct[1])&&(sel[2]==correct[2]))
         fprintf(f,"%s\t%s\t%s\t%s\n",fit,wrdA->tagdict[0]->entry[correct[0]].key,wrdA->tagdict[1]->entry[correct[1]].key,wrdA->tagdict[2]->entry[correct[2]].key);
        else
         {
          char out[256];
          for(ii=0;ii<3;ii++)
           if(sel[ii]!=correct[ii])
            {
             sprintf(out,"%s\t%s\t%s",tagname[ii],wrdA->tagdict[ii]->entry[correct[ii]].key,wrdA->tagdict[ii]->entry[sel[ii]].key);
             dict_add(errorlist,out,1);
            }
          fprintf(f,"%s\t%s\t%s\t%s\t%s\t%s\t%s ***\n",fit,wrdA->tagdict[0]->entry[correct[0]].key,wrdA->tagdict[1]->entry[correct[1]].key,wrdA->tagdict[2]->entry[correct[2]].key,
                                                           wrdA->tagdict[0]->entry[sel[0]].key,wrdA->tagdict[1]->entry[sel[1]].key,wrdA->tagdict[2]->entry[sel[2]].key);
         }
       
       cnt++;
       if((cnt&0xFF)==0xFF)
       {
        ner_qual(goodN,addN,missN,start,F1,3);
        printf("[%s] Q: %.2f - POS: %.2f CHUNK: %.2f NER: %.2f \r",name,(float)good*100/(float)(good+bad),F1[0],F1[1],F1[2]);
       }
      }
    }
   if(f)
    fprintf(f,"\n");
  }
 ner_qual(goodN,addN,missN,start,F1,3);
 sprintf(what,"[%s] Q: %.2f - POS: %.2f CHUNK: %.2f NER: %.2f",name,(float)good*100/(float)(good+bad),F1[0],F1[1],F1[2]);
 printf("%s \n",what);

 if(f)
 {
  sprintf(what,"EPOCHS: %d [%d] - Q: %.2f - POS: %.2f CHUNK: %.2f NER: %.2f",trainingepochs,ner->N.opt.optimize,(float)good*100/(float)(good+bad),F1[0],F1[1],F1[2]);  
  history_add(&ner->N,"bin/bann.log",name,what);
 }

 if(f)
  {
   int i,ii;
   fprintf(f,"\n[ERRORS]\n");
   qsort(errorlist->entry,errorlist->c,sizeof(errorlist->entry[0]),dictentry_compare);
   for(i=0;i<errorlist->c;i++)
    fprintf(f,"%s\t%d\n",errorlist->entry[i].key,errorlist->entry[i].cnt);
   dict_free(errorlist);
   fprintf(f,"\n[RESULTS]\n");
   fprintf(f,"TAG\tF1\tPREC\tRECALL\tGOOD\tADD\tMISS\n");
   for(ii=0;ii<3;ii++)
    {
     for(i=0;i<wrdA->tagdict[ii]->c;i++)
     {
      int   good=goodN[start[ii]+i],miss=missN[start[ii]+i],add=addN[start[ii]+i];
      if(good)
       {
        float prec=good*100.0f/(good+add);
        float recall=good*100.0f/(good+miss);
        float F1=2*prec*recall/(prec+recall);
        fprintf(f,"%s\t%.2f\t%.2f\t%.2f\t%d\t%d\t%d\n",wrdA->tagdict[ii]->entry[i].key,F1,prec,recall,good,add,miss);
       }
     }
     fprintf(f,"\n");
    }
   fclose(f);
  }
 return (float)good*100/(float)(good+bad);
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
 corpus    train,dev,test; 
 int       e,inw[256],o[8],*index;   
 real      lasterr=100,prevdevqual=0;
 bann_idx  a,b;
 bann_io   in,out;
 int       start[4];

 // read TRAIN corpus
 corpus_new(&train);
 if(corpus_read(&train,"bin/data/ner/eng.train"))
  {
   // create BANN with this data
   NERtagger_new(&NER,train.dict,&train.tagdict[0]);
   corpus_remap(&train,NER.cdict,NULL);
  }
 
 // read DEV corpus
 corpus_new(&dev);
 if(corpus_read(&dev,"bin/data/ner/eng.dev"))
  {
   corpus_remap(&dev,NER.cdict,&train);
  }

 // read TEST corpus
 corpus_new(&test);
 if(corpus_read(&test,"bin/data/ner/eng.test"))
  {
   corpus_remap(&test,NER.cdict,&train);
  }
 
 // set INPUT&OUTPUT bann_io
 in.kind=idx;in.idx=&a;
 a.hotpos=&inw[0];

 out.kind=idx;out.idx=&b;
 b.hotpos=&o[0];

 // create index array (for shuffle)
 index=(int*)malloc(train.cl.c*sizeof(int));
 for(e=0;e<train.cl.c;e++)
  index[e]=e;

 start[0]=0;
 start[1]=start[0]+NER.poscnt;
 start[2]=start[1]+NER.chunkcnt;
 start[3]=start[2]+NER.nercnt; 

 // train BANN
 for(e=0;e<MAX_EPOCHS;e++)
  {
   real err=0;
   int  cnt=0,cl,w,goodN[256],addN[256],missN[256],hmOUT=NER.N.weights[NER.N.weightscount-1].outs,good=0,bad=0;
   real F1[3];

   memset(goodN,0,hmOUT*sizeof(goodN[0]));
   memset(missN,0,hmOUT*sizeof(missN[0]));
   memset(addN,0,hmOUT*sizeof(addN[0]));

   shuffle(index,train.cl.c);
   for(cl=0;cl<train.cl.c;cl++)
    {
     int    shuffledcl=index[cl];
     clause*lc=&train.cl.b[shuffledcl];
     if(lc->from>=lc->to)
      cnt=0;
#if defined(BANN_RNN)
     bann_startrnn(&NER.N);
#endif
     for(w=lc->from;w<lc->to;w++)
      {       
       corpus_setinputlayer(&NER,&train,shuffledcl,w,&in);               
       if(in.idx->hotcnt)
        {
         int        g,ii,sel[3],correct[3];
         const char*fit=train.dict->entry[train.wrd.b[w].fit].key;
         const char*spos=train.tagdict[0]->entry[train.wrd.b[w].tag[0]].key;
         const char*schunk=train.tagdict[1]->entry[train.wrd.b[w].tag[1]].key;
         const char*sner=train.tagdict[2]->entry[train.wrd.b[w].tag[2]].key;

         for(ii=0;ii<3;ii++)
          correct[ii]=train.wrd.b[w].tag[ii];
                  
         for(ii=0;ii<3;ii++)
          out.idx->hotpos[ii]=start[ii]+train.wrd.b[w].tag[ii];
         out.idx->hotcnt=ii;
         
         err+=bann_train(&NER.N,&in,&out);

         bann_getPOSCHUNKNER(&NER.N,&start[0],&sel[0],3);

         for(g=ii=0;ii<3;ii++)
          if(sel[ii]==correct[ii])
           {goodN[start[ii]+sel[ii]]++;g++;}
          else
           {
            addN[start[ii]+sel[ii]]++;
            missN[start[ii]+correct[ii]]++;
           }   

         if(g==ii)
          good++;
         else
          bad++;
                  
         for(ii=0;ii<3;ii++)
          train.wrd.b[w].tagT[ii]=sel[ii];

         cnt++;
         if((cnt&0xFF)==0xFF)
         {
          ner_qual(goodN,addN,missN,start,F1,3);
          printf("[%s.%d] Q: %.2f - POS: %.2f CHUNK: %.2f NER: %.2f - MSE: %.2f  \r","eng.train",e+1,(float)good*100/(float)(good+bad),F1[0],F1[1],F1[2],err/cnt);
         }
        }
      }
    }
   ner_qual(goodN,addN,missN,start,F1,3);
   printf("[%s.%d] Q: %.2f - POS: %.2f CHUNK: %.2f NER: %.2f - MSE: %.2f  \n","eng.train",e+1,(float)good*100/(float)(good+bad),F1[0],F1[1],F1[2],err/cnt);
   err/=cnt;

   // when ready, check quality (and save)
   if((err<0.05f)||(e+1==MAX_EPOCHS))
    {
     real devqual;
     if(e+1==MAX_EPOCHS)
      devqual=corpus_predict(&NER,&in,&dev,"eng.dev",MAX_EPOCHS,"bin/eng.dev.log");
     else
      devqual=corpus_predict(&NER,&in,&dev,"eng.dev",-1,NULL);
     if(devqual<prevdevqual)
      NER.N.opt.optimize=bann_optimizer_decay;
     else
      NER.N.opt.optimize=bann_optimizer_off;
     prevdevqual=devqual;
    }
   if((err<0.02f)||(e+1==MAX_EPOCHS))
    if(e+1==MAX_EPOCHS)
     corpus_predict(&NER,&in,&test,"eng.test",MAX_EPOCHS,"bin/eng.test.log");         
    else
     corpus_predict(&NER,&in,&test,"eng.test",-1,NULL);         
   if(err<lasterr)
    {
     bann_write(&NER.N,"bin/ner.bann",0);
     bann_write(&NER.N,"bin/ner.bannQ",1);
     lasterr=err;
     if(err<MSE_STOP)
      break;
    }     
   // change optimization when lower than 0.05f
   /*if(err<0.05f)
     if(NER.N.opt.optimize==bann_optimizer_decay)
      NER.N.opt.optimize=bann_optimizer_adaptive;*/
  }

 NERtagger_delete(&NER);

 free(index);

 corpus_delete(&train);
 corpus_delete(&dev);
 corpus_delete(&test);

 return 1;
}