#include <parsing.h>
#include <display.h>
#include <cstdio>
#include <iostream>
using namespace std;

vector<Object*> allObjects;
vector<Mesh*> allMeshes;
vector<Light*> allLights;
map<std::string,Material*> allMaterials;
map<Triangle*,DTriangle*> t_map;


#define DINF 64
#define DEPSILON 0.001
#define ushort unsigned short
#define ubyte unsigned char
#define uint unsigned int
//#define DEBUG

extern ushort W,H;

// Kernel functions

__device__ float minf(float a, float b) { return a<b? a:b;}
__device__ float maxf(float a, float b) { return a>b? a:b;}

__device__ void iniVec(float v[3], float a, float b, float c) {
  v[0]=a; v[1]=b; v[2]=c;
}

__device__ void copyVec(float v[3], const float v1[3]) {
  v[0]=v1[0]; v[1]=v1[1]; v[2]=v1[2];
}

__device__ void printVec(const float v[3]) {
  printf("%f %f %f\n", v[0], v[1], v[2]);
}

__device__ float dotProd(const float v1[3], const float v2[3]){
  return v1[0]*v2[0]+v1[1]*v2[1]+v1[2]*v2[2];
}

__device__ float* norm(float v[3]) { float M = normf(3,v);
  v[0]/=M; v[1]/=M; v[2]/=M; return v;
}

__device__ float* subf(float v[3], const float v1[3],
    const float v2[3]) { v[0]=v1[0]-v2[0];
  v[1]=v1[1]-v2[1]; v[2]=v1[2]-v2[2]; return v;
}

__device__ float* sumf(float v[3], const float v1[3],
    const float v2[3]) { v[0]=v1[0]+v2[0]; 
  v[1]=v1[1]+v2[1]; v[2]=v1[2]+v2[2]; return v;
}

__device__ float* mulf(float v[3], const float v1[3],
    const float v2[3]) { v[0]=v1[0]*v2[0]; 
  v[1]=v1[1]*v2[1]; v[2]=v1[2]*v2[2]; return v;
}

__device__ float* upscaf(float v[3], const float v1[3], float m) {
   v[0]=v1[0]*m; v[1]=v1[1]*m; v[2]=v1[2]*m; return v;
}

__device__ float* doscaf(float v[3], const float v1[3], float m) {
  v[0]=v1[0]/m; v[1]=v1[1]/m; v[2]=v1[2]/m; return v;
}

__device__ float* expv(float v[3], const float v1[3]) {
  v[0]=expf(v1[0]); v[1]=expf(v1[1]); v[2]=expf(v1[2]); return v;
}

__device__ float* cross(float v[3], const float v1[3],
    const float v2[3]) { v[0]=v1[2]*v2[1]-v1[1]*v2[2];
  v[1]=v1[0]*v2[2]-v1[2]*v2[0]; v[2]=v1[1]*v2[0]-v1[0]*v2[1];
  return v;
}

__device__ float* refractf(float rr[3], const float d[3],
    const float n[3], float rori) {

  float costhe = -dotProd(d,n), sinthe = sqrtf(1-powf(costhe,2)),
    sinfi = rori*sinthe, cosfi = sqrtf(1-powf(sinfi,2)),b[3],
    N1[3], N2[3];

  upscaf(N1,n,costhe); sumf(N1,N1,d); doscaf(b,N1,sinthe);
  upscaf(b,b,sinfi); upscaf(N2,n,cosfi); subf(rr,b,N2); return rr;
}

__device__ float* linCom(float v[3], const float v1[3],
    const float v2[3], const float v3[3], const float s[3]) {
  v[0]=s[0]*v1[0]+s[1]*v2[0]+s[2]*v3[0];
  v[1]=s[0]*v1[1]+s[1]*v2[1]+s[2]*v3[1];
  v[2]=s[0]*v1[2]+s[1]*v2[2]+s[2]*v3[2];
  return v;
}

__device__ void initStack(DStack *stack) {
  stack->size = 0;
}

__device__ void copyRay(DRay *copy, DRay *ray) {
  for (int i=0; i<3; i++) {
    copy->eye[i] = ray->eye[i];
    copy->dir[i] = ray->dir[i];
    copy->att[i] = ray->att[i];
    copy->level[i] = ray->level[i];
    copy->col[i] = ray->col[i];
  }
  copy->s      = ray->s;
  copy->tri    = ray->tri;
  copy->oct    = ray->oct;
  copy->t      = ray->t;
  copy->tri    = ray->tri;
  copy->nLevel = ray->nLevel;
  copy->depth  = ray->depth;
}


__device__ void stackPush(DStack *stack, float pos[3],
    float dir[3], float level[3], ushort nLevel, ushort depth,
    const float att[3], DOctTree *oct) {

  DRay *ray = &stack->ray[stack->size];
  for (int i=0; i<3; i++) {
    ray->eye[i] = pos[i];
    ray->dir[i] = dir[i];
    ray->att[i] = att[i];
    ray->level[i] = level[i];
    ray->col[i] = 0;
  }
  ray->s      = NULL;
  ray->tri    = NULL;
  ray->t      = DINF;
  ray->nLevel = nLevel;
  ray->depth  = depth;
  ray->oct    = oct;
  stack->size++;
}

__device__ DRay* stackPop(DStack *stack) {
  stack->size--;
  return &stack->ray[stack->size]; 
}

__device__ char stackEmpty(DStack *stack) {
  return stack->size == 0;
}

__global__ void printNode2(DOctTree *n, int k) {

  printf("Type: %d, Triangles: %d\n", n->type, n->nTriangles);
}


__device__ void initOctStack(DOctStack *stack) {
  stack->size = 0;
}

__device__ void octStackPush(DOctStack *stack, DOctTree* oct,
    int child) {

  stack->oct[stack->size] = oct;
  stack->child[stack->size] = child;
  stack->size++;
}

__device__ DOctTree* octStackPop(DOctStack *stack) {
  stack->size--;
  return stack->oct[stack->size];
}

__device__ int topChild(DOctStack *stack) {
  return stack->child[stack->size-1];
}

__device__ void octStackAdd(DOctStack *stack) {
  stack->child[stack->size-1]--;
}

__device__ char intersectSphere(DSphere *sph, DSpheres *sphs,
    DRay *ray, DCamera *cam) {

  float ec[3],B,C,det, t1,t2, dt[3];
  subf(ec, ray->eye, sph->c);
  B = 2*dotProd(ec, ray->dir);
  C = dotProd(ec,ec) - sph->r*sph->r;
  det = B*B - 4*C;

  if (det>=0) {
    det = sqrtf(det);
    t2=-B+det; t1=-B-det;
    if (t2>=0) {

      if (t1>0 && t1<2*ray->t) { t1/=2; ray->t=t1; }
      else if (t1<0 && t2<2*ray->t) { t2/=2; ray->t=t2; }
      else return false;
     
      if (cam) { 
        sumf(ray->iPoint,ray->eye,upscaf(dt,ray->dir,ray->t));
        norm(subf(ray->n,ray->iPoint,sph->c)); 
        ray->s = sph; ray->tri = NULL;
      }
      return true;
    }
  }
  return false;
}

__device__ bool intersectTriangle(DTriangle *tri, DRay *ray,
    DCamera *cam) {
  

  float A[3][3],mini[3][3], det,alp,bet,gam,t1,b[3],x[3],
    *v0=tri->v[0],*v1=tri->v[1],*v2=tri->v[2],
    *n0=tri->n[0],*n1=tri->n[1],*n2=tri->n[2],
    wv0[3],wv1[3],wv2[3],wn0[3],wn1[3],wn2[3],
    *e=ray->eye,*d=ray->dir;

  b[0]=e[0]-v0[0]; b[1]=e[1]-v0[1]; b[2]=e[2]-v0[2];
  A[0][0]=v1[0]-v0[0]; A[0][1]=v2[0]-v0[0]; A[0][2]=-d[0];
  A[1][0]=v1[1]-v0[1]; A[1][1]=v2[1]-v0[1]; A[1][2]=-d[1];
  A[2][0]=v1[2]-v0[2]; A[2][1]=v2[2]-v0[2]; A[2][2]=-d[2];
  mini[0][0]=A[1][1]*A[2][2]-A[1][2]*A[2][1];
  mini[0][1]=-A[1][0]*A[2][2]+A[2][0]*A[1][2];
  mini[0][2]=A[1][0]*A[2][1]-A[2][0]*A[1][1];
  mini[1][0]=-A[0][1]*A[2][2]+A[2][1]*A[0][2];
  mini[1][1]=A[0][0]*A[2][2]-A[2][0]*A[0][2];
  mini[1][2]=-A[0][0]*A[2][1]+A[2][0]*A[0][1];
  mini[2][0]=A[0][1]*A[1][2]-A[1][1]*A[0][2];
  mini[2][1]=-A[0][0]*A[1][2]+A[1][0]*A[0][2];
  mini[2][2]=A[0][0]*A[1][1]-A[1][0]*A[0][1];
  det=A[0][0]*mini[0][0]+A[1][0]*mini[1][0]+A[2][0]*mini[2][0];
  x[0]=(mini[0][0]*b[0]+mini[1][0]*b[1]+mini[2][0]*b[2])/det;
  x[1]=(mini[0][1]*b[0]+mini[1][1]*b[1]+mini[2][1]*b[2])/det;
  x[2]=(mini[0][2]*b[0]+mini[1][2]*b[1]+mini[2][2]*b[2])/det;

  bet=x[0]; gam=x[1]; alp=1-bet-gam; t1=x[2];

  if (0<t1&&t1<ray->t&&0<=bet&&bet<=1&&0<=gam&&gam<=1&&0<=alp) {
    
    ray->t = t1;
    if (cam) {
      upscaf(wv0,v0,alp); upscaf(wv1,v1,bet); upscaf(wv2,v2,gam);

      upscaf(wn0,n0,alp);upscaf(wn1,n1,bet);upscaf(wn2,n2,gam);
      norm(sumf(ray->n,sumf(ray->n,wn0,wn1),wn2));

#ifdef DEBUG
      printVec(ray->n);
#endif

      //float v1d[3], v2d[3]; 
      //norm(cross(ray->n,subf(v1d, v2, v0),subf(v2d,v1,v0)));

#ifdef DEBUG
      printVec(ray->n);
#endif

      sumf(ray->iPoint,sumf(ray->iPoint,wv0,wv1),wv2);
      ray->tri=tri; ray->s = NULL; 
    }
    return true;
  }
  return false;
}

__device__ bool intersectChildren(DOctTree *oct, DRay *ray,
    DCamera *cam) {

  bool inter = false; int i;
  
  for (i=0; i<oct->nTriangles; i++) {
    inter = intersectTriangle(oct->triangles[i],ray,cam);
  }

  return inter;
}

__device__ void intersectOctNode(DOctTree *oct, DRay *ray,
    DCamera *cam, DOctStack *stack) {

  float A[3][3],b[3],x[3],det,alp,bet,t1,*d=ray->dir,*e=ray->eye,
        mini[3][3], *ini=oct->ini,*end=oct->end,v0[][3] = {
          {end[0],ini[1],ini[2]},{ini[0],ini[1],ini[2]},
          {ini[0],ini[1],end[2]},{end[0],ini[1],end[2]},
          {ini[0],end[1],end[2]},{ini[0],ini[1],ini[2]}},
        v1[][3] = {
          {ini[0],ini[1],ini[2]},{ini[0],ini[1],end[2]},
          {end[0],ini[1],end[2]},{end[0],ini[1],ini[2]},
          {end[0],end[1],end[2]},{end[0],ini[1],ini[2]}},
        v3[][3] = {
          {end[0],end[1],ini[2]},{ini[0],end[1],ini[2]},
          {ini[0],end[1],end[2]},{end[0],end[1],end[2]},
          {ini[0],end[1],ini[2]},{ini[0],ini[1],end[2]}};

  for (int i=0; i<6; i++) {

    A[0][0]=v1[i][0]-v0[i][0]; A[0][1]=v3[i][0]-v0[i][0]; 
    A[1][0]=v1[i][1]-v0[i][1]; A[1][1]=v3[i][1]-v0[i][1]; 
    A[2][0]=v1[i][2]-v0[i][2]; A[2][1]=v3[i][2]-v0[i][2]; 
    A[0][2]=-d[0]; b[0]=e[0]-v0[i][0]; 
    A[1][2]=-d[1]; b[1]=e[1]-v0[i][1]; 
    A[2][2]=-d[2]; b[2]=e[2]-v0[i][2]; 
    mini[0][0]=A[1][1]*A[2][2]-A[1][2]*A[2][1];
    mini[0][1]=-A[1][0]*A[2][2]+A[2][0]*A[1][2];
    mini[0][2]=A[1][0]*A[2][1]-A[2][0]*A[1][1];
    mini[1][0]=-A[0][1]*A[2][2]+A[2][1]*A[0][2];
    mini[1][1]=A[0][0]*A[2][2]-A[2][0]*A[0][2];
    mini[1][2]=-A[0][0]*A[2][1]+A[2][0]*A[0][1];
    mini[2][0]=A[0][1]*A[1][2]-A[1][1]*A[0][2];
    mini[2][1]=-A[0][0]*A[1][2]+A[1][0]*A[0][2];
    mini[2][2]=A[0][0]*A[1][1]-A[1][0]*A[0][1];
    det=A[0][0]*mini[0][0]+A[1][0]*mini[1][0]+A[2][0]*mini[2][0];
    x[0]=(mini[0][0]*b[0]+mini[1][0]*b[1]+mini[2][0]*b[2])/det;
    x[1]=(mini[0][1]*b[0]+mini[1][1]*b[1]+mini[2][1]*b[2])/det;
    x[2]=(mini[0][2]*b[0]+mini[1][2]*b[1]+mini[2][2]*b[2])/det;

    alp=x[0]; bet=x[1]; t1=x[2];

    if (0<t1&&t1<ray->t && 0<=bet&&bet<=1 && 0<=alp&&alp<=1) {
      octStackPush(stack,oct,0); return;
    }
  }
}

__device__ void intersectRay(DRay *ray, DSpheres *sphs,
    DCamera *cam) {

  DOctTree oct_s[20], *oct; DOctStack *octStack, octStack_s;
  octStack = &octStack_s; int child;
  for (int i=0; i<20; i++) octStack->oct[i] = &oct_s[i];
  initOctStack(octStack); 
  if (ray->oct) {
    octStackPush(octStack, ray->oct, 0);
  }

  while (octStack->size) {
    
    child = topChild(octStack);
    oct = octStackPop(octStack);
    if (oct->type == LEAVE) {
      intersectChildren(oct, ray, cam); 
    } else if (child < 8) { 
      octStackPush(octStack, oct, child+1);
      intersectOctNode(oct->child[child], ray, cam, octStack); 
    } else continue;
  }

  for (int is=0; is<sphs->nSpheres; is++)
    intersectSphere(sphs->s[is],sphs,ray,cam);
}

__device__ char throwShadowRay(float pos[3], float dir[3],
    DSpheres *sphs, DOctTree *oct, float lDist) {

  DRay ray;
  for (int i=0; i<3; i++) {
    ray.eye[i] = pos[i];
    ray.dir[i] = dir[i];
  }
  ray.s     = NULL;
  ray.tri   = NULL;
  ray.t     = DINF;
  ray.oct   = oct;
  intersectRay(&ray, sphs, NULL);

  if (lDist>ray.t) return 1; 
  else return 0;
}

__device__ void iluminateLambert(DLight *li, DCamera *cam,
    const DMaterial *mat, DSpheres *sphs, DRay *ray) {

  float lDist, lambert, sP[3], l[3], dl[3],
    mulCols[3],dt[3];

  switch (li->type) {
  
    case PUNCTUAL:
      norm(subf(l,li->pos,ray->iPoint));
      sumf(sP,ray->iPoint,upscaf(dl,l,cam->epsilon));
      lDist = normf(3,subf(dt,li->pos,sP));
  
      if (cam->shadows&&throwShadowRay(sP,l,sphs,ray->oct,lDist))
        return;

      lambert = maxf(dotProd(ray->n,l),0.f);
      mulf(mulCols,mat->col,li->col);
      sumf(ray->col,ray->col,upscaf(mulCols,mulCols,lambert));
      return;

    case SPOT:
      norm(subf(l,li->pos,ray->iPoint));
      sumf(sP,ray->iPoint,upscaf(dl,l,cam->epsilon));
      lDist = normf(3,subf(dt,li->pos,sP));

      if (cam->shadows&&throwShadowRay(sP,l,sphs,ray->oct,lDist))
        return;

      if (-dotProd(l,li->dir) > cosf(li->angle/2)){
        lambert = maxf(dotProd(ray->n,l),0.f);
        mulf(mulCols,mat->col,li->col);
        sumf(ray->col,ray->col,upscaf(mulCols,mulCols,lambert));
      }

      return;
    case AMBIENT:
      mulf(mulCols,mat->col,li->col);
      sumf(ray->col,ray->col,mulCols);
      return;
    case DIRECTIONAL:
      upscaf(l,li->dir,-1);
      sumf(sP,ray->iPoint,upscaf(dl,l,cam->epsilon));
      lDist = DINF;

      if (cam->shadows&&throwShadowRay(sP,l,sphs,ray->oct,lDist))
        return;

      lambert = maxf(dotProd(ray->n,l),0.f);
      mulf(mulCols,mat->col,li->col);
      sumf(ray->col,ray->col,upscaf(mulCols,mulCols,lambert));
      return;
  }
}
__device__ void iluminatePhong(DLight *li, DCamera *cam,
    const DMaterial *mat, DSpheres *sphs, DRay* ray) {

  float lDist, phong, sP[3], l[3], dl[3], v[3], h[3],
    mulCols[3],dt[3];
  
  switch (li->type) {
  
    case PUNCTUAL:
      norm(subf(l,li->pos,ray->iPoint));
      sumf(sP,ray->iPoint,upscaf(dl,l,cam->epsilon));
      lDist = normf(3,subf(dt,li->pos,sP));

      if (cam->shadows&&throwShadowRay(sP,l,sphs,ray->oct,lDist))
        return;

      upscaf(v,ray->dir,-1); norm(sumf(h,v,l));
      phong = powf(maxf(dotProd(ray->n,h),0.f), mat->param);
      mulf(mulCols,mat->col,li->col);
      sumf(ray->col,ray->col,upscaf(mulCols,mulCols,phong));
      return;
    case SPOT:
      norm(subf(l,li->pos,ray->iPoint));
      sumf(sP,ray->iPoint,upscaf(dl,l,cam->epsilon));
      lDist = normf(3,subf(dt,li->pos,sP));

      if (cam->shadows&&throwShadowRay(sP,l,sphs,ray->oct,lDist))
        return;

      if (-dotProd(l,li->dir) > cosf(li->angle/2)){
        upscaf(v,ray->dir,-1); norm(sumf(h,v,l));
        phong = powf(maxf(dotProd(ray->n,h),0.f), mat->param);
        mulf(mulCols,mat->col,li->col);
        sumf(ray->col,ray->col,upscaf(mulCols,mulCols,phong));
      }

      return;
    case AMBIENT:
      return;
    case DIRECTIONAL:
      upscaf(l,li->dir,-1); 
      sumf(sP,ray->iPoint,upscaf(dl,l,cam->epsilon));
      lDist = DINF;

      if (cam->shadows&&throwShadowRay(sP,l,sphs,ray->oct,lDist))
        return;

      upscaf(v,ray->dir,-1); norm(sumf(h,v,l));
      phong = powf(maxf(dotProd(ray->n,h),0.f), mat->param);
      mulf(mulCols,mat->col,li->col);
      sumf(ray->col,ray->col,upscaf(mulCols,mulCols,phong));
      return;
  }
}


__device__ void colorate(DStack *stack, 
    const DMaterial *mat, DSpheres *sphs, const DLights *lights,
    DCamera *cam, DRay *ray) {

  switch (mat->type) {
    case LAMBERT:
      for (int i=0; i<lights->nLights; i++)
        iluminateLambert(lights->l[i],
            cam,mat,sphs,ray); 
      return;
    case PHONG:
      for (int i=0; i<lights->nLights; i++)
        iluminatePhong(lights->l[i],
            cam,mat,sphs,ray);
      return;
    case REFLECTIVE:
      if (ray->depth < cam->ray_depth) {
        float r[3], sP[3], N[3], dr[3], att[3];
        upscaf(N, ray->n, 2*dotProd(ray->dir, ray->n));
        norm(subf(r, ray->dir, N)); upscaf(dr, r, DEPSILON);
        sumf(sP, ray->iPoint, dr);
        mulf(att,mat->col,ray->att);
        stackPush(stack, sP,r,ray->level,ray->nLevel,
            ray->depth+1,att,ray->oct);
      }
      return;
    case DIELECTRIC:
      if (ray->depth < cam->ray_depth) {
        float rl[3],rlsP[3],drl[3],rf[3],rfsP[3],drf[3],N[3],refr,
        R0,rOrI,totRefl,K[3],ni[3],cos,fresnel,attl[3],attf[3];
        refr = ray->level[ray->nLevel];
        upscaf(N,ray->n,2*dotProd(ray->dir,ray->n));
        norm(subf(rl,ray->dir,N));
        upscaf(drl, ray->n, DEPSILON);
        sumf(rlsP, ray->iPoint, drl);
        R0 = powf((refr-mat->param)/(refr+mat->param),2);

        
        if (dotProd(ray->dir, ray->n)<0) {
          rOrI = refr/mat->param;
          cos = -dotProd(ray->dir, ray->n);
          totRefl = 1-powf(rOrI,2)*(1-powf(cos,2));
          iniVec(K,1.0,1.0,1.0);
          mulf(attl,mulf(attl,mat->col,K),ray->att);
          mulf(attf,mulf(attf,mat->col,K),ray->att);; 
          if (totRefl >= 0) {
            refractf(rf, ray->dir, ray->n, rOrI);
            upscaf(drf,rf,DEPSILON);
            sumf(rfsP, ray->iPoint, drf);
            fresnel = R0+(1-R0)*pow(1.0-cos,5);

            upscaf(attf,attf,1-fresnel);
            ray->nLevel++; ray->level[ray->nLevel]=mat->param;
            stackPush(stack, rfsP,rf,ray->level,ray->nLevel,
                ray->depth+1,attf,ray->oct);
            ray->nLevel--;

            upscaf(attl,attl,fresnel);
            stackPush(stack, rlsP,rl,ray->level,ray->nLevel,
                ray->depth+1,attl,ray->oct);
          } else {
            stackPush(stack, rlsP,rl,ray->level,ray->nLevel,
                ray->depth+1,attl,ray->oct);
          }
        } else {
          rOrI = mat->param/refr; upscaf(ni,ray->n,-1);
          cos = dotProd(ray->dir, ray->n);
          totRefl = 1-powf(rOrI,2)*(1-powf(cos,2));
          expv(K,upscaf(K,mat->att,-ray->t));
          mulf(attl,mulf(attl,mat->col,K),ray->att);
          mulf(attf,mulf(attf,mat->col,K),ray->att);; 
          if (totRefl >= 0) {
            refractf(rf,ray->dir,ni,rOrI);
            upscaf(drf,rf,DEPSILON);
            sumf(rfsP,ray->iPoint,drf);
            fresnel = R0+(1-R0)*pow(1.0-cos,5);

            upscaf(attf,attf,1-fresnel);
            stackPush(stack, rfsP,rf,ray->level,ray->nLevel-1,
                ray->depth+1,attf, ray->oct);

            upscaf(attl,attl,fresnel);
            stackPush(stack, rlsP,rl,ray->level,ray->nLevel,
                ray->depth+1,attl, ray->oct);
          } else {
            stackPush(stack, rlsP,rl,ray->level,ray->nLevel,
                ray->depth+1,attl,ray->oct);
          }
        }
      }
      return;
  }
}


__device__ void colorateRay(DStack *stack, DRay *ray,
    DSpheres *sphs, DCamera *cam, DLights *ls) {

  if (ray->s) {
    for (int im=0; im<ray->s->nMats; im++) {
      colorate(stack,ray->s->mats[im],sphs,ls,cam,ray); 
    }
  } else if (ray->tri) {
    for (int im=0; im<ray->tri->nMats; im++) {
      colorate(stack,ray->tri->mats[im],sphs,ls,cam,ray); 
    }
  }
  else copyVec(ray->col, cam->bg_col);
}
    

__global__ void film(ubyte *img, DCamera *d_c, DLights *d_l,
    DSpheres *d_s, DOctTree *d_o) {

  DRay *ray, ray_s; ray = &ray_s; 
  float ic,jc,d[3],coords[3],level[3]={d_c->scene_refr,0,0};
  uint W,H,S, i,j,c_pos; W=d_c->width; H=d_c->height; S=W*H;

  for (uint pixId = __mul24(blockIdx.x,blockDim.x)+threadIdx.x;
    pixId<S; pixId+=blockDim.x*gridDim.x) {
    
#ifndef DEBUG
    i=pixId%W; j=pixId/W;
#else
    i=255; j=255;//j=246;
#endif

    ic = d_c->left+((d_c->right-d_c->left)/d_c->width)*(i+0.5);
    jc = d_c->bottom+((d_c->top-d_c->bottom)/d_c->height)*(j+0.5);
    iniVec(coords,ic,jc,-d_c->near);
    norm(linCom(d,d_c->u,d_c->v,d_c->w,coords));

    float color[3] = {0,0,0}, att[3] = {1,1,1}, attCol[3];
    DStack stack; initStack(&stack);
    stackPush(&stack, d_c->pos, d, level, 0, 0, att, d_o);

    while (!stackEmpty(&stack)) {
      copyRay(ray, stackPop(&stack));
      intersectRay(ray, d_s, d_c);
      colorateRay(&stack, ray, d_s, d_c, d_l);

      mulf(attCol, ray->att, ray->col);
      sumf(color, color, attCol);
    }
    color[0] = minf(color[0],1.f); 
    color[1] = minf(color[1],1.f);
    color[2] = minf(color[2],1.f);
    upscaf(color,color, 255.f);
#ifdef DEBUG
    printf("%f %f %f\n", color[0],color[1],color[2]);
    break;
#else
    //c_pos = 4*(d_c->width*j + i);
    c_pos = 3*(d_c->width*(d_c->height-1-j) + i);

    img[c_pos+0] = color[0];
    img[c_pos+1] = color[1];
    img[c_pos+2] = color[2];
#endif
  }
}


int main(int argc, char **argv) {
  
  clock_t begin = clock();
  Camera *cam = NULL;
  string img_name = parse(argc, argv, cam); 
 
  CameraGPUAdapter *ca = new CameraGPUAdapter(cam);
  LightsGPUAdapter *la = new LightsGPUAdapter(allLights);
  ObjectsGPUAdapter *oa = new ObjectsGPUAdapter(allObjects);
  MeshGPUAdapter *ma = new MeshGPUAdapter(allMeshes);
  OctNode *octN = NULL; OctGPUAdapter *octa = NULL;
  if (allMeshes.size()) {
    octN = new OctBranch(
      vector<float>{-INF/4,-INF/4,INF/4},
      vector<float>{INF/4,INF/4,-INF/4});
    for (auto &mesh: allMeshes)
      mesh->insertInto(octN);
  }
  octa = new OctGPUAdapter(octN);

#ifndef DEBUG
  cout << "Oct taken seconds: ";
  cout << (clock()-begin)/(double)CLOCKS_PER_SEC << endl;
  begin = clock();
#endif
  size_t reqBlocksCount;
  ushort threadsCount,blocksCount;
  threadsCount   = 128;
  reqBlocksCount = W*H / threadsCount;
  blocksCount    = (ushort)min((size_t)32768, reqBlocksCount);

  ubyte *d_img, *h_img;
  h_img = (ubyte*)malloc(W*H*3*sizeof(ubyte));
  cudaMalloc((void **)&d_img, W*H*3*sizeof(ubyte));

#ifndef DEBUG
  film<<<blocksCount,threadsCount>>>(d_img, ca->getDCamera(),
      la->getDLights(),oa->getDObjects(), octa->getDOct());
#else
  film<<<1,1>>>(d_img, ca->getDCamera(),
      la->getDLights(),oa->getDObjects(), octa->getDOct());
#endif

  cudaError_t cudaerr = cudaDeviceSynchronize();
  if (cudaerr != cudaSuccess)
    printf("kernel launch failed with error \"%s\".\n",
        cudaGetErrorString(cudaerr));

#ifndef DEBUG
  cudaMemcpy(h_img, d_img, W*H*3*sizeof(ubyte),
      cudaMemcpyDeviceToHost);

  write_image(img_name, W, H, h_img);
#endif

  delete octa; delete octN;
  delete ca; delete la; delete oa; delete cam; delete ma;
  for (auto &obj: allObjects) delete obj;
  for (auto &mesh: allMeshes) delete mesh;
  for (auto &light: allLights) delete light;
  for (auto &mat: allMaterials) delete mat.second;
  
#ifndef DEBUG
  cout << "Rendering seconds: ";
  cout << (clock()-begin)/(double)CLOCKS_PER_SEC << endl;
#endif
}
