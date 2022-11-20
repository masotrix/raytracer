#include <algorithm>
#include <functional>
#include <numeric>
#include <iostream>
#include <vector.cuh>
#include <ray.cuh>
#include <model.cuh>
using namespace std;

vector<float> sumfv(vector<float> v1, const vector<float> &v2){
  transform(v1.begin(),v1.end(),v2.begin(),v1.begin(),
      plus<float>()); return v1;
}

vector<float> subfv(vector<float> v1, const vector<float> &v2){
  transform(v1.begin(),v1.end(),v2.begin(),v1.begin(),
      minus<float>()); return v1;
}

vector<float> mulfv(vector<float> v1, const vector<float> &v2){
  transform(v1.begin(),v1.end(),v2.begin(),v1.begin(),
      multiplies<float>()); return v1;
}

vector<float> upscafv(float c, vector<float> v){
  transform(v.begin(),v.end(),v.begin(),
      bind1st(multiplies<float>(),c)); return v;
}

vector<float> doscafv(float c, vector<float> v){
  transform(v.begin(),v.end(),v.begin(),
      bind2nd(divides<float>(),c)); return v;
}

void matmul(const vector<vector<float>> &mat, vector<float> &v) {
  float v2[3];
  for (int i=0; i<3; i++)
    v2[i] = mat[i][0]*v[0]+mat[i][1]*v[1]+mat[i][2]*v[2];

  for (int i=0; i<3; i++) v[i] = v2[i];
}

struct exptiate{float operator()(float f)const {return exp(f);}};
vector<float> expfv(vector<float> v){
  transform(v.begin(),v.end(),v.begin(),exptiate()); return v;
}

float dotfv(const vector<float> &v1, const vector<float> &v2){
  return inner_product(v1.begin(),v1.end(),v2.begin(),0.0);
}

vector<float> crossf3(const vector<float> &u,
    const vector<float> &v){ return
  {u[1]*v[2]-u[2]*v[1],u[2]*v[0]-u[0]*v[2],u[0]*v[1]-u[1]*v[0]};
}

float magfv(const vector<float> &v){
  return sqrt(dotfv(v,v));
}

vector<float> normfv(const vector<float> &v){
  return doscafv(magfv(v),v);
}

float accfv(const vector<float> &v){
  return accumulate(v.begin(),v.end(),0.0);
}

void printVector(const vector<float> &v) {
  for (auto &c: v) cout << c << ' ';
  cout << endl;
}

vector<float> refract(const vector<float> &d,
 const vector<float> &n, float rori) {

  float costhe = -dotfv(d,n), sinthe = sqrt(1-pow(costhe,2)),
    sinfi = rori*sinthe, cosfi = sqrt(1-pow(sinfi,2));

  vector<float> nn;
  if (costhe < 0){ nn = upscafv(-1,n); costhe = -costhe; }
  else nn = n;

  vector<float> b = doscafv(sinthe,sumfv(d, upscafv(costhe,nn))),
    rr = subfv(upscafv(sinfi,b),upscafv(cosfi,nn));

  return rr;
}

__device__ float minf(float a, float b)  { return a<b? a:b; }

__device__ float maxf(float a, float b)  { return a>b? a:b; }

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
