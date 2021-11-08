#ifndef VECTOR_H
#define VECTOR_H
#include <vector>

struct DRay;

std::vector<float> sumfv(std::vector<float> v1,
    const std::vector<float> &v2);
std::vector<float> subfv(std::vector<float> v1,
    const std::vector<float> &v2);
std::vector<float> mulfv(std::vector<float> v1,
    const std::vector<float> &v2);
std::vector<float> upscafv(float c, std::vector<float> v);
std::vector<float> doscafv(float c, std::vector<float> v);
void matmul(const std::vector<std::vector<float>> &mat,
  std::vector<float> &v);
std::vector<float> expfv(std::vector<float> v);

float dotfv(const std::vector<float> &v1,
  const std::vector<float> &v2);
std::vector<float> crossf3(const std::vector<float> &u,
    const std::vector<float> &v);
float magfv(const std::vector<float> &v);
std::vector<float> normfv(const std::vector<float> &v);
float accfv(const std::vector<float> &v);
void printVector(const std::vector<float> &v);
std::vector<float> refract(const std::vector<float> &d,
 const std::vector<float> &n, float rori);

__device__ float minf(float a, float b);
__device__ float maxf(float a, float b);
__device__ void iniVec(float v[3], float a, float b, float c);
__device__ void copyVec(float v[3], const float v1[3]);
__device__ void printVec(const float v[3]);
__device__ float dotProd(const float v1[3], const float v2[3]);
__device__ float* norm(float v[3]);
__device__ float* subf(float v[3], const float v1[3],
  const float v2[3]);
__device__ float* sumf(float v[3], const float v1[3],
    const float v2[3]);
__device__ float* mulf(float v[3], const float v1[3],
    const float v2[3]);

__device__ float* upscaf(float v[3], const float v1[3], float m);
__device__ float* doscaf(float v[3], const float v1[3], float m);
__device__ float* expv(float v[3], const float v1[3]);
__device__ float* cross(float v[3], const float v1[3],
    const float v2[3]);
__device__ float* refractf(float rr[3], const float d[3],
    const float n[3], float rori);
__device__ float* linCom(float v[3], const float v1[3],
    const float v2[3], const float v3[3], const float s[3]);

struct DStack; struct DOctTree;
__device__ void initStack(DStack *stack);
__device__ void copyRay(DRay *copy, DRay *ray);
__device__ void stackPush(DStack *stack, float pos[3],
    float dir[3], float level[3], ushort nLevel, ushort depth,
    const float att[3], DOctTree *oct);
__device__ DRay* stackPop(DStack *stack);
__device__ char stackEmpty(DStack *stack);

#endif /*VECTOR_H*/
