#include <lights.cuh>
#include <vector.cuh>
#include <model.cuh>
#include <camera.cuh>
#include <object.cuh>
#include <ray.cuh>
#include <material.cuh>
using namespace std;

Light::Light(const vector<float> color): _color(color) {}
Light::~Light(void) {};

PunctualLight::PunctualLight(vector<float> color,
  vector<float> pos): Light(color), _pos(pos) {}
PunctualLight::~PunctualLight(void) {}

SpotLight::SpotLight(std::vector<float> color, std::vector<float> pos,
    float angle, std::vector<float> dir): Light(color),
  _pos(pos), _dir(dir), _angle(angle) {}
SpotLight::~SpotLight(void) {}

AmbientLight::AmbientLight(std::vector<float> color): Light(color) {}
AmbientLight::~AmbientLight(void){}

DirectionalLight::DirectionalLight(std::vector<float> color,
    std::vector<float> dir): Light(color),
  _dir(normfv(dir)) {}
DirectionalLight::~DirectionalLight(void) {}

void PunctualLight::iluminatePhong(vector<float> &color,
    const vector<float> &mat_color, const vector<float> &iPoint,
    const vector<float> &n, const vector<float> &d, int shi){

  vector<float> sPoint,v,l, dumI,dumN;
  float lDist, phong, objDist=INF;
  vector<const Material*>dumM(0);
  l       = normfv(subfv(_pos,iPoint));
  sPoint  = sumfv(iPoint,upscafv(EPSILON,l));
  lDist   = magfv(subfv(_pos,sPoint));

  if (lDist && SHADOWS_ENABLED) {
    for (auto &anObj: allObjects)
        anObj->intersectsRay(sPoint,l,dumI,dumN,objDist,dumM);
  }

  if (lDist>objDist) return;

  v = upscafv(-1,d);
  phong = pow(max(dotfv(n,normfv(sumfv(v,l))),0.f), shi);
  color = sumfv(color,upscafv(phong,mulfv(mat_color,_color)));
}

DLight* PunctualLight::buildDLight(void) const {

  DLight *l = (DLight*)malloc(sizeof(DLight)), *d_l;
  cudaMalloc((void **)&d_l, sizeof(DLight));

  l->type = PUNCTUAL;
  for (int i=0; i<3; i++){
    l->pos[i]=_pos[i];
    l->col[i]=_color[i];
  }

  cudaMemcpy(d_l, l, sizeof(DLight),cudaMemcpyHostToDevice);
  free(l);

  return d_l;
}


void SpotLight::iluminateLambert(vector<float> &color,
   const vector<float> &mat_color, const vector<float> &iPoint,
   const vector<float> &n) {

  vector<float> l,sPoint, dumI,dumN;
  float lDist, lambert, objDist=INF;
  vector<const Material*>dumM(0);
  l       = normfv(subfv(_pos,iPoint));
  sPoint  = sumfv(iPoint,upscafv(EPSILON,l));
  lDist   = magfv(subfv(_pos,sPoint));

  if (lDist && SHADOWS_ENABLED) {
    for (auto &anObj: allObjects)
        anObj->intersectsRay(sPoint,l,dumI,dumN,objDist,dumM);
  }

  if (lDist > objDist) return;

  if (-dotfv(l,_dir) > cos(_angle/2)){
    lambert = max(dotfv(n,l),0.f);
    color=sumfv(color,upscafv(lambert,mulfv(mat_color,_color)));
  }
}

void SpotLight::iluminatePhong(vector<float> &color,
    const vector<float> &mat_color, const vector<float> &iPoint,
    const vector<float> &n, const vector<float> &d, int shi) {

  vector<float> sPoint,v,l, dumI,dumN;
  float lDist, phong, objDist=INF;
  vector<const Material*>dumM(0);
  l       = normfv(subfv(_pos,iPoint));
  sPoint  = sumfv(iPoint,upscafv(EPSILON,l));
  lDist   = magfv(subfv(_pos,sPoint));

  if (lDist && SHADOWS_ENABLED) {
    for (auto &anObj: allObjects)
      anObj->intersectsRay(sPoint,l,dumI,dumN,objDist,dumM);
  }

  if (lDist > objDist) return;

  if (-dotfv(l,_dir) > cos(_angle/2)){
    v     = upscafv(-1,d);
    phong = pow(max(dotfv(n,normfv(sumfv(v,l))),0.f), shi);
    color = sumfv(color,upscafv(phong,mulfv(mat_color,_color)));
  }
}

DLight* SpotLight::buildDLight(void) const {

  DLight *l = (DLight*)malloc(sizeof(DLight)), *d_l;
  cudaMalloc((void **)&d_l, sizeof(DLight));

  l->type = SPOT;
  l->angle = _angle;
  for (int i=0; i<3; i++){
    l->pos[i]=_pos[i];
    l->col[i]=_color[i];
    l->dir[i]=_dir[i];
  }

  cudaMemcpy(d_l, l, sizeof(DLight),cudaMemcpyHostToDevice);
  free(l);

  return d_l;
}

void AmbientLight::iluminateLambert(
  std::vector<float> &color,
  const std::vector<float>&mat_color,
  const std::vector<float>&iPoint,
  const std::vector<float> &n) {
  color = sumfv(color,mulfv(_color,mat_color));
}

void AmbientLight::iluminatePhong(
  std::vector<float> &color,
  const std::vector<float> &mat_color,
  const std::vector<float> &iPoint,
  const std::vector<float> &n,
  const std::vector<float> &d, int shi) {}

DLight* AmbientLight::buildDLight(void) const {

  DLight *l = (DLight*)malloc(sizeof(DLight)), *d_l;
  cudaMalloc((void **)&d_l, sizeof(DLight));

  l->type = AMBIENT;
  for (int i=0; i<3; i++){
    l->col[i]=_color[i];
  }

  cudaMemcpy(d_l, l, sizeof(DLight),cudaMemcpyHostToDevice);
  free(l);

  return d_l;
}

void DirectionalLight::iluminateLambert(vector<float> &color,
   const vector<float> &mat_color, const vector<float> &iPoint,
   const vector<float> &n){

  vector<float> sPoint,l, dumI,dumN;
  float lDist, lambert, objDist=INF;
  vector<const Material*>dumM(0);
  l      = upscafv(-1,_dir);
  sPoint = sumfv(iPoint,upscafv(EPSILON,l));
  lDist  = INF;

  if (lDist && SHADOWS_ENABLED) {
    for (auto &anObj: allObjects)
      anObj->intersectsRay(sPoint,l,dumI,dumN,objDist,dumM);
  }

  if (lDist > objDist) return;

  lambert= max(dotfv(n,l),0.f);
  color  = sumfv(color,upscafv(lambert,mulfv(mat_color,_color)));
}

void DirectionalLight::iluminatePhong(vector<float> &color,
    const vector<float> &mat_color, const vector<float> &iPoint,
    const vector<float> &n, const vector<float> &d, int shi){

  vector<float> sPoint,v,l, dumI, dumN;
  float lDist, phong, objDist=INF;
  vector<const Material*>dumM(0);
  l      = upscafv(-1,_dir);
  sPoint = sumfv(iPoint,upscafv(EPSILON,l));
  lDist  = INF;

  if (lDist && SHADOWS_ENABLED) {
    for (auto &anObj: allObjects)
      anObj->intersectsRay(sPoint,l,dumI,dumN,objDist,dumM);
  }

  if (lDist > objDist) return;

  v = upscafv(-1,d);
  phong = pow(max(dotfv(n,normfv(sumfv(v,l))),0.f), shi);
  color = sumfv(color,upscafv(phong,mulfv(mat_color,_color)));
}

DLight* DirectionalLight::buildDLight(void) const {

  DLight *l = (DLight*)malloc(sizeof(DLight)), *d_l;
  cudaMalloc((void **)&d_l, sizeof(DLight));

  l->type = DIRECTIONAL;
  for (int i=0; i<3; i++){
    l->col[i]=_color[i];
    l->dir[i]=_dir[i];
  }

  cudaMemcpy(d_l, l, sizeof(DLight),cudaMemcpyHostToDevice);
  free(l);

  return d_l;
}

LightsGPUAdapter::LightsGPUAdapter(const vector<Light*> &l) {

  // Init GPU point & assosiated pointers

  _nLights = l.size();
  DLight **d_ls;

  htod_l = (DLight**)malloc(_nLights*sizeof(DLight*));
  for (int i=0; i<_nLights; i++)
    htod_l[i] = l[i]->buildDLight();

  cudaMalloc((void **)&d_ls, l.size()*sizeof(DLight*));
  cudaMemcpy(d_ls, htod_l, l.size()*sizeof(DLight*),
      cudaMemcpyHostToDevice);

  h_l = (DLights*)malloc(sizeof(DLights));
  h_l->l = d_ls; h_l->nLights = _nLights;
  cudaMalloc((void **)&d_l, sizeof(DLights));
  cudaMemcpy(d_l, h_l, sizeof(DLights),
      cudaMemcpyHostToDevice);
}

LightsGPUAdapter::~LightsGPUAdapter(void) {

  for (int i=0; i<_nLights; i++) cudaFree(htod_l[i]);
  free(htod_l); cudaFree(d_l); cudaFree(h_l->l); free(h_l);
}

__device__ void iluminateLambert(DLight *li, DCamera *cam,
    const DMaterial *mat, DSpheres *sphs, DRay *ray) {

  float lDist, lambert, sP[3], l[3], dl[3],
    mulCols[3],dt[3];

  switch (li->type) {

    case PUNCTUAL:
      norm(subf(l,li->pos,ray->iPoint));
      sumf(sP, ray->iPoint, upscaf(dl, l, cam->epsilon));
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
