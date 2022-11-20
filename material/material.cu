#include <material.cuh>
#include <lights.cuh>
#include <spheres.cuh>
#include <vector.cuh>
#include <ray.cuh>
#include <camera.cuh>
#include <model.cuh>
using namespace std;

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

void LambertMaterial::colorate(vector<float> &color,
    const vector<float> &iSphere, const vector<float> &n,
    const vector<float> &d, stack<float> &level,
    unsigned char depth, float t) const {

  for (auto &light: allLights){
    light->iluminateLambert(color, _color, iSphere, n);
  }
}

DMaterial *LambertMaterial::buildDMaterial(void) const {

  DMaterial *mat = (DMaterial*)malloc(sizeof(DMaterial)), *dmat;
  cudaMalloc((void **)&dmat, sizeof(DMaterial));

  mat->type = LAMBERT;
  for (int i=0; i<3; i++)
    mat->col[i] = _color[i];

  cudaMemcpy(dmat,mat, sizeof(DMaterial),cudaMemcpyHostToDevice);
  free(mat);

  return dmat;
}

void PhongMaterial::colorate(vector<float> &color,
    const vector<float> &iSphere, const vector<float> &n,
    const vector<float> &d, stack<float> &level,
    unsigned char depth, float t) const {

  for (auto &light: allLights){
    light->iluminatePhong(color, _color, iSphere, n, d, _shi);
  }
}

DMaterial *PhongMaterial::buildDMaterial(void) const {

  DMaterial *mat = (DMaterial*)malloc(sizeof(DMaterial)), *dmat;
  cudaMalloc((void **)&dmat, sizeof(DMaterial));

  mat->type = PHONG;
  mat->param = _shi;
  for (int i=0; i<3; i++)
    mat->col[i] = _color[i];

  cudaMemcpy(dmat,mat, sizeof(DMaterial),cudaMemcpyHostToDevice);
  free(mat);

  return dmat;
}

void ReflectiveMaterial::colorate(vector<float> &color,
    const vector<float> &iPoint, const vector<float> &n,
    const vector<float> &d, stack<float> &level,
    unsigned char depth, float t) const {

  if (depth<DEPTH) {
    vector <float> r,sPoint,refColor(3,0.0); Ray *ray;

    r      = normfv(subfv(d,upscafv(2*dotfv(d,n),n)));
    sPoint = sumfv(iPoint, upscafv(EPSILON,r));
    ray    = new Ray(refColor, sPoint, r, level, depth+1);
    ray->intersect();

    delete ray;
    color  = sumfv(color,mulfv(_color,refColor));
  }
}

DMaterial *ReflectiveMaterial::buildDMaterial(void) const {

  DMaterial *mat = (DMaterial*)malloc(sizeof(DMaterial)), *dmat;
  cudaMalloc((void **)&dmat, sizeof(DMaterial));

  mat->type = REFLECTIVE;
  for (int i=0; i<3; i++)
    mat->col[i] = _color[i];

  cudaMemcpy(dmat,mat, sizeof(DMaterial),cudaMemcpyHostToDevice);
  free(mat);

  return dmat;
}

void DielectricMaterial::colorate(vector<float> &color,
    const vector<float> &iPoint, const vector<float> &n,
    const vector<float> &d, stack<float> &level,
    unsigned char depth, float t) const {

  if (depth<DEPTH) { float refr,rOrI,R0,fresnel,cosine,totRefl;
    vector <float> rl,rlsPoint,rlColor, rr,rrsPoint,rrColor,K,ni;
    Ray *rayRl,*rayRr; refr = level.top();

    rl       = normfv(subfv(d,upscafv(2*dotfv(d,n),n)));
    rlsPoint = sumfv(iPoint, upscafv(EPSILON,rl));
    rlColor  = vector<float>(3,0.0);
    rayRl    = new Ray(rlColor, rlsPoint, rl, level, depth+1);
    rayRl->intersect(); delete rayRl;
    R0       = pow((_refr-refr)/(_refr+refr),2);


    if (dotfv(d,n)<0){
      rOrI     = refr/_refr;
      totRefl  = 1-pow(rOrI,2)*(1-pow(dotfv(d,n),2));
      K        = vector<float>(3,1.0);
      if (totRefl >= 0){
        rr = refract(d,n,rOrI);
        cosine   = -dotfv(d,n);
        rrsPoint = sumfv(iPoint, upscafv(EPSILON,rr));
        rrColor  = vector<float>(3,0.0);
        level.push(_refr);
        rayRr    = new Ray(rrColor,rrsPoint,rr,level,depth+1);
        level.pop(); rayRr->intersect(); delete rayRr;
      } else {
        color    =  sumfv(color,mulfv(K,mulfv(_color,rlColor)));
        return;
      }
    } else {
      rOrI     = _refr/refr;
      totRefl  = 1-pow(rOrI,2)*(1-pow(dotfv(d,n),2));
      K        = expfv(upscafv(-t,_att));
      if (totRefl >= 0){
        ni       = upscafv(-1,n);
        rr = refract(d,ni,rOrI);
        cosine   = -dotfv(d,ni);
        rrsPoint = sumfv(iPoint, upscafv(EPSILON,rr));
        rrColor  = vector<float>(3,0.0);
        level.pop();
        rayRr    = new Ray(rrColor,rrsPoint,rr,level,depth+1);
        level.push(refr); rayRr->intersect(); delete rayRr;
      } else {
        color    =  sumfv(color,mulfv(K,mulfv(_color,rlColor)));
        return;
      }
    }
    fresnel  = R0+(1-R0)*pow(1.0-cosine,5);
    color    = sumfv(color,mulfv(K,mulfv(_color,sumfv(
                  upscafv(fresnel,rlColor),
                  upscafv((1-fresnel),rrColor)))));
  }
}

DMaterial *DielectricMaterial::buildDMaterial(void) const {

  DMaterial *mat = (DMaterial*)malloc(sizeof(DMaterial)), *dmat;
  cudaMalloc((void **)&dmat, sizeof(DMaterial));

  mat->type = DIELECTRIC;
  mat->param = _refr;
  for (int i=0; i<3; i++) {
    mat->col[i] = _color[i];
    mat->att[i] = _att[i];
  }

  cudaMemcpy(dmat,mat, sizeof(DMaterial),cudaMemcpyHostToDevice);
  free(mat);

  return dmat;
}

TextureMaterial::TextureMaterial(string mat_file) {

  _file = mat_file;
}

void TextureMaterial::colorate(vector<float>&color,
        const vector<float>&iPoint, const vector<float> &n,
        const vector<float>&d, stack<float> &level,
        unsigned char depth, float t) const {

  // CPU
}

DMaterial *TextureMaterial::buildDMaterial() const {

  DMaterial *mat = (DMaterial*)malloc(sizeof(DMaterial)), *dmat;
  cudaMalloc((void**)&dmat, sizeof(DMaterial));

  float *tex = (float*)malloc(_W*_H*3*sizeof(float));
  // Llenar tex con lo de la imagen
  // (en mat_file) (con lo de la parte de jpg y png)

  cudaMalloc((void**)&d_tex, _W*_H*3*sizeof(float));
  cudaMemcpy(mat->tex,tex, _W*_H*3*sizeof(float),
      cudaMemcpyHostToDevice);
  mat->type = TEXTURE;
  mat->tex  = d_tex;


  cudaMemcpy(dmat,mat, sizeof(DMaterial),cudaMemcpyHostToDevice);
  free(mat);

  return dmat;
}

TextureMaterial::~TextureMaterial(void) {

  cudaFree(d_tex);
}

void PunctualLight::iluminateLambert(vector<float> &color,
   const vector<float> &mat_color, const vector<float> &iPoint,
   const vector<float> &n){

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

  lambert = max(dotfv(n,l),0.f);
  color = sumfv(color,upscafv(lambert,mulfv(mat_color,_color)));
}

__global__ void iniMats(DSphere *s, DMaterial **mats) {
  s->mats = mats;
}
