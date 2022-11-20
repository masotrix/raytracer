#include <spheres.cuh>
#include <vector.cuh>
#include <material.cuh>
#include <ray.cuh>
using namespace std;

bool Sphere::intersectsRay(const vector<float> &e,
    const vector<float> &d, vector<float> &iSphere,
    vector<float> &n, float &t, vector<const Material*>&mats){

  float B,C,det,t1,t2;
  vector <float> ec;

  ec  = subfv(e,_c);
  B   = 2*dotfv(ec,d);
  C   = dotfv(ec,ec)-_r*_r;
  det = B*B - 4*C;

  if (det>=0) {
    det = sqrt(det);
    t2=-B+det; t1=-B-det;
    if (t2>=0) {

      if (t1>0 && t1<2*t) { t1/=2; t=t1; }
      else if (t1<0 && t2<2*t) { t2/=2; t=t2; }
      else return false;

      iSphere = sumfv(e,upscafv(t,d));
      n      = normfv(subfv(iSphere,_c));
      mats   = _mats;
      return true;
    }
  }
  return false;
}

SphereGPUAdapter::SphereGPUAdapter(Sphere *s) {

  // Init GPU point & assosiated pointers

  int nMats = s->getMaterials().size();
  h_s = (DSphere*)malloc(sizeof(DSphere));
  for (int i=0; i<3; i++) h_s->c[i] = s->getCenter()[i];
  h_s->r = s->getRadius();
  h_s->nMats = nMats;

  cudaMalloc((void **)&d_s, sizeof(DSphere));
  cudaMalloc((void **)&d_mats, nMats*sizeof(DMaterial*));
  htod_mats = (DMaterial**)malloc(nMats*sizeof(DMaterial*));
  for (int i=0; i<nMats; i++)
    htod_mats[i] = s->getMaterials()[i]->buildDMaterial();


  // Copy host pointers to GPU

  cudaMemcpy(d_s, h_s, sizeof(DSphere),
      cudaMemcpyHostToDevice);
  cudaMemcpy(d_mats, htod_mats, nMats*sizeof(DMaterial*),
      cudaMemcpyHostToDevice);
  iniMats<<<1,1>>>(d_s, d_mats);
}

SphereGPUAdapter::~SphereGPUAdapter(void) {

  int nMats = h_s->nMats;
  for (int i=0; i<nMats; i++)
    cudaFree(htod_mats[i]);

  free(htod_mats); cudaFree(d_mats);
  cudaFree(d_s); free(h_s);
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
