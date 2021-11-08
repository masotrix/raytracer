#ifndef SPHERES_H
#define SPHERES_H
#include <object.cuh>

struct DMaterial; struct DRay; struct DCamera;

struct DSphere {
  float c[3], r;
  int nMats;
  DMaterial **mats;
};

struct DSpheres {
  DSphere **s;
  unsigned short nSpheres;
};

class Sphere: public Object {

  private:
    float _r; std::vector<float> _c;

  public:
    Sphere(float r,const std::vector<float>&c,
        std::vector<const Material*>&mats): _r(r), _c(c)
      { _mats = mats; }
    virtual ~Sphere(void){};
    bool intersectsRay(const std::vector<float>&e,
        const std::vector<float>&d,
        std::vector<float>&iSphere, std::vector<float>&n, float&t,
        std::vector<const Material*>&mats) override;
    virtual bool intersectsOctNode(const std::vector<float>&ini,
        const std::vector<float>&end){ return false; }
    const std::vector<float>& getCenter(void) const { return _c; }
    float getRadius(void) { return _r; }
};

class SphereGPUAdapter {

  private:

    DSphere *h_s,*d_s;
    DMaterial **d_mats, **htod_mats;

  public:

    SphereGPUAdapter(Sphere *s);
    DSphere* getDS(void) const {return d_s;}
    ~SphereGPUAdapter(void);
};

__device__ char intersectSphere(DSphere*, DSpheres*, DRay*, DCamera*);

#endif /*SPHERES_H*/
