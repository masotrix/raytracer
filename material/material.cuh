#ifndef MATERIAL_H
#define MATERIAL_H
#include <vector>
#include <stack>
#include <string>

struct DStack; struct DSphere; struct DLights;
struct DSpheres; struct DCamera; struct DRay;

enum DMatType {LAMBERT, PHONG, REFLECTIVE, DIELECTRIC, TEXTURE};

struct DMaterial {
  float col[3], att[3], param, *tex;
  DMatType type;
};

class Material {

  protected:
    std::vector<float> _color;

  public:
    virtual ~Material(){};
    virtual void colorate(std::vector<float> &color,
       const std::vector<float> &iSphere,
       const std::vector<float> &n,
       const std::vector<float> &d, std::stack<float> &level,
       unsigned char depth, float t)
      const = 0;
    virtual DMaterial* buildDMaterial() const = 0;
};

class LambertMaterial : public Material {

  public:
    LambertMaterial(std::vector<float> color) { _color = color; }
    virtual ~LambertMaterial(){};
    void colorate(std::vector<float>&color,
        const std::vector<float>&iSphere,
        const std::vector<float>&n,
        const std::vector<float>&d, std::stack<float> &level,
        unsigned char depth, float t)
      const override;
    DMaterial* buildDMaterial() const override;
};

class PhongMaterial : public Material {

  private:
    float _shi;

  public:
    PhongMaterial(std::vector<float> color, float shi):
      _shi(shi) { _color=color; }
    virtual ~PhongMaterial(){};
    void colorate(std::vector<float>&color,
        const std::vector<float>&iSphere,
        const std::vector<float>&n, const std::vector<float>&d,
        std::stack<float> &level, unsigned char depth, float t)
      const override;
    DMaterial* buildDMaterial() const override;
};

class ReflectiveMaterial : public Material {

  public:
    ReflectiveMaterial(std::vector<float> color) { _color=color; }
    virtual ~ReflectiveMaterial(){};
    void colorate(std::vector<float>&color,
        const std::vector<float>&iPoint,
        const std::vector<float> &n,
        const std::vector<float>&d, std::stack<float> &level,
        unsigned char depth, float t) const override;
    DMaterial* buildDMaterial() const override;
};

class DielectricMaterial : public Material {

  private:
    std::vector<float> _att;
    float _refr;

  public:
    DielectricMaterial(std::vector<float> color,
        std::vector<float> att, float refr):
      _att(att), _refr(refr) { _color=color; }
    virtual ~DielectricMaterial(){}
    void colorate(std::vector<float>&color,
        const std::vector<float>&iPoint,
        const std::vector<float> &n,
        const std::vector<float>&d, std::stack<float> &level,
        unsigned char depth, float t) const override;
    DMaterial* buildDMaterial() const override;
};

class TextureMaterial : public Material {

  private:
    float *d_tex; std::string _file;
    unsigned short _W,_H;

  public:
    TextureMaterial(std::string mat_path);
    ~TextureMaterial();
    void colorate(std::vector<float>&color,
        const std::vector<float>&iPoint,
        const std::vector<float> &n,
        const std::vector<float>&d, std::stack<float> &level,
        unsigned char depth, float t) const override;
    DMaterial* buildDMaterial() const override;
};

__device__ void colorate(DStack*, const DMaterial*, DSpheres*,
  const DLights*, DCamera*, DRay*);

__global__ void iniMats(DSphere *s, DMaterial **mats);

#endif /*MATERIAL_H*/
