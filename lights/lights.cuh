#ifndef LIGHTS_H
#define LIGHTS_H
#include <vector>

struct DCamera; struct DMaterial; struct DSpheres;
struct DRay;

enum DLigType {PUNCTUAL, SPOT, AMBIENT, DIRECTIONAL};

struct DLight {
  float col[3], pos[3], dir[3], angle;
  DLigType type;
};

struct DLights {
  DLight **l;
  unsigned short nLights;
};

class Light {

  protected:
    std::vector<float> _color;

  public:
    Light(const std::vector<float> color);
    virtual ~Light(void);
    virtual void iluminateLambert(
        std::vector<float> &color,
        const std::vector<float>&mat_color,
        const std::vector<float>&iSphere,
        const std::vector<float> &n) = 0;
    virtual void iluminatePhong(
        std::vector<float> &color,
        const std::vector<float> &mat_color,
        const std::vector<float> &iSphere,
        const std::vector<float> &n,
        const std::vector<float> &d, int shi) = 0;
    virtual DLight* buildDLight(void) const = 0;
};

class PunctualLight : public Light {

  private:
    std::vector<float> _pos;

  public:
    PunctualLight(std::vector<float> color,
        std::vector<float> pos);
    virtual ~PunctualLight(void);
    void iluminateLambert(
        std::vector<float> &color,
        const std::vector<float>&mat_color,
        const std::vector<float>&iPoint,
        const std::vector<float> &n)
      override;
    void iluminatePhong(
        std::vector<float> &color,
        const std::vector<float> &mat_color,
        const std::vector<float> &iPoint,
        const std::vector<float> &n,
        const std::vector<float> &d, int shi) override;
    DLight* buildDLight(void) const override;
};

class SpotLight : public Light {

  private:
    std::vector<float> _pos, _dir;
    float _angle;

  public:
    SpotLight(std::vector<float> color, std::vector<float> pos,
        float angle, std::vector<float> dir);
    virtual ~SpotLight(void);
    void iluminateLambert(
        std::vector<float> &color,
        const std::vector<float>&mat_color,
        const std::vector<float>&iPoint,
        const std::vector<float> &n)
      override;
    void iluminatePhong(
        std::vector<float> &color,
        const std::vector<float> &mat_color,
        const std::vector<float> &iPoint,
        const std::vector<float> &n,
        const std::vector<float> &d, int shi) override;
    DLight* buildDLight(void) const override;
};

class AmbientLight : public Light {

  public:
    AmbientLight(std::vector<float> color);
    virtual ~AmbientLight(void);
    void iluminateLambert(
        std::vector<float> &color,
        const std::vector<float>&mat_color,
        const std::vector<float>&iPoint,
        const std::vector<float> &n) override;
    void iluminatePhong(
        std::vector<float> &color,
        const std::vector<float> &mat_color,
        const std::vector<float> &iPoint,
        const std::vector<float> &n,
        const std::vector<float> &d, int shi) override;
    DLight* buildDLight(void) const override;
};

class DirectionalLight : public Light {

  private:
    std::vector<float> _dir;

  public:
    DirectionalLight(std::vector<float> color,
        std::vector<float> dir);
    virtual ~DirectionalLight(void);
    void iluminateLambert(
        std::vector<float> &color,
        const std::vector<float>&mat_color,
        const std::vector<float>&iPoint,
        const std::vector<float> &n)
      override;
    void iluminatePhong(
        std::vector<float> &color,
        const std::vector<float> &mat_color,
        const std::vector<float> &iPoint,
        const std::vector<float> &n,
        const std::vector<float> &d, int shi) override;
    DLight* buildDLight(void) const override;
};

class LightsGPUAdapter {

  private:

    DLights *d_l, *h_l; DLight **htod_l;
    int _nLights;

  public:

    LightsGPUAdapter(const std::vector<Light*> &l);
    ~LightsGPUAdapter(void);
    DLights* getDLights(void) const { return d_l; }
};

__device__ void iluminateLambert(DLight*, DCamera*,
    const DMaterial*, DSpheres*, DRay*);
__device__ void iluminatePhong(DLight*, DCamera*,
    const DMaterial*, DSpheres*, DRay*);

#endif /*LIGHTS_H*/
