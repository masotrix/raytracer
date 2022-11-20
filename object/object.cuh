#ifndef OBJECT_H
#define OBJECT_H
#include <vector>

class Material; class SphereGPUAdapter;
struct DSpheres;

class Object {

  protected:
    std::vector<const Material *> _mats;

  public:
    Object(void){};
    virtual ~Object(void){};
    virtual bool intersectsRay(const std::vector<float> &e,
        const std::vector<float> &d, std::vector<float> &iSphere,
        std::vector<float> &n, float &t,
        std::vector<const Material*>&mats) = 0;
    virtual bool intersectsOctNode(const std::vector<float>&ini,
        const std::vector<float>&end)=0;
    const std::vector<const Material*>& getMaterials(void) const {
      return _mats;
    }
};

class ObjectsGPUAdapter {

  private:

    DSpheres *d_sphs, *h_sphs;
    std::vector<SphereGPUAdapter*> obj_adapters;

  public:

    ObjectsGPUAdapter(const std::vector<Object*> &objects);
    ~ObjectsGPUAdapter(void);
    DSpheres* getDObjects(void) const { return d_sphs; }
};

#endif /*OBJECT_H*/
