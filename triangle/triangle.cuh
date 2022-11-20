#ifndef TRIANGLE_H
#define TRIANGLE_H
#include <vector>
#include <object.cuh>

class Mesh; class Light;
class Vertex; class Material;
struct DCamera; struct DRay; struct DMaterial;

struct DTriangle {
  float v[3][3], uv[3][2], n[3][3];
  bool uvA, nA; int nMats;
  DMaterial **mats;
};

class Triangle: public Object {

  public:
    std::vector<Vertex*> _v;
    std::vector<float> _n; Mesh *_mesh;
    Triangle(std::vector<Vertex*> &v);
    bool intersectsRay(const std::vector<float>&e,
      const std::vector<float>&d, std::vector<float>&iPoint,
      std::vector<float>&n, float&t,
      std::vector<const Material*>&mats) override;
    bool intersectsOctNode(const std::vector<float>&ini,
      const std::vector<float>&end);
    void setN();
};

__device__ bool intersectTriangle(DTriangle *tri, DRay *ray,
    DCamera *cam);
__global__ void printTriangle(DTriangle *t);

#endif /*TRIANGLE_H*/
