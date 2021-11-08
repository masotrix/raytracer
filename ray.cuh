#ifndef RAY_H
#define RAY_H
#include <vector>
#include <stack>

struct DSphere; struct DSpheres; struct DTriangle;
struct DOctTree; struct DCamera; struct DLights;

struct DRay {
  float eye[3], dir[3],t,iPoint[3],n[3],col[3],att[3],level[3];
  DSphere *s; DTriangle *tri; DOctTree *oct;
  unsigned short depth,nLevel;
};

struct DStack {
  DRay ray[10];
  unsigned short size;
};

class Ray {

  private:

    std::vector<float> &_color;
    const std::vector<float> _e,_d;
    std::stack<float> _level;
    unsigned char _depth;

  public:

    Ray(std::vector<float> &color, const std::vector<float> &e,
        const std::vector<float> &d, std::stack<float> &level,
        unsigned char depth):
      _color(color),_e(e),_d(d),_level(level),_depth(depth) {}
    bool intersect(void);
};

__device__ void intersectRay(DRay*, DSpheres*, DCamera*);
__device__ char throwShadowRay(float pos[3], float dir[3],
    DSpheres*, DOctTree*, float lDist);
__device__ void colorateRay(DStack*, DRay*,
    DSpheres*, DCamera*, DLights*);

#endif /*RAY_H*/
