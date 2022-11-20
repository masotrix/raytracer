#include <vector>
#include <ray.cuh>
#include <object.cuh>
#include <vector.cuh>
#include <octtree.cuh>
#include <material.cuh>
#include <lights.cuh>
#include <spheres.cuh>
#include <camera.cuh>
#include <triangle.cuh>
#include <model.cuh>
using namespace std;

bool Ray::intersect(void){

  vector<float> iPoint, n; float t = INF;
  vector<const Material*> mats;

  for (auto &anObj: allObjects)
    anObj->intersectsRay(_e,_d,iPoint,n,t,mats);

  if (t < INF) {
    for (auto &mat: mats)
      mat->colorate(_color,iPoint,n,_d,_level,_depth,t);
    return true;
  } else { _color=sumfv(_color,BACKGROUND_COLOR); return false; }
}

__device__ void intersectRay(DRay *ray, DSpheres *sphs,
    DCamera *cam) {

  DOctTree oct_s[20], *oct; DOctStack *octStack, octStack_s;
  octStack = &octStack_s; int child;
  for (int i=0; i<20; i++) octStack->oct[i] = &oct_s[i];
  initOctStack(octStack);
  if (ray->oct) {
    octStackPush(octStack, ray->oct, 0);
  }

  while (octStack->size) {

    child = topChild(octStack);
    oct = octStackPop(octStack);
    if (oct->type == LEAVE) {
      intersectChildren(oct, ray, cam);
    } else if (child < 8) {
      octStackPush(octStack, oct, child+1);
      intersectOctNode(oct->child[child], ray, cam, octStack);
    } else continue;
  }

  for (int is=0; is<sphs->nSpheres; is++)
    intersectSphere(sphs->s[is],sphs,ray,cam);
}

__device__ char throwShadowRay(float pos[3], float dir[3],
    DSpheres *sphs, DOctTree *oct, float lDist) {

  DRay ray;
  for (int i=0; i<3; i++) {
    ray.eye[i] = pos[i];
    ray.dir[i] = dir[i];
  }
  ray.s     = NULL;
  ray.tri   = NULL;
  ray.t     = DINF;
  ray.oct   = oct;
  intersectRay(&ray, sphs, NULL);

  if (lDist>ray.t) return 1;
  else return 0;
}

__device__ void colorateRay(DStack *stack, DRay *ray,
    DSpheres *sphs, DCamera *cam, DLights *ls) {

  if (ray->s) {
    for (int im=0; im<ray->s->nMats; im++) {
      colorate(stack,ray->s->mats[im],sphs,ls,cam,ray);
    }
  } else if (ray->tri) {
    for (int im=0; im<ray->tri->nMats; im++) {
      colorate(stack, ray->tri->mats[im], sphs, ls, cam, ray);
    }
  }
  else copyVec(ray->col, cam->bg_col);
}
