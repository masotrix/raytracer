#include <object.cuh>
#include <spheres.cuh>
using namespace std;

ObjectsGPUAdapter::ObjectsGPUAdapter(
    const vector<Object*> &objects) {

  // Init GPU point & assosiated pointers

  int nObjs = objects.size();
  DSphere **htod_objs, **sphs;

  cudaMalloc((void **)&sphs, nObjs*sizeof(DSphere*));
  htod_objs = (DSphere**)malloc(nObjs*sizeof(DSphere*));

  for (auto &obj: objects)
    obj_adapters.push_back(new SphereGPUAdapter((Sphere*)obj));

  for (int i=0; i<nObjs; i++)
    htod_objs[i] = obj_adapters[i]->getDS();

  // Copy host pointers to GPU

  cudaMemcpy(sphs, htod_objs, nObjs*sizeof(DSphere*),
      cudaMemcpyHostToDevice);

  h_sphs = (DSpheres*)malloc(sizeof(DSpheres));
  cudaMalloc((void **)&d_sphs, sizeof(DSpheres));
  h_sphs->s = sphs;
  h_sphs->nSpheres = nObjs;

  cudaMemcpy(d_sphs, h_sphs, sizeof(DSpheres),
      cudaMemcpyHostToDevice);

  free(htod_objs);
}

ObjectsGPUAdapter::~ObjectsGPUAdapter(void) {

  for (int i=0; i<obj_adapters.size(); i++)
    delete obj_adapters[i];

  cudaFree(d_sphs); cudaFree(h_sphs->s); free(h_sphs);
}
