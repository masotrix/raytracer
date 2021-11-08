#include <cstdio>
#include <iostream>
#include <main.cuh>
using namespace std;

vector<Object*> allObjects;
vector<Mesh*> allMeshes;
vector<Light*> allLights;
map<std::string,Material*> allMaterials;
map<Triangle*,DTriangle*> t_map;

extern unsigned short W,H;

int main(int argc, char **argv) {
  
  clock_t begin = clock();
  Camera *cam = NULL;
  string img_name = parse(argc, argv, cam); 
 
  CameraGPUAdapter *ca = new CameraGPUAdapter(cam);
  LightsGPUAdapter *la = new LightsGPUAdapter(allLights);
  ObjectsGPUAdapter *oa = new ObjectsGPUAdapter(allObjects);
  MeshGPUAdapter *ma = new MeshGPUAdapter(allMeshes);
  OctNode *octN = NULL; OctGPUAdapter *octa = NULL;
  if (allMeshes.size()) {
    octN = new OctBranch(
      vector<float>{-INF/4,-INF/4,INF/4},
      vector<float>{INF/4,INF/4,-INF/4});
    for (auto &mesh: allMeshes)
      mesh->insertInto(octN);
  }
  octa = new OctGPUAdapter(octN);

  cout << "Oct taken seconds: ";
  cout << (clock()-begin)/(double)CLOCKS_PER_SEC << endl;
  begin = clock();

  size_t reqBlocksCount;
  ushort threadsCount,blocksCount;
  threadsCount   = 128;
  reqBlocksCount = W*H / threadsCount;
  blocksCount    = (ushort)min((size_t)32768, reqBlocksCount);

  ubyte *d_img, *h_img;
  h_img = (ubyte*)malloc(W*H*3*sizeof(ubyte));
  cudaMalloc((void **)&d_img, W*H*3*sizeof(ubyte));

  film<<<blocksCount,threadsCount>>>(d_img, ca->getDCamera(),
      la->getDLights(),oa->getDObjects(), octa->getDOct());

  cudaError_t cudaerr = cudaDeviceSynchronize();
  if (cudaerr != cudaSuccess)
    printf("kernel launch failed with error \"%s\".\n",
        cudaGetErrorString(cudaerr));

  cudaMemcpy(h_img, d_img, W*H*3*sizeof(ubyte),
      cudaMemcpyDeviceToHost);

  write_image(img_name, W, H, h_img);

  delete octa; delete octN;
  delete ca; delete la; delete oa; delete cam; delete ma;
  for (auto &obj: allObjects) delete obj;
  for (auto &mesh: allMeshes) delete mesh;
  for (auto &light: allLights) delete light;
  for (auto &mat: allMaterials) delete mat.second;
  
  cout << "Rendering seconds: ";
  cout << (clock()-begin)/(double)CLOCKS_PER_SEC << endl;
}
