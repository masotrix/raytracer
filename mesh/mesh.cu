#include <mesh.cuh>
#include <triangle.cuh>
#include <material.cuh>
#include <vertex.cuh>
#include <vector.cuh>
#include <octtree.cuh>
#include <model.cuh>
using namespace std;

Mesh::Mesh(const std::pair<std::vector<Triangle*>,
    std::vector<Vertex*>>&md,
    const std::vector<const Material*>&mats):
  _t(md.first),_v(md.second) { _mats = mats;
  for (auto &t: _t) t->_mesh = this;
}

Mesh::~Mesh() {
  for (auto &t: _t) delete t;
  for (auto &v: _v) delete v;
}

void Mesh::scale(const vector<float> &sca) {
  for (int i=0; i<_v.size(); i++) {
    _v[i]->_p = mulfv(_v[i]->_p, sca);
  }
}

void Mesh::rotate(const vector<float> &rot) {

  vector<vector<float>> rotMat(3,vector<float>(3));
  float a=M_PI*rot[0]/180,b=M_PI*rot[1]/180,c=M_PI*rot[2]/180;

  rotMat = {{1,0,0},{0,cos(a),-sin(a)},{0,sin(a),cos(a)}};
  for (int i=0; i<_v.size(); i++) matmul(rotMat, _v[i]->_p);
  rotMat = {{cos(b),0,sin(b)},{0,1,0},{-sin(b),0,cos(b)}};
  for (int i=0; i<_v.size(); i++) matmul(rotMat, _v[i]->_p);
  rotMat = {{cos(c),-sin(c),0},{sin(c),cos(c),0},{0,0,1}};
  for (int i=0; i<_v.size(); i++) matmul(rotMat, _v[i]->_p);

  for (int i=0; i<_t.size(); i++) _t[i]->setN();
  for (int i=0; i<_v.size(); i++) _v[i]->setN();
}

void Mesh::translate(const vector<float> &trans) {
  for (int i=0; i<_v.size(); i++) {
    _v[i]->_p = sumfv(_v[i]->_p, trans);
  }
}

void Mesh::buildDMats(DMaterial ***htod_mats,
    DMaterial ***htod_mat, int *nMats) {

  cudaMalloc((void**)htod_mats, _mats.size()*sizeof(DMaterial*));
  *htod_mat=(DMaterial**)malloc(_mats.size()*sizeof(DMaterial*));
  *nMats = _mats.size();

  for (int i=0; i<_mats.size(); i++) {
    (*htod_mat)[i] = _mats[i]->buildDMaterial();
  }

  cudaMemcpy(*htod_mats,*htod_mat,
      _mats.size()*sizeof(DMaterial*), cudaMemcpyHostToDevice);
}

void Mesh::buildDTriangles(DTriangle **htod_triangles,
    DMaterial **mats, int nMats, int &index) {

  DTriangle *h_tri;
  for (int i=0; i<_t.size(); i++) {
    h_tri = (DTriangle*)malloc(sizeof(DTriangle));

    for (int j=0; j<3; j++)
      for (int k=0; k<3; k++)
        h_tri->v[j][k] = _t[i]->_v[j]->_p[k];

    if (_t[0]->_v[0]->_n.size()) {
      for (int j=0; j<3; j++)
        for (int k=0; k<3; k++)
          h_tri->n[j][k] = _t[i]->_v[j]->_n[k];
    } else h_tri->nA = false;
    if (_t[0]->_v[0]->_uv.size()) {
      for (int j=0; j<3; j++)
        for (int k=0; k<2; k++)
          h_tri->uv[j][k] = _t[i]->_v[j]->_uv[k];
    } else h_tri->uvA = false;

    h_tri->mats = mats;
    h_tri->nMats = nMats;

    cudaMemcpy(htod_triangles[index], h_tri, sizeof(DTriangle),
        cudaMemcpyHostToDevice);

    t_map[_t[i]] = htod_triangles[index];
    //printTriangle<<<1,1>>>(htod_triangles[index]);
    free(h_tri); index++;
  }
}


void Mesh::insertInto(OctNode *n) {
  for (auto &t: _t) n->insert(t);
}

bool Mesh::intersectsRay(const vector<float> &e,
    const vector<float> &d, vector<float> &iPoint,
    vector<float> &n, float &t, vector<const Material*>&mats){

  bool clash = false;
  for (auto &tri: _t)
    if (tri->intersectsRay(e, d, iPoint, n, t, mats)){
      clash = true; }

  return clash;
}

MeshGPUAdapter::MeshGPUAdapter(vector<Mesh*> ms) {

  _count = 0; _meshes = ms.size();
  for (auto m: ms)
    m->countTriangles(_count);

  htod_triangles=(DTriangle**)malloc(_count*sizeof(DTriangle*));
  for (int i=0; i<_count; i++)
    cudaMalloc((void**)&htod_triangles[i], sizeof(DTriangle));

  int index = 0;
  htod_mats=(DMaterial***)malloc(ms.size()*sizeof(DMaterial**));
  htod_mat=(DMaterial***)malloc(ms.size()*sizeof(DMaterial**));
  nMats = (int*)malloc(ms.size()*sizeof(int));
  for (int i=0; i<ms.size(); i++) {
    ms[i]->buildDMats(&htod_mats[i], &htod_mat[i], &nMats[i]);
    ms[i]->buildDTriangles(htod_triangles, htod_mats[i],
        nMats[i], index);
  }

  //for (int i=0; i<_count; i++)
    //printTriangle<<<1,1>>>(htod_triangles[i]);
}

MeshGPUAdapter::~MeshGPUAdapter(void) {

  for (int i=0; i<_meshes; i++) {
    for (int j=0; j<_count; j++) {
      cudaFree(htod_mat[i][j]);
      cudaFree(htod_triangles[j]);
    }
    free(htod_mat[i]);
    cudaFree(htod_mats[i]);
  }
  free(htod_triangles);
  free(htod_mats);
  free(htod_mat);
  free(nMats);
}
