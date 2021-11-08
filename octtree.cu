#include <octtree.cuh>
#include <model.cuh>
#include <triangle.cuh>
#include <camera.cuh>
#include <ray.cuh>
#include <vector.cuh>
using namespace std;


OctLeave::OctLeave(int slot, const vector<float>&ini,
  const vector<float>&end, OctBranch *f): _slot(slot), _father(f)
  { _ini=ini; _end=end; }

void OctLeave::insert(Object *obj){
  if (this->intersectsObj(obj)){
    _objs.push_back(obj);
    if (_objs.size()>50)
      _father->upgrade(_slot, _objs, _ini, _end);
  }
}


void OctLeave::buildDNodes(DOctTree **h_nodes,
    DTriangle ***htod_triangles, int *hierarchy, int *child_pos,
    int &index, int father_index, int pos) const {

  hierarchy[index] = father_index;
  child_pos[index] = pos;

  h_nodes[index]->type = LEAVE;
  h_nodes[index]->child = NULL;

  h_nodes[index]->nTriangles = _objs.size();
  DTriangle **htod_ts =
    (DTriangle**)malloc(_objs.size()*sizeof(DTriangle*));
  for (int i=0; i<_objs.size(); i++)
    htod_ts[i] = t_map[(Triangle*)_objs[i]];
  cudaMalloc((void**)&h_nodes[index]->triangles,
      _objs.size()*sizeof(DTriangle*));

  cudaMemcpy(h_nodes[index]->triangles, htod_ts,
      _objs.size()*sizeof(DTriangle*), cudaMemcpyHostToDevice);
  free(htod_ts);

  for (int i=0; i<3; i++) {
    h_nodes[index]->ini[i] = _ini[i];
    h_nodes[index]->end[i] = _end[i];
  }

  index++;
}

void OctLeave::countNode(int &count) const {
  count++;
}

void OctBranch::countNode(int &count) const {
  count++;
  for (auto &c: _children)
    c->countNode(count);
}


void OctBranch::buildDNodes(DOctTree **h_nodes,
    DTriangle ***htod_triangles, int *hierarchy, int *child_pos,
    int &index, int father_index, int pos) const {

  hierarchy[index] = father_index;
  child_pos[index] = pos;

  h_nodes[index]->triangles = NULL;
  h_nodes[index]->nTriangles = 0;
  h_nodes[index]->type = BRANCH;
  cudaMalloc((void**)&h_nodes[index]->child,8*sizeof(DOctTree*));

  for (int i=0; i<3; i++) {
    h_nodes[index]->ini[i] = _ini[i];
    h_nodes[index]->end[i] = _end[i];
  }

  father_index = index;
  index++;
  for (int i=0; i<8; i++) {
    _children[i]->buildDNodes(h_nodes, htod_triangles, hierarchy,
        child_pos, index, father_index, i);
  }
}


OctBranch::OctBranch(const vector<float>&ini,
    const vector<float>&end){

  _ini = ini; _end = end;
  _children = vector<OctNode*>(8);
  vector<float> mid = doscafv(2,sumfv(end, ini));
  _children[0] = new OctLeave(0,
      {ini[0],ini[1],ini[2]},{mid[0],mid[1],mid[2]},this);
  _children[1] = new OctLeave(1,
      {mid[0],ini[1],ini[2]},{end[0],mid[1],mid[2]},this);
  _children[2] = new OctLeave(2,
      {ini[0],mid[1],ini[2]},{mid[0],end[1],mid[2]},this);
  _children[3] = new OctLeave(3,
      {mid[0],mid[1],ini[2]},{end[0],end[1],mid[2]},this);
  _children[4] = new OctLeave(4,
      {ini[0],ini[1],mid[2]},{mid[0],mid[1],end[2]},this);
  _children[5] = new OctLeave(5,
      {mid[0],ini[1],mid[2]},{end[0],mid[1],end[2]},this);
  _children[6] = new OctLeave(6,
      {ini[0],mid[1],mid[2]},{mid[0],end[1],end[2]},this);
  _children[7] = new OctLeave(7,
      {mid[0],mid[1],mid[2]},{end[0],end[1],end[2]},this);
}

bool OctNode::intersectsRay(const vector<float> &e,
    const vector<float> &d, vector<float> &iP,
    vector<float> &n, float &t, vector<const Material*>&mats){

  vector<vector<float>> A(3,vector<float>(3)),
    min(3,vector<float>(3)), v0,v1,v3; float det;
  vector<float> b(3),x(3);
  v0 = {{_end[0],_ini[1],_ini[2]},{_ini[0],_ini[1],_ini[2]},
        {_ini[0],_ini[1],_end[2]},{_end[0],_ini[1],_end[2]},
        {_ini[0],_end[1],_end[2]},{_ini[0],_ini[1],_ini[2]}};
  v1 = {{_ini[0],_ini[1],_ini[2]},{_ini[0],_ini[1],_end[2]},
        {_end[0],_ini[1],_end[2]},{_end[0],_ini[1],_ini[2]},
        {_end[0],_end[1],_end[2]},{_end[0],_ini[1],_ini[2]}};
  v3 = {{_end[0],_end[1],_ini[2]},{_ini[0],_end[1],_ini[2]},
        {_ini[0],_end[1],_end[2]},{_end[0],_end[1],_end[2]},
        {_ini[0],_end[1],_ini[2]},{_ini[0],_ini[1],_end[2]}};

  for (int i=0; i<6; i++) {

    A[0][0]=v1[i][0]-v0[i][0]; A[0][1]=v3[i][0]-v0[i][0];
    A[1][0]=v1[i][1]-v0[i][1]; A[1][1]=v3[i][1]-v0[i][1];
    A[2][0]=v1[i][2]-v0[i][2]; A[2][1]=v3[i][2]-v0[i][2];
    A[0][2]=-d[0]; b[0]=e[0]-v0[i][0];
    A[1][2]=-d[1]; b[1]=e[1]-v0[i][1];
    A[2][2]=-d[2]; b[2]=e[2]-v0[i][2];
    min[0][0]=A[1][1]*A[2][2]-A[1][2]*A[2][1];
    min[0][1]=-A[1][0]*A[2][2]+A[2][0]*A[1][2];
    min[0][2]=A[1][0]*A[2][1]-A[2][0]*A[1][1];
    min[1][0]=-A[0][1]*A[2][2]+A[2][1]*A[0][2];
    min[1][1]=A[0][0]*A[2][2]-A[2][0]*A[0][2];
    min[1][2]=-A[0][0]*A[2][1]+A[2][0]*A[0][1];
    min[2][0]=A[0][1]*A[1][2]-A[1][1]*A[0][2];
    min[2][1]=-A[0][0]*A[1][2]+A[1][0]*A[0][2];
    min[2][2]=A[0][0]*A[1][1]-A[1][0]*A[0][1];
    det=A[0][0]*min[0][0]+A[1][0]*min[1][0]+A[2][0]*min[2][0];
    x[0]=(min[0][0]*b[0]+min[1][0]*b[1]+min[2][0]*b[2])/det;
    x[1]=(min[0][1]*b[0]+min[1][1]*b[1]+min[2][1]*b[2])/det;
    x[2]=(min[0][2]*b[0]+min[1][2]*b[1]+min[2][2]*b[2])/det;

    float alp=x[0], bet=x[1], t1=x[2];

    if (0<t1&&t1<t && 0<=bet&&bet<=1 && 0<=alp&&alp<=1)
      return this->intersectsChildren(e,d,iP,n,t,mats);
  }

  return false;
}

void OctBranch::insert(Object *obj){
  if (this->intersectsObj(obj)){
    for (auto &c: _children) c->insert(obj);
  }
}

void OctBranch::upgrade(int slot, vector<Object*> objs,
    const vector<float>&ini, const vector<float>&end){

  OctBranch *newChild = new OctBranch(ini,end);
  delete _children[slot]; _children[slot] = newChild;
  for (auto &obj: objs) _children[slot]->insert(obj);
}

OctGPUAdapter::OctGPUAdapter(const OctNode *o) {

  htod_nodes = NULL; _count = 0; htod_nodeTriangles = NULL;

  if (!o) {
    /*htod_nodes = (DOctTree**)malloc(sizeof(DOctTree*));
    cudaMalloc((void**)&htod_nodes[0], sizeof(DOctTree));
    cudaFree(htod_nodes[0]);*/
  } else {
    o->countNode(_count);

    int *node_hierarchy, *node_child_pos; DOctTree **h_nodes;
    node_hierarchy = (int*)malloc(_count*sizeof(int));
    node_child_pos = (int*)malloc(_count*sizeof(int));
    h_nodes = (DOctTree**)malloc(_count*sizeof(DOctTree*));

    htod_nodeTriangles =
      (DTriangle***)malloc(_count*sizeof(DTriangle**));
    htod_nodes = (DOctTree**)malloc(_count*sizeof(DOctTree*));

    for (int i=0; i<_count; i++) {
      h_nodes[i] = (DOctTree*)malloc(sizeof(DOctTree));
      cudaMalloc((void**)&htod_nodes[i], sizeof(DOctTree));
    }

    int index = 0;
    o->buildDNodes(h_nodes, htod_nodeTriangles, node_hierarchy,
       node_child_pos, index, -1, -1);


    for (int i=_count-1; i>0; i--) {
      cudaMemcpy(htod_nodes[i], h_nodes[i], sizeof(DOctTree),
          cudaMemcpyHostToDevice);
      initChildNode<<<1,1>>>(h_nodes[node_hierarchy[i]]->child,
          node_child_pos[i], htod_nodes[i]);
    }
    cudaMemcpy(htod_nodes[0], h_nodes[0], sizeof(DOctTree),
        cudaMemcpyHostToDevice);

    //printNodeTriangles<<<1,1>>>(htod_nodes[0], 0);
    /*
    for (int i=0; i<_count; i++) {
      printNode<<<1,1>>>(htod_nodes[i], i);
    }*/
    free(h_nodes); free(node_hierarchy); free(node_child_pos);
  }
}

OctGPUAdapter::~OctGPUAdapter(void) {

  for (int i=0; i<_count; i++) {
    cudaFree(htod_nodeTriangles[i]);
    cudaFree(htod_nodes[i]);
  }
  free(htod_nodeTriangles);
  free(htod_nodes);
}

__global__ void printNode2(DOctTree *n, int k) {

  printf("Type: %d, Triangles: %d\n", n->type, n->nTriangles);
}

__device__ void initOctStack(DOctStack *stack) {
  stack->size = 0;
}

__device__ void octStackPush(DOctStack *stack, DOctTree* oct,
    int child) {

  stack->oct[stack->size] = oct;
  stack->child[stack->size] = child;
  stack->size++;
}

__device__ DOctTree* octStackPop(DOctStack *stack) {
  stack->size--;
  return stack->oct[stack->size];
}

__device__ int topChild(DOctStack *stack) {
  return stack->child[stack->size-1];
}

__device__ void octStackAdd(DOctStack *stack) {
  stack->child[stack->size-1]--;
}

__device__ bool intersectChildren(DOctTree *oct, DRay *ray,
    DCamera *cam) {

  bool inter = false; int i;

  for (i=0; i<oct->nTriangles; i++) {
    inter = intersectTriangle(oct->triangles[i],ray,cam);
  }

  return inter;
}

__device__ void intersectOctNode(DOctTree *oct, DRay *ray,
    DCamera *cam, DOctStack *stack) {

  float A[3][3],b[3],x[3],det,alp,bet,t1,*d=ray->dir,*e=ray->eye,
        mini[3][3], *ini=oct->ini,*end=oct->end,v0[][3] = {
          {end[0],ini[1],ini[2]},{ini[0],ini[1],ini[2]},
          {ini[0],ini[1],end[2]},{end[0],ini[1],end[2]},
          {ini[0],end[1],end[2]},{ini[0],ini[1],ini[2]}},
        v1[][3] = {
          {ini[0],ini[1],ini[2]},{ini[0],ini[1],end[2]},
          {end[0],ini[1],end[2]},{end[0],ini[1],ini[2]},
          {end[0],end[1],end[2]},{end[0],ini[1],ini[2]}},
        v3[][3] = {
          {end[0],end[1],ini[2]},{ini[0],end[1],ini[2]},
          {ini[0],end[1],end[2]},{end[0],end[1],end[2]},
          {ini[0],end[1],ini[2]},{ini[0],ini[1],end[2]}};

  for (int i=0; i<6; i++) {

    A[0][0]=v1[i][0]-v0[i][0]; A[0][1]=v3[i][0]-v0[i][0];
    A[1][0]=v1[i][1]-v0[i][1]; A[1][1]=v3[i][1]-v0[i][1];
    A[2][0]=v1[i][2]-v0[i][2]; A[2][1]=v3[i][2]-v0[i][2];
    A[0][2]=-d[0]; b[0]=e[0]-v0[i][0];
    A[1][2]=-d[1]; b[1]=e[1]-v0[i][1];
    A[2][2]=-d[2]; b[2]=e[2]-v0[i][2];
    mini[0][0]=A[1][1]*A[2][2]-A[1][2]*A[2][1];
    mini[0][1]=-A[1][0]*A[2][2]+A[2][0]*A[1][2];
    mini[0][2]=A[1][0]*A[2][1]-A[2][0]*A[1][1];
    mini[1][0]=-A[0][1]*A[2][2]+A[2][1]*A[0][2];
    mini[1][1]=A[0][0]*A[2][2]-A[2][0]*A[0][2];
    mini[1][2]=-A[0][0]*A[2][1]+A[2][0]*A[0][1];
    mini[2][0]=A[0][1]*A[1][2]-A[1][1]*A[0][2];
    mini[2][1]=-A[0][0]*A[1][2]+A[1][0]*A[0][2];
    mini[2][2]=A[0][0]*A[1][1]-A[1][0]*A[0][1];
    det=A[0][0]*mini[0][0]+A[1][0]*mini[1][0]+A[2][0]*mini[2][0];
    x[0]=(mini[0][0]*b[0]+mini[1][0]*b[1]+mini[2][0]*b[2])/det;
    x[1]=(mini[0][1]*b[0]+mini[1][1]*b[1]+mini[2][1]*b[2])/det;
    x[2]=(mini[0][2]*b[0]+mini[1][2]*b[1]+mini[2][2]*b[2])/det;

    alp=x[0]; bet=x[1]; t1=x[2];

    if (0<t1&&t1<ray->t && 0<=bet&&bet<=1 && 0<=alp&&alp<=1) {
      octStackPush(stack,oct,0); return;
    }
  }
}

__global__ void printNodeTriangles(DOctTree *n, int k) {

  switch (n->type) {

    case BRANCH:
      printf("Empty Branch: %d\n", k);
      for (int i=0; i<8; i++)
        printNodeTriangles<<<1,1>>>(n->child[i],++k);
      return;

    case LEAVE:
      printf("Leave %d: \n", k);
      for (int i=0; i<n->nTriangles; i++)
        printTriangle<<<1,1>>>(n->triangles[i]);
      return;
  }
}

__global__ void printNode(DOctTree *n, int k) {

  printf("Type: %d, Triangles: %d\n", n->type, n->nTriangles);
}

__global__ void initChildNode(DOctTree **children, int pos,
    DOctTree* child) { children[pos] = child; }
