#include <display.h>
#include <model.h>
#include <cmath>
#include <limits>
#include <numeric>
#include <algorithm>
#include <iostream>
using namespace std;

#define EPSILON 0.001 // 10^-3

__global__ void printTriangle(DTriangle *t) {
  
  float *v,*uv,*n;
  for (int i=0; i<3; i++) { 
    v=t->v[i], uv=t->uv[i], n=t->n[i];
    printf("Vertex %d:\n", i);
    printf("  V%d: %f %f %f\n", i, v[0],v[1],v[2]);
    if (t->uvA)
      printf("  UV%d: %f %f\n", i, uv[0],uv[1]);
    if (t->nA)
      printf("  N%d: %f %f %f\n", i, n[0],n[1],n[2]);
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

__global__ void printVal(void) {
  printf("Hola!!\n");
}


vector<float> sumfv(vector<float> v1, const vector<float> &v2){
  transform(v1.begin(),v1.end(),v2.begin(),v1.begin(),
      plus<float>()); return v1;
}

vector<float> subfv(vector<float> v1, const vector<float> &v2){
  transform(v1.begin(),v1.end(),v2.begin(),v1.begin(),
      minus<float>()); return v1;
}

vector<float> mulfv(vector<float> v1, const vector<float> &v2){
  transform(v1.begin(),v1.end(),v2.begin(),v1.begin(),
      multiplies<float>()); return v1;
}

vector<float> upscafv(float c, vector<float> v){
  transform(v.begin(),v.end(),v.begin(),
      bind1st(multiplies<float>(),c)); return v;
}

vector<float> doscafv(float c, vector<float> v){
  transform(v.begin(),v.end(),v.begin(),
      bind2nd(divides<float>(),c)); return v;
}

void matmul(const vector<vector<float>> &mat, vector<float> &v) {
  float v2[3];
  for (int i=0; i<3; i++) 
    v2[i] = mat[i][0]*v[0]+mat[i][1]*v[1]+mat[i][2]*v[2];

  for (int i=0; i<3; i++) v[i] = v2[i];
}

struct exptiate{float operator()(float f)const {return exp(f);}};
vector<float> expfv(vector<float> v){
  transform(v.begin(),v.end(),v.begin(),exptiate()); return v;
}

float dotfv(const vector<float> &v1, const vector<float> &v2){
  return inner_product(v1.begin(),v1.end(),v2.begin(),0.0);
}

vector<float> crossf3(const vector<float> &u, 
    const vector<float> &v){ return 
  {u[1]*v[2]-u[2]*v[1],u[2]*v[0]-u[0]*v[2],u[0]*v[1]-u[1]*v[0]};
}

float magfv(const vector<float> &v){
  return sqrt(dotfv(v,v));
}

vector<float> normfv(const vector<float> &v){
  return doscafv(magfv(v),v);
}

float accfv(const vector<float> &v){
  return accumulate(v.begin(),v.end(),0.0);
}

void printVector(const vector<float> &v) {
  for (auto &c: v) cout << c << ' ';
  cout << endl;
}

vector<float> refract(const vector<float> &d,
 const vector<float> &n, float rori) {

  float costhe = -dotfv(d,n), sinthe = sqrt(1-pow(costhe,2)),
    sinfi = rori*sinthe, cosfi = sqrt(1-pow(sinfi,2));

  vector<float> nn;
  if (costhe < 0){ nn = upscafv(-1,n); costhe = -costhe; }
  else nn = n;

  vector<float> b = doscafv(sinthe,sumfv(d,upscafv(costhe,nn))),
    rr = subfv(upscafv(sinfi,b),upscafv(cosfi,nn));

  return rr;
}


// Data Types

Triangle::Triangle(vector<Vertex*> &v): _v(v) {
  this->setN();
}

void Triangle::setN() {
  _n = normfv(crossf3(
        subfv(_v[1]->_p,_v[0]->_p),
        subfv(_v[2]->_p,_v[0]->_p)));
}



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

void LambertMaterial::colorate(vector<float> &color,
    const vector<float> &iSphere, const vector<float> &n, 
    const vector<float> &d, stack<float> &level,
    unsigned char depth, float t) const {

  for (auto &light: allLights){
    light->iluminateLambert(color, _color, iSphere, n);
  }
}

DMaterial *LambertMaterial::buildDMaterial(void) const {
  
  DMaterial *mat = (DMaterial*)malloc(sizeof(DMaterial)), *dmat;
  cudaMalloc((void **)&dmat, sizeof(DMaterial));
  
  mat->type = LAMBERT;
  for (int i=0; i<3; i++)
    mat->col[i] = _color[i];

  cudaMemcpy(dmat,mat, sizeof(DMaterial),cudaMemcpyHostToDevice);
  free(mat);

  return dmat;
}

void PhongMaterial::colorate(vector<float> &color,
    const vector<float> &iSphere, const vector<float> &n,
    const vector<float> &d, stack<float> &level,
    unsigned char depth, float t) const {
 
  for (auto &light: allLights){
    light->iluminatePhong(color, _color, iSphere, n, d, _shi);
  }
}

DMaterial *PhongMaterial::buildDMaterial(void) const {
  
  DMaterial *mat = (DMaterial*)malloc(sizeof(DMaterial)), *dmat;
  cudaMalloc((void **)&dmat, sizeof(DMaterial));
  
  mat->type = PHONG;
  mat->param = _shi;
  for (int i=0; i<3; i++)
    mat->col[i] = _color[i];

  cudaMemcpy(dmat,mat, sizeof(DMaterial),cudaMemcpyHostToDevice);
  free(mat);

  return dmat;
}

void ReflectiveMaterial::colorate(vector<float> &color,
    const vector<float> &iPoint, const vector<float> &n, 
    const vector<float> &d, stack<float> &level,
    unsigned char depth, float t) const {

  if (depth<DEPTH) {
    vector <float> r,sPoint,refColor(3,0.0); Ray *ray;

    r      = normfv(subfv(d,upscafv(2*dotfv(d,n),n)));
    sPoint = sumfv(iPoint, upscafv(EPSILON,r));
    ray    = new Ray(refColor, sPoint, r, level, depth+1);
    ray->intersect();

    delete ray;
    color  = sumfv(color,mulfv(_color,refColor));
  }
}

DMaterial *ReflectiveMaterial::buildDMaterial(void) const {
  
  DMaterial *mat = (DMaterial*)malloc(sizeof(DMaterial)), *dmat;
  cudaMalloc((void **)&dmat, sizeof(DMaterial));
  
  mat->type = REFLECTIVE;
  for (int i=0; i<3; i++)
    mat->col[i] = _color[i];

  cudaMemcpy(dmat,mat, sizeof(DMaterial),cudaMemcpyHostToDevice);
  free(mat);

  return dmat;
}



void DielectricMaterial::colorate(vector<float> &color,
    const vector<float> &iPoint, const vector<float> &n, 
    const vector<float> &d, stack<float> &level,
    unsigned char depth, float t) const {

  if (depth<DEPTH) { float refr,rOrI,R0,fresnel,cosine,totRefl; 
    vector <float> rl,rlsPoint,rlColor, rr,rrsPoint,rrColor,K,ni;
    Ray *rayRl,*rayRr; refr = level.top();
    
    rl       = normfv(subfv(d,upscafv(2*dotfv(d,n),n)));
    rlsPoint = sumfv(iPoint, upscafv(EPSILON,rl));
    rlColor  = vector<float>(3,0.0);
    rayRl    = new Ray(rlColor, rlsPoint, rl, level, depth+1);
    rayRl->intersect(); delete rayRl;
    R0       = pow((_refr-refr)/(_refr+refr),2);


    if (dotfv(d,n)<0){  
      rOrI     = refr/_refr; 
      totRefl  = 1-pow(rOrI,2)*(1-pow(dotfv(d,n),2));
      K        = vector<float>(3,1.0);
      if (totRefl >= 0){
        rr = refract(d,n,rOrI);
        cosine   = -dotfv(d,n);
        rrsPoint = sumfv(iPoint, upscafv(EPSILON,rr));
        rrColor  = vector<float>(3,0.0);
        level.push(_refr);
        rayRr    = new Ray(rrColor,rrsPoint,rr,level,depth+1); 
        level.pop(); rayRr->intersect(); delete rayRr;
      } else {
        color    =  sumfv(color,mulfv(K,mulfv(_color,rlColor)));
        return;
      }
    } else {
      rOrI     = _refr/refr; 
      totRefl  = 1-pow(rOrI,2)*(1-pow(dotfv(d,n),2));
      K        = expfv(upscafv(-t,_att));
      if (totRefl >= 0){
        ni       = upscafv(-1,n);
        rr = refract(d,ni,rOrI);
        cosine   = -dotfv(d,ni);
        rrsPoint = sumfv(iPoint, upscafv(EPSILON,rr));
        rrColor  = vector<float>(3,0.0);
        level.pop();
        rayRr    = new Ray(rrColor,rrsPoint,rr,level,depth+1);
        level.push(refr); rayRr->intersect(); delete rayRr;
      } else {
        color    =  sumfv(color,mulfv(K,mulfv(_color,rlColor)));
        return;
      }
    }
    fresnel  = R0+(1-R0)*pow(1.0-cosine,5);
    color    = sumfv(color,mulfv(K,mulfv(_color,sumfv(
                  upscafv(fresnel,rlColor),
                  upscafv((1-fresnel),rrColor)))));
  }
}

DMaterial *DielectricMaterial::buildDMaterial(void) const {
  
  DMaterial *mat = (DMaterial*)malloc(sizeof(DMaterial)), *dmat;
  cudaMalloc((void **)&dmat, sizeof(DMaterial));
  
  mat->type = DIELECTRIC;
  mat->param = _refr;
  for (int i=0; i<3; i++) {
    mat->col[i] = _color[i];
    mat->att[i] = _att[i];
  }

  cudaMemcpy(dmat,mat, sizeof(DMaterial),cudaMemcpyHostToDevice);
  free(mat);

  return dmat;
}

TextureMaterial::TextureMaterial(string mat_file) {

  _file = mat_file;
}

void TextureMaterial::colorate(vector<float>&color,
        const vector<float>&iPoint, const vector<float> &n,
        const vector<float>&d, stack<float> &level,
        unsigned char depth, float t) const {

  // CPU
}

DMaterial *TextureMaterial::buildDMaterial() const {

  DMaterial *mat = (DMaterial*)malloc(sizeof(DMaterial)), *dmat;
  cudaMalloc((void**)&dmat, sizeof(DMaterial));

  float *tex = (float*)malloc(_W*_H*3*sizeof(float));
  // Llenar tex con lo de la imagen
  // (en mat_file) (con lo de la parte de jpg y png)

  cudaMalloc((void**)&d_tex, _W*_H*3*sizeof(float));
  cudaMemcpy(mat->tex,tex, _W*_H*3*sizeof(float),
      cudaMemcpyHostToDevice);
  mat->type = TEXTURE;
  mat->tex  = d_tex;
  

  cudaMemcpy(dmat,mat, sizeof(DMaterial),cudaMemcpyHostToDevice);
  free(mat);

  return dmat;
}

TextureMaterial::~TextureMaterial(void) {

  cudaFree(d_tex);
}

void PunctualLight::iluminateLambert(vector<float> &color,
   const vector<float> &mat_color, const vector<float> &iPoint,
   const vector<float> &n){

  vector<float> l,sPoint, dumI,dumN;
  float lDist, lambert, objDist=INF;
  vector<const Material*>dumM(0);

  l       = normfv(subfv(_pos,iPoint));
  sPoint  = sumfv(iPoint,upscafv(EPSILON,l));
  lDist   = magfv(subfv(_pos,sPoint));

  if (lDist && SHADOWS_ENABLED) {
    for (auto &anObj: allObjects)
      anObj->intersectsRay(sPoint,l,dumI,dumN,objDist,dumM);
  }

  if (lDist > objDist) return;

  lambert = max(dotfv(n,l),0.f);
  color = sumfv(color,upscafv(lambert,mulfv(mat_color,_color)));
}

void PunctualLight::iluminatePhong(vector<float> &color,
    const vector<float> &mat_color, const vector<float> &iPoint,
    const vector<float> &n, const vector<float> &d, int shi){

  vector<float> sPoint,v,l, dumI,dumN;
  float lDist, phong, objDist=INF;
  vector<const Material*>dumM(0);
  l       = normfv(subfv(_pos,iPoint));
  sPoint  = sumfv(iPoint,upscafv(EPSILON,l));
  lDist   = magfv(subfv(_pos,sPoint));

  if (lDist && SHADOWS_ENABLED) {
    for (auto &anObj: allObjects)
        anObj->intersectsRay(sPoint,l,dumI,dumN,objDist,dumM);
  }

  if (lDist>objDist) return;

  v = upscafv(-1,d);
  phong = pow(max(dotfv(n,normfv(sumfv(v,l))),0.f), shi);
  color = sumfv(color,upscafv(phong,mulfv(mat_color,_color)));
}

DLight* PunctualLight::buildDLight(void) const {
  
  DLight *l = (DLight*)malloc(sizeof(DLight)), *d_l;
  cudaMalloc((void **)&d_l, sizeof(DLight));
  
  l->type = PUNCTUAL;
  for (int i=0; i<3; i++){
    l->pos[i]=_pos[i];
    l->col[i]=_color[i];
  } 

  cudaMemcpy(d_l, l, sizeof(DLight),cudaMemcpyHostToDevice);
  free(l);

  return d_l;
}


void SpotLight::iluminateLambert(vector<float> &color,
   const vector<float> &mat_color, const vector<float> &iPoint,
   const vector<float> &n) {

  vector<float> l,sPoint, dumI,dumN; 
  float lDist, lambert, objDist=INF; 
  vector<const Material*>dumM(0);
  l       = normfv(subfv(_pos,iPoint));
  sPoint  = sumfv(iPoint,upscafv(EPSILON,l));
  lDist   = magfv(subfv(_pos,sPoint));

  if (lDist && SHADOWS_ENABLED) {
    for (auto &anObj: allObjects)
        anObj->intersectsRay(sPoint,l,dumI,dumN,objDist,dumM);
  }

  if (lDist > objDist) return;

  if (-dotfv(l,_dir) > cos(_angle/2)){
    lambert = max(dotfv(n,l),0.f);
    color=sumfv(color,upscafv(lambert,mulfv(mat_color,_color)));
  }
}

void SpotLight::iluminatePhong(vector<float> &color,
    const vector<float> &mat_color, const vector<float> &iPoint,
    const vector<float> &n, const vector<float> &d, int shi) {

  vector<float> sPoint,v,l, dumI,dumN; 
  float lDist, phong, objDist=INF; 
  vector<const Material*>dumM(0);
  l       = normfv(subfv(_pos,iPoint));
  sPoint  = sumfv(iPoint,upscafv(EPSILON,l));
  lDist   = magfv(subfv(_pos,sPoint));

  if (lDist && SHADOWS_ENABLED) {
    for (auto &anObj: allObjects)
      anObj->intersectsRay(sPoint,l,dumI,dumN,objDist,dumM);
  }

  if (lDist > objDist) return;
  
  if (-dotfv(l,_dir) > cos(_angle/2)){
    v     = upscafv(-1,d);
    phong = pow(max(dotfv(n,normfv(sumfv(v,l))),0.f), shi);
    color = sumfv(color,upscafv(phong,mulfv(mat_color,_color)));
  }
}

DLight* SpotLight::buildDLight(void) const {
  
  DLight *l = (DLight*)malloc(sizeof(DLight)), *d_l;
  cudaMalloc((void **)&d_l, sizeof(DLight));
  
  l->type = SPOT;
  l->angle = _angle;
  for (int i=0; i<3; i++){
    l->pos[i]=_pos[i];
    l->col[i]=_color[i];
    l->dir[i]=_dir[i];
  } 

  cudaMemcpy(d_l, l, sizeof(DLight),cudaMemcpyHostToDevice);
  free(l);

  return d_l;
}


DLight* AmbientLight::buildDLight(void) const {

  DLight *l = (DLight*)malloc(sizeof(DLight)), *d_l;
  cudaMalloc((void **)&d_l, sizeof(DLight));
  
  l->type = AMBIENT;
  for (int i=0; i<3; i++){
    l->col[i]=_color[i];
  } 

  cudaMemcpy(d_l, l, sizeof(DLight),cudaMemcpyHostToDevice);
  free(l);

  return d_l;
}

void DirectionalLight::iluminateLambert(vector<float> &color,
   const vector<float> &mat_color, const vector<float> &iPoint,
   const vector<float> &n){

  vector<float> sPoint,l, dumI,dumN; 
  float lDist, lambert, objDist=INF; 
  vector<const Material*>dumM(0);
  l      = upscafv(-1,_dir);
  sPoint = sumfv(iPoint,upscafv(EPSILON,l));
  lDist  = INF;

  if (lDist && SHADOWS_ENABLED) {
    for (auto &anObj: allObjects)
      anObj->intersectsRay(sPoint,l,dumI,dumN,objDist,dumM);
  }

  if (lDist > objDist) return;
  
  lambert= max(dotfv(n,l),0.f);
  color  = sumfv(color,upscafv(lambert,mulfv(mat_color,_color)));
}

void DirectionalLight::iluminatePhong(vector<float> &color,
    const vector<float> &mat_color, const vector<float> &iPoint,
    const vector<float> &n, const vector<float> &d, int shi){

  vector<float> sPoint,v,l, dumI, dumN; 
  float lDist, phong, objDist=INF; 
  vector<const Material*>dumM(0);
  l      = upscafv(-1,_dir);
  sPoint = sumfv(iPoint,upscafv(EPSILON,l));
  lDist  = INF;

  if (lDist && SHADOWS_ENABLED) {
    for (auto &anObj: allObjects)
      anObj->intersectsRay(sPoint,l,dumI,dumN,objDist,dumM);
  }

  if (lDist > objDist) return;

  v = upscafv(-1,d);
  phong = pow(max(dotfv(n,normfv(sumfv(v,l))),0.f), shi);
  color = sumfv(color,upscafv(phong,mulfv(mat_color,_color)));
}

DLight* DirectionalLight::buildDLight(void) const {
  
  DLight *l = (DLight*)malloc(sizeof(DLight)), *d_l;
  cudaMalloc((void **)&d_l, sizeof(DLight));
  
  l->type = DIRECTIONAL;
  for (int i=0; i<3; i++){
    l->col[i]=_color[i];
    l->dir[i]=_dir[i];
  }

  cudaMemcpy(d_l, l, sizeof(DLight),cudaMemcpyHostToDevice);
  free(l);

  return d_l;
}

bool Triangle::intersectsRay(const vector<float> &e,
    const vector<float> &d, vector<float> &iPoint,
    vector<float> &n, float &t, vector<const Material*>&mats){

  vector<vector<float>> A(3,vector<float>(3)),
    min(3,vector<float>(3)); float det;
  vector<float> b(3),x(3),v0=_v[0]->_p,v1=_v[1]->_p,v2=_v[2]->_p;
  b[0]=e[0]-v0[0]; b[1]=e[1]-v0[1]; b[2]=e[2]-v0[2];
  A[0][0]=v1[0]-v0[0]; A[0][1]=v2[0]-v0[0]; A[0][2]=-d[0];
  A[1][0]=v1[1]-v0[1]; A[1][1]=v2[1]-v0[1]; A[1][2]=-d[1];
  A[2][0]=v1[2]-v0[2]; A[2][1]=v2[2]-v0[2]; A[2][2]=-d[2];
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

  float bet=x[0], gam=x[1], alp=1-bet-gam, t1=x[2];

  if (0<t1&&t1<t && 0<=bet&&bet<=1 && 0<=gam&&gam<=1 && 0<=alp) {
  
    t      = t1;
    iPoint = sumfv(sumfv(
              upscafv(alp,v0),
              upscafv(bet,v1)),
              upscafv(gam,v2));
    n      = normfv(sumfv(sumfv(
              upscafv(alp,_v[0]->_n),
              upscafv(bet,_v[1]->_n)),
              upscafv(gam,_v[2]->_n)));
    mats   = _mesh->getMaterials();

    return true;
  }
  else return false;
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


bool Triangle::intersectsOctNode(const vector<float>&ini,
    const vector<float>&end) {

  vector<vector<float>> axes(9), box_norms;
  box_norms={{1,0,0},{0,1,0},{0,0,1}};
  for (int i=0; i<3; i++) {
    axes[3*i+0]=crossf3(subfv(_v[1]->_p,_v[0]->_p),box_norms[i]);
    axes[3*i+1]=crossf3(subfv(_v[2]->_p,_v[1]->_p),box_norms[i]);
    axes[3*i+2]=crossf3(subfv(_v[0]->_p,_v[2]->_p),box_norms[i]);
  }

  float mint,maxt,minb,maxb,proy;
  for (int i=0; i<9; i++) {

    if (!dotfv(axes[i],axes[i])) continue;

    mint=INF; maxt=-INF; minb=INF; maxb=-INF;
    for (int j=0; j<3; j++){
      proy = dotfv(_v[j]->_p,axes[i]);
      maxt = max(maxt,proy); mint = min(mint,proy);
    }

    proy = dotfv({ini[0],ini[1],ini[2]},axes[i]);
    maxb = max(maxb,proy); minb = min(minb,proy);
    proy = dotfv({end[0],ini[1],ini[2]},axes[i]);
    maxb = max(maxb,proy); minb = min(minb,proy);
    proy = dotfv({ini[0],end[1],ini[2]},axes[i]);
    maxb = max(maxb,proy); minb = min(minb,proy);
    proy = dotfv({ini[0],ini[1],end[2]},axes[i]);
    maxb = max(maxb,proy); minb = min(minb,proy);
    proy = dotfv({end[0],end[1],ini[2]},axes[i]);
    maxb = max(maxb,proy); minb = min(minb,proy);
    proy = dotfv({end[0],ini[1],end[2]},axes[i]);
    maxb = max(maxb,proy); minb = min(minb,proy);
    proy = dotfv({ini[0],end[1],end[2]},axes[i]);
    maxb = max(maxb,proy); minb = min(minb,proy);
    proy = dotfv({end[0],end[1],end[2]},axes[i]);
    maxb = max(maxb,proy); minb = min(minb,proy);

    if (maxt < minb || mint > maxb) return false;
    else continue;
  }

  return true;
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

bool Mesh::intersectsRay(const vector<float> &e,
    const vector<float> &d, vector<float> &iPoint,
    vector<float> &n, float &t, vector<const Material*>&mats){

  bool clash = false;
  for (auto &tri: _t)
    if (tri->intersectsRay(e, d, iPoint, n, t, mats)){
      clash = true; }
  
  return clash;
}


bool Sphere::intersectsRay(const vector<float> &e,
    const vector<float> &d, vector<float> &iSphere,
    vector<float> &n, float &t, vector<const Material*>&mats){

  float B,C,det,t1,t2;
  vector <float> ec;

  ec  = subfv(e,_c);
  B   = 2*dotfv(ec,d);
  C   = dotfv(ec,ec)-_r*_r;
  det = B*B - 4*C;

  if (det>=0) {
    det = sqrt(det);
    t2=-B+det; t1=-B-det;
    if (t2>=0) {

      if (t1>0 && t1<2*t) { t1/=2; t=t1; }
      else if (t1<0 && t2<2*t) { t2/=2; t=t2; }
      else return false;

      iSphere = sumfv(e,upscafv(t,d));
      n      = normfv(subfv(iSphere,_c));
      mats   = _mats;
      return true;
    }
  }
  return false;
}

void Camera::film(string img_name){

  vector<float> w,u,v,d;
  float top,bottom,right,left,ic,jc; 
  vector<unsigned char> img(_width*_height*3);

  w = normfv(subfv(_e,_target));
  u = normfv(crossf3(_up,w));
  v = crossf3(w,u);
  top = _near*tan(_fov/2);
  bottom=-top;
  right = _width*top/_height;
  left = -right;

  for (int i=0; i<_width; i++){
    for (int j=0; j<_height; j++){
      
      ic=left+((right-left)/_width)*(i+0.5);
      jc=bottom+((top-bottom)/_height)*(j+0.5);
      d = normfv(sumfv(upscafv(ic,u),subfv(upscafv(jc,v),
              upscafv(_near,w))));

      vector<float> color(3,0.0);
      stack<float> level; level.push(SCENE_REFR);
      Ray *ray = new Ray(color, _e, d, level, 0);
      ray->intersect();
      delete ray;

      color[0]=min(color[0],1.f); color[1]=min(color[1],1.f);
      color[2]=min(color[2],1.f); color = upscafv(255,color);
      int c_pos = 3*(i+_width*(_height-1-j));
      img[c_pos]=color[0];
      img[c_pos+1]=color[1];
      img[c_pos+2]=color[2];
    }
  }

  //write_image(img_name, _width, _height, &img[0]);
}

DCamera* Camera::buildDCamera(void) const {

  DCamera *cam = (DCamera*)malloc(sizeof(DCamera)), *dcam;
  cudaMalloc((void **)&dcam, sizeof(DCamera));
  
  vector<float> w,u,v;
  w = normfv(subfv(_e,_target));
  u = normfv(crossf3(_up,w));
  v = crossf3(w,u);
  float top,bottom,right,left; 
  top = _near*tan(_fov/2); bottom=-top;
  right = _width*top/_height; left = -right;

  for (int i=0; i<3; i++) {
    cam->u[i] = u[i]; cam->v[i] = v[i]; cam->w[i] = w[i];
    cam->pos[i]=_e[i]; cam->bg_col[i] = BACKGROUND_COLOR[i];
  }

  cam->top=top; cam->bottom=bottom;
  cam->right=right; cam->left=left;
  cam->width=_width; cam->height=_height;
  cam->near = _near; cam->shadows = SHADOWS_ENABLED;
  cam->scene_refr = SCENE_REFR; cam->ray_depth = DEPTH;
  cam->epsilon = EPSILON; cam->inf = INF;

  cudaMemcpy(dcam,cam, sizeof(DCamera),cudaMemcpyHostToDevice);
  free(cam);

  return dcam;
}


/**   ---> GPU functions and adapters <---   **/


__global__ void iniMats(DSphere *s, DMaterial **mats) {
  s->mats = mats;
}

CameraGPUAdapter::CameraGPUAdapter(const Camera *c) {
  d_c = c->buildDCamera();
}

CameraGPUAdapter::~CameraGPUAdapter(void) {
  cudaFree(d_c);
}

LightsGPUAdapter::LightsGPUAdapter(const vector<Light*> &l) {

  // Init GPU point & assosiated pointers

  _nLights = l.size();
  DLight **d_ls;

  htod_l = (DLight**)malloc(_nLights*sizeof(DLight*));
  for (int i=0; i<_nLights; i++)
    htod_l[i] = l[i]->buildDLight();

  cudaMalloc((void **)&d_ls, l.size()*sizeof(DLight*));
  cudaMemcpy(d_ls, htod_l, l.size()*sizeof(DLight*),
      cudaMemcpyHostToDevice);

  h_l = (DLights*)malloc(sizeof(DLights));
  h_l->l = d_ls; h_l->nLights = _nLights;
  cudaMalloc((void **)&d_l, sizeof(DLights));
  cudaMemcpy(d_l, h_l, sizeof(DLights),
      cudaMemcpyHostToDevice);
}

LightsGPUAdapter::~LightsGPUAdapter(void) {

  for (int i=0; i<_nLights; i++) cudaFree(htod_l[i]);
  free(htod_l); cudaFree(d_l); cudaFree(h_l->l); free(h_l);
}


SphereGPUAdapter::SphereGPUAdapter(Sphere *s) {

  // Init GPU point & assosiated pointers

  int nMats = s->getMaterials().size();
  h_s = (DSphere*)malloc(sizeof(DSphere));
  for (int i=0; i<3; i++) h_s->c[i] = s->getCenter()[i];
  h_s->r = s->getRadius();
  h_s->nMats = nMats;

  cudaMalloc((void **)&d_s, sizeof(DSphere));
  cudaMalloc((void **)&d_mats, nMats*sizeof(DMaterial*));
  htod_mats = (DMaterial**)malloc(nMats*sizeof(DMaterial*));
  for (int i=0; i<nMats; i++)
    htod_mats[i] = s->getMaterials()[i]->buildDMaterial();


  // Copy host pointers to GPU

  cudaMemcpy(d_s, h_s, sizeof(DSphere),
      cudaMemcpyHostToDevice);
  cudaMemcpy(d_mats, htod_mats, nMats*sizeof(DMaterial*),
      cudaMemcpyHostToDevice);
  iniMats<<<1,1>>>(d_s, d_mats);
}

SphereGPUAdapter::~SphereGPUAdapter(void) {

  int nMats = h_s->nMats;
  for (int i=0; i<nMats; i++)
    cudaFree(htod_mats[i]);

  free(htod_mats); cudaFree(d_mats);
  cudaFree(d_s); free(h_s);
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

