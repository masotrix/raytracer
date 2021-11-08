#include <triangle.cuh>
#include <model.cuh>
#include <vector.cuh>
#include <vertex.cuh>
#include <mesh.cuh>
#include <ray.cuh>
using namespace std;

Triangle::Triangle(vector<Vertex*> &v): _v(v) { this->setN(); }

void Triangle::setN() {
  _n = normfv(crossf3(
    subfv(_v[1]->_p,_v[0]->_p), subfv(_v[2]->_p,_v[0]->_p)));
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

__device__ bool intersectTriangle(DTriangle *tri, DRay *ray,
    DCamera *cam) {


  float A[3][3],mini[3][3], det,alp,bet,gam,t1,b[3],x[3],
    *v0=tri->v[0],*v1=tri->v[1],*v2=tri->v[2],
    *n0=tri->n[0],*n1=tri->n[1],*n2=tri->n[2],
    wv0[3],wv1[3],wv2[3],wn0[3],wn1[3],wn2[3],
    *e=ray->eye,*d=ray->dir;

  b[0]=e[0]-v0[0]; b[1]=e[1]-v0[1]; b[2]=e[2]-v0[2];
  A[0][0]=v1[0]-v0[0]; A[0][1]=v2[0]-v0[0]; A[0][2]=-d[0];
  A[1][0]=v1[1]-v0[1]; A[1][1]=v2[1]-v0[1]; A[1][2]=-d[1];
  A[2][0]=v1[2]-v0[2]; A[2][1]=v2[2]-v0[2]; A[2][2]=-d[2];
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

  bet=x[0]; gam=x[1]; alp=1-bet-gam; t1=x[2];

  if (0<t1&&t1<ray->t&&0<=bet&&bet<=1&&0<=gam&&gam<=1&&0<=alp) {

    ray->t = t1;
    if (cam) {
      upscaf(wv0,v0,alp); upscaf(wv1,v1,bet); upscaf(wv2,v2,gam);

      upscaf(wn0,n0,alp);upscaf(wn1,n1,bet);upscaf(wn2,n2,gam);
      norm(sumf(ray->n,sumf(ray->n,wn0,wn1),wn2));

      sumf(ray->iPoint,sumf(ray->iPoint,wv0,wv1),wv2);
      ray->tri=tri; ray->s = NULL;
    }
    return true;
  }
  return false;
}

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
