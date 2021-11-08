#include <camera.cuh>
#include <ray.cuh>
#include <vector.cuh>
#include <ray.cuh>
#include <model.cuh>
using namespace std;

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

CameraGPUAdapter::CameraGPUAdapter(const Camera *c) {
  d_c = c->buildDCamera();
}

CameraGPUAdapter::~CameraGPUAdapter(void) {
  cudaFree(d_c);
}

__global__ void film(ubyte *img, DCamera *d_c, DLights *d_l,
    DSpheres *d_s, DOctTree *d_o) {

  DRay *ray, ray_s; ray = &ray_s;
  float ic,jc,d[3],coords[3],level[3]={d_c->scene_refr,0,0};
  uint W,H,S, i,j,c_pos; W=d_c->width; H=d_c->height; S=W*H;

  for (uint pixId = __mul24(blockIdx.x,blockDim.x)+threadIdx.x;
    pixId<S; pixId+=blockDim.x*gridDim.x) {

    i=pixId%W; j=pixId/W;

    ic = d_c->left+((d_c->right-d_c->left)/d_c->width)*(i+0.5);
    jc = d_c->bottom+((d_c->top-d_c->bottom)/d_c->height)*(j+0.5);
    iniVec(coords,ic,jc,-d_c->near);
    norm(linCom(d,d_c->u,d_c->v,d_c->w,coords));

    float color[3] = {0,0,0}, att[3] = {1,1,1}, attCol[3];
    DStack stack; initStack(&stack);
    stackPush(&stack, d_c->pos, d, level, 0, 0, att, d_o);

    while (!stackEmpty(&stack)) {
      copyRay(ray, stackPop(&stack));
      intersectRay(ray, d_s, d_c);
      colorateRay(&stack, ray, d_s, d_c, d_l);

      mulf(attCol, ray->att, ray->col);
      sumf(color, color, attCol);
    }
    color[0] = minf(color[0],1.f);
    color[1] = minf(color[1],1.f);
    color[2] = minf(color[2],1.f);
    upscaf(color,color, 255.f);

    c_pos = 3*(d_c->width*(d_c->height-1-j) + i);

    img[c_pos+0] = color[0];
    img[c_pos+1] = color[1];
    img[c_pos+2] = color[2];
  }
}
