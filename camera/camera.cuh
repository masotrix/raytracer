#ifndef CAMERA_H
#define CAMERA_H
#include <string>
#include <vector>
#define NEAR 0.1

struct DLights; struct DSpheres;
struct DOctTree;

struct DCamera {
  float u[3],v[3],w[3],pos[3],
        left,right,bottom,top,
        scene_refr,bg_col[3],
        near,epsilon,inf;
  unsigned short width,height,ray_depth;
  bool shadows;
};

class Camera {

  private:

    unsigned short _width,_height;
    std::vector<float> _e,_up,_target;
    float _fov,_near;


  public:

    Camera(unsigned short W, unsigned short H, float fov,
        std::vector<float> &e, std::vector<float> &up,
        std::vector<float> &target): _width(W),_height(H),_e(e),
    _up(up),_target(target),_fov(fov),_near(NEAR) {}
    void film(std::string img_name);
    DCamera* buildDCamera(void) const;
};


class CameraGPUAdapter {

  private:

    DCamera *d_c;

  public:

    CameraGPUAdapter(const Camera *c);
    ~CameraGPUAdapter(void);
    DCamera* getDCamera(void) const { return d_c; }
};

__global__ void film(unsigned char*, DCamera*, DLights*,
  DSpheres*, DOctTree*);

#endif /*CAMERA_H*/
