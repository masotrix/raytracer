#ifndef OCTTREE_H
#define OCTTREE_H
#include <vector>
#include <object.cuh>

struct DRay; struct DCamera;

struct DTriangle;

enum DOctType {BRANCH, LEAVE};

struct DOctTree {
  DOctTree **child;
  DTriangle **triangles;
  unsigned short nTriangles;
  float ini[3], end[3];
  DOctType type;
};

// child accounts for current sub node for every node
// if it gets to 9, it contains not what is being searched for
struct DOctStack {
  DOctTree *oct[20];
  unsigned short size, child[20];
};

class OctNode: public Object {

  protected:
    std::vector<float> _ini, _end;

  public:
    virtual void insert(Object *obj)=0;
    virtual bool intersectsRay(const std::vector<float> &e,
      const std::vector<float> &d, std::vector<float> &iP,
      std::vector<float> &n, float &t,
      std::vector<const Material*>&mats);
    virtual bool intersectsChildren(const std::vector<float> &e,
      const std::vector<float> &d, std::vector<float> &iP,
      std::vector<float> &n, float &t,
      std::vector<const Material*>&mats)=0;

    bool intersectsObj(Object *obj){
      return obj->intersectsOctNode(_ini, _end);}
    bool intersectsOctNode(const std::vector<float>&ini,
        const std::vector<float>&end){return false;}
    virtual void countNode(int &count) const = 0;
    virtual void buildDNodes(DOctTree **h_nodes,
        DTriangle ***d_triangles, int *hierarchy, int *child_pos,
        int &index, int father_index, int pos) const = 0;
    virtual ~OctNode(){}
};

class OctBranch: public OctNode {

  private:
    std::vector<OctNode*> _children;

  public:
    OctBranch(const std::vector<float>&ini,
        const std::vector<float>&end);
    void insert(Object *obj);
    void upgrade(int slot, std::vector<Object*> objs,
        const std::vector<float>&ini,
        const std::vector<float>&end);
    virtual bool intersectsChildren(const std::vector<float> &e,
      const std::vector<float> &d, std::vector<float> &iP,
      std::vector<float> &n, float &t,
      std::vector<const Material*>&mats) {

      bool inter = false;
      for (auto &c: _children)
        inter = c->intersectsRay(e,d,iP,n,t,mats) || inter;
      return inter;
    }
    virtual ~OctBranch() {
      for (auto &c: _children) delete c;
    }
    void countNode(int &count) const override;
    void buildDNodes(DOctTree **h_nodes, DTriangle ***d_triangles,
        int *hierarchy, int *child_pos, int &index,
        int father_index, int pos) const override;
};

class OctLeave: public OctNode {

  private:
    std::vector<Object*> _objs; int _slot; OctBranch *_father;

  public:
    OctLeave(int, const std::vector<float>&,
        const std::vector<float>&,
        OctBranch *f);
    void insert(Object *obj);
    virtual bool intersectsChildren(const std::vector<float> &e,
      const std::vector<float> &d, std::vector<float> &iP,
      std::vector<float> &n, float &t,
      std::vector<const Material*>&mats) {

      bool inter = false;
      for (auto &obj: _objs)
        inter = obj->intersectsRay(e,d,iP,n,t,mats) || inter;
      return inter;
    }
    void countNode(int &count) const override;
    void buildDNodes(DOctTree **h_nodes, DTriangle ***d_triangles,
        int *hierarchy, int *child_pos, int &index,
        int father_index, int pos) const override;
};

class OctGPUAdapter {

  private:

    DOctTree **htod_nodes;
    DTriangle ***htod_nodeTriangles; int _count;

  public:

    OctGPUAdapter(const OctNode *o);
    DOctTree* getDOct(void) const {
      if (htod_nodes) return htod_nodes[0];
      else return NULL;
    }
    ~OctGPUAdapter(void);
};

__global__ void printNode(DOctTree*, int k);
__global__ void printNode2(DOctTree*, int k);

__device__ void initOctStack(DOctStack*);
__device__ void octStackPush(DOctStack*, DOctTree*, int child);
__device__ DOctTree* octStackPop(DOctStack *stack);
__device__ int topChild(DOctStack*);
__device__ void octStackAdd(DOctStack*);

__device__ bool intersectChildren(DOctTree*, DRay*, DCamera*);
__device__ void intersectOctNode(DOctTree*, DRay*, DCamera*, DOctStack*);
__global__ void initChildNode(DOctTree**, int, DOctTree*);
__global__ void printNodeTriangles(DOctTree*, int);

#endif /*OCTTREE_H*/
