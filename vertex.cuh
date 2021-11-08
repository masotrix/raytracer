#ifndef VERTEX_H
#define VERTEX_H
#include <vector>

class Triangle;

class Vertex {

  private:
    std::vector<const Triangle *> _t;

  public:
    std::vector<float> _p,_uv,_n;
    Vertex(const std::vector<float> &p);
    void addT(const Triangle *t);
    void setN(void);
};

#endif /*VERTEX_H*/
