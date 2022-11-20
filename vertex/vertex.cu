#include <vertex.cuh>
#include <triangle.cuh>
#include <vector.cuh>

Vertex::Vertex(const std::vector<float> &p): _p(p) {}
void Vertex::addT(const Triangle *t){ _t.push_back(t); }

void Vertex::setN(void){
  std::vector<float> n(3,0);
  for (auto &t: _t) n = sumfv(n,t->_n);
  _n = normfv(n);
}
