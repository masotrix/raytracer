#ifndef MESH_H
#define MESH_H
#include <vector>
#include <object.cuh>

class OctNode; class Triangle; class Vertex;
class Material;

struct DTriangle; struct DMaterial;

class Mesh: public Object {

  private:
    std::vector<Triangle*> _t; std::vector<Vertex*> _v;

  public:
    Mesh(const std::pair<std::vector<Triangle*>,
        std::vector<Vertex*>>&md,
        const std::vector<const Material*>&mats);
    bool intersectsRay(const std::vector<float>&e,
        const std::vector<float>&d,
        std::vector<float>&iPoint, std::vector<float>&n, float&t,
        std::vector<const Material*>&mats) override;
    virtual bool intersectsOctNode(const std::vector<float>&ini,
        const std::vector<float>&end){return false;}
    void insertInto(OctNode *n);
    void countTriangles(int &count) { count += _t.size(); }
    void buildDTriangles(DTriangle **htod_triangles,
       DMaterial **mats, int nMats, int &index);
    void buildDMats(DMaterial ***htod_mats, DMaterial ***htod_mat,
        int *nMats);
    void scale(const std::vector<float> &sca);
    void rotate(const std::vector<float> &rot);
    void translate(const std::vector<float> &trans);
    ~Mesh();
};

class MeshGPUAdapter {

  private:

    DTriangle **htod_triangles;
    DMaterial ***htod_mats, ***htod_mat;
    int _count,_meshes, *nMats;

  public:

    MeshGPUAdapter(std::vector<Mesh*> meshes);
    ~MeshGPUAdapter(void);
};

#endif /*MESH_H*/
