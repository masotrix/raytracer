#ifndef MODEL_H
#define MODEL_H
#include <vector>
#include <map>

class Object; class Mesh; class Material;
class Triangle; class Light; class Camera;
class Vertex;
struct DTriangle;

extern std::vector<Object*> allObjects;
extern std::vector<Mesh*> allMeshes;
extern std::vector<Light*> allLights;
extern std::map<std::string,Material*> allMaterials;
extern std::map<Triangle*,DTriangle*> t_map;
extern std::vector<float> BACKGROUND_COLOR;
extern unsigned short W,H,DEPTH;
extern float SCENE_REFR;
extern bool SHADOWS_ENABLED;

static float INF = 64;
static float EPSILON = 0.001;

#define DINF 64
#define DEPSILON 0.001
#define ubyte unsigned char

#endif /*MODEL_H*/
