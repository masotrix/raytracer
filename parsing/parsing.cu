#include <cmath>
#include <boost/property_tree/json_parser.hpp>
#include <parsing.cuh>
#include <vertex.cuh>
#include <camera.cuh>
#include <lights.cuh>
#include <triangle.cuh>
#include <material.cuh>
#include <mesh.cuh>
#include <vector.cuh>
#include <spheres.cuh>
using namespace std;
using namespace boost;

unsigned short W,H,DEPTH;
vector<float> BACKGROUND_COLOR = {0, 0, 0};
float SCENE_REFR = 1.0;
bool SHADOWS_ENABLED = true;


static pair<vector<Triangle*>,vector<Vertex*>> loadMesh(
    string filename){

  string s, token;
  ifstream file(filename);
  if (!file.is_open()) { 
    printf("No %s has been found\n", filename.c_str());
    exit(1);
  }

  vector<vector<float>> v,uv,n;
  vector<Triangle*> triangles;
  istringstream iss;

  do {
    getline(file, s);
    iss.clear(); iss.str(s);
    getline(iss, token, ' ');
  } while (token != "v");

  while (token == "v") {

    v.push_back({
        stof((getline(iss,token,' '),token)),
        stof((getline(iss,token,' '),token)),
        stof((getline(iss,token,' '),token))
    });
    
    getline(file, s);
    iss.clear(); iss.str(s);
    getline(iss, token, ' ');
  }

  while (token == "vt") {

    uv.push_back({
        stof((getline(iss,token,' '),token)),
        stof((getline(iss,token,' '),token))
    });
    
    getline(file, s);
    iss.clear(); iss.str(s);
    getline(iss, token, ' ');
  }

  while (token == "vn") {

    n.push_back({
        stof((getline(iss,token,' '),token)),
        stof((getline(iss,token,' '),token)),
        stof((getline(iss,token,' '),token))
    });
    
    getline(file, s);
    iss.clear(); iss.str(s);
    getline(iss, token, ' ');
  }


  while (token != "f") {
    getline(file, s);
    iss.clear(); iss.str(s);
    getline(iss, token, ' ');
  }

  map<int,Vertex*> vertex;

  while (token == "f") {
    
    vector<Vertex*> vt;

    for (int i=0; i<3; i++) {
       
      int vin, uvin=-1, nin=-1;
      if (n.size())
        vin = stoul((getline(iss,token,'/'),token))-1;
      else vin = stoul((getline(iss,token,' '),token))-1;
      if (uv.size())
        uvin = stoul((getline(iss,token,'/'),token))-1;
      else if (n.size()) getline(iss,token,'/');
      if (n.size())
        nin = stoul((getline(iss,token,' '),token))-1;

      if (vertex.find(vin)==vertex.end()){
        vertex[vin] = new Vertex(v[vin]);
        if (uvin!=-1) vertex[vin]->_uv = uv[uvin];
        if (nin!=-1) vertex[vin]->_n = n[nin];
      }
      vt.push_back(vertex[vin]);
    }

    triangles.push_back(new Triangle(vt));
    for (auto &v: vt) v->addT(triangles[triangles.size()-1]);
    
    getline(file, s);
    iss.clear(); iss.str(s);
    getline(iss, token, ' ');
  }
  file.close();
  
  if (!n.size()) for (auto &v: vertex) v.second->setN();
  
  pair<vector<Triangle*>,vector<Vertex*>> mesh_data;
  mesh_data.first = triangles;
  vector<Vertex*> allVertex;
  transform(vertex.begin(),vertex.end(),back_inserter(allVertex),
      [](auto &kv) { return kv.second; });
  mesh_data.second = allVertex;
  return mesh_data;
}

static bool option(char **begin, char **end, const string&option){
    return find(begin, end, option) != end;
}

static char *getOption(char **begin, char **end,
    const string &option){

    char **itr = find(begin,end,option);
    if (itr!=end && ++itr!=end) return *itr;
    return 0;
}


string parse(int ac, char **av, Camera *&camera){


  if (ac<2 || !option(av,av+ac,"-s")) {

    printf("Usage: ./raytracer -w W -h H -s scene.json ");
    printf("-r resources.txt -i image.png");
    exit(0);
  }

  string deflts_str; 
  string imgname("image.png"), resfile("resources.json");
  vector<string> deflts={"Using following defaults:\n"};
  if (option(av,av+ac,"-w")) 
    W=atoi(getOption(av,av+ac,"-w"));
  else {deflts.push_back("+ Width: "+to_string(512)+"\n"); W=512;}
  if (option(av,av+ac,"-h")) 
    H=atoi(getOption(av,av+ac,"-h"));
  else {deflts.push_back("+ Height: "+to_string(512)+"\n");H=512;}
  if (option(av,av+ac,"-m")) 
    DEPTH=atoi(getOption(av,av+ac,"-m"));
  else {deflts.push_back("+ MaxRayDepth: "+to_string(3)+"\n");
    DEPTH=3;}
  if (option(av,av+ac,"-r"))
    resfile = string(getOption(av,av+ac,"-r"));
  else deflts.push_back("+ Resources: resources.json\n");
  if (option(av,av+ac,"-i")) 
    imgname = string(getOption(av,av+ac,"-i"));
  else deflts.push_back("+ Image: image.png\n");
  for (const auto &deflt: deflts) deflts_str += deflt;
  printf("%s", deflts_str.c_str());

  ifstream fs(resfile);
  property_tree::ptree world_objects, world_resourses; 
  property_tree::read_json(fs, world_resourses);

  for (auto &res_prop: world_resourses) {

    if (res_prop.first == "__type__"){
      continue;
    }

    if (res_prop.first == "materials"){
      for (auto &material: res_prop.second) {
        vector<float> mat_color, mat_attenuation; 
        string mat_type, mat_name, mat_file;
        int shininess; float refraction_index;
        for (auto &mat_prop: material.second) {

          if (mat_prop.first == "__type__"){
            mat_type = mat_prop.second.get_value<string>();
            continue;
          }

          if (mat_prop.first == "name"){
            mat_name = mat_prop.second.get_value<string>();
            continue;
          }

          if (mat_prop.first == "color"){
            for (auto &coord: mat_prop.second){
              mat_color.push_back(
                  stof(coord.second.get_value<string>()));
            } continue;
          }

          if (mat_prop.first == "brdfParams"){
            for (auto &param: mat_prop.second){
              if (param.first == "shininess"){
                shininess = 
                  stoi(param.second.get_value<string>());
                continue;
              }
            } continue;
          }

          if (mat_prop.first == "attenuation"){
            for (auto &coord: mat_prop.second){
              mat_attenuation.push_back(
                  stof(coord.second.get_value<string>()));
            } continue;
          }

          if (mat_prop.first == "refraction_index"){
            refraction_index = 
              stof(mat_prop.second.get_value<string>());
            continue;
          }

          if (mat_prop.first == "brdf"){
            mat_type = mat_prop.second.get_value<string>();
            continue;
          }

          if (mat_prop.first == "use_for_ambient"){
            continue;
          }
        }

        if (mat_type=="lambert") {
          Material *mat = new LambertMaterial(mat_color);
          allMaterials[mat_name] = mat;
        }
        else if (mat_type=="blinnPhong") {
          Material *mat = new PhongMaterial(mat_color, 
              shininess);
          allMaterials[mat_name] = mat;
        }
        else if (mat_type=="reflective_material") {
          Material *mat = new ReflectiveMaterial(mat_color);
          allMaterials[mat_name] = mat;
        }
        else if (mat_type=="dielectric_material") {
          Material *mat = new DielectricMaterial(mat_color,
              mat_attenuation, refraction_index);
          allMaterials[mat_name] = mat;
        }
        else if (mat_type=="color_texture") {
          Material *mat = new TextureMaterial(mat_file);
          allMaterials[mat_name] = mat;
        }
      }
      continue;
    }
  }

  fs = ifstream(getOption(av,av+ac,"-s"));
  property_tree::read_json(fs, world_objects);
  vector<float> e,up,target; float fov;

  for (auto &world_prop: world_objects) {

    if (world_prop.first == "__type__"){
      continue;
    }

    if (world_prop.first == "camera"){
      for (auto &cam_prop: world_prop.second){

        if (cam_prop.first == "__type__")
          continue;

        if (cam_prop.first == "fov"){
          fov=M_PI*stof(cam_prop.second.get_value<string>())/180;
          continue;
        }

        if (cam_prop.first == "position"){
          for (auto &coord: cam_prop.second){
            e.push_back(stof(coord.second.get_value<string>()));
          } continue;
        }

        if (cam_prop.first == "up"){
          for (auto &coord: cam_prop.second){
            up.push_back(stof(coord.second.get_value<string>()));
          } continue;
        }

        if (cam_prop.first == "target"){
          for (auto &coord: cam_prop.second){
            target.push_back(
                stof(coord.second.get_value<string>()));
          } continue;
        }
      }
      camera = new Camera(W,H,fov,e,up,target);
      continue;
    }

    if (world_prop.first == "objects"){
      for (auto &object: world_prop.second) {
        vector<float> pos,sca,rot,trans; float radis;
        vector<const Material *> obj_mats;
        string obj_type, file_path;
        for (auto &obj_prop: object.second) {

          if (obj_prop.first == "__type__") {
            obj_type = obj_prop.second.get_value<string>();
            continue;
          }

          if (obj_prop.first == "file_path") {
            file_path = obj_prop.second.get_value<string>();
            continue;
          }

          if (obj_prop.first == "scaling") {
            for (auto &coord: obj_prop.second) {
              sca.push_back(
                  stof(coord.second.get_value<string>()));
            } continue;
          }

          if (obj_prop.first == "rotation") {
            for (auto &coord: obj_prop.second) {
              rot.push_back(
                  stof(coord.second.get_value<string>()));
            } continue;
          }

          if (obj_prop.first == "translation") {
            for (auto &coord: obj_prop.second) {
              trans.push_back(
                  stof(coord.second.get_value<string>()));
            } continue;
          }

          if (obj_prop.first == "radius") {
            radis = stof(obj_prop.second.get_value<string>());
            continue;
          }

          if (obj_prop.first == "position") {
            for (auto &coord: obj_prop.second) {
              pos.push_back(
                  stof(coord.second.get_value<string>()));
            } continue;
          }

          if (obj_prop.first == "materials") {
            for (auto &mat: obj_prop.second) {
              obj_mats.push_back(
                  allMaterials[mat.second.get_value<string>()]);
            } continue;
          }
        }
        if (obj_type == "sphere") {
          allObjects.push_back(new Sphere(radis, pos, obj_mats));
          continue;
        }
        else if (obj_type == "mesh") {
          Mesh *mesh =
              new Mesh(loadMesh(file_path), obj_mats);
          
          if (sca.size()) mesh->scale(sca);
          if (rot.size()) mesh->rotate(rot);
          if (trans.size()) mesh->translate(trans);

          allMeshes.push_back(mesh);
          continue;
        }
      }
    }

    if (world_prop.first == "lights"){
      for (auto &light: world_prop.second) {
        
        string light_type; float light_angle;
        vector<float> light_pos, light_col, light_dir; 

        for (auto &light_prop: light.second) {
          if (light_prop.first == "__type__") {
            light_type = light_prop.second.get_value<string>();
            continue;
          }
          if (light_prop.first == "position") {
            for (auto &coord: light_prop.second){
              light_pos.push_back(
                  stof(coord.second.get_value<string>())); }
            continue;
          }
          if (light_prop.first == "color") {
            for (auto &coord: light_prop.second){
              light_col.push_back(
                  stof(coord.second.get_value<string>())); }
            continue;
          }
          if (light_prop.first == "direction") {
            for (auto &coord: light_prop.second){
              light_dir.push_back(
                  stof(coord.second.get_value<string>())); }
            continue;
          }
          if (light_prop.first == "angle") {
            light_angle = stof(
                light_prop.second.get_value<string>());
            continue;
          }
        }
        if (light_type == "spot_light"){
          allLights.push_back(new SpotLight(light_col, light_pos,
                M_PI*light_angle/180, normfv(light_dir)));
          continue; 
        }
        if (light_type == "point_light"){
          allLights.push_back(
              new PunctualLight(light_col, light_pos));
          continue;
        }
        if (light_type == "directional_light"){
          allLights.push_back(new DirectionalLight(light_col,
                normfv(light_dir)));
          continue;
        }
        if (light_type == "ambient_light"){
          allLights.push_back(new AmbientLight(light_col));
          continue;
        }
      }
      continue;
    }

    if (world_prop.first == "params"){
      for (const auto &param_prop: world_prop.second){
        if (param_prop.first == "background_color") {
          int index = 0; 
          for (const auto &coord: param_prop.second)
            BACKGROUND_COLOR[index++] =
                stof(coord.second.get_value<string>());
          continue;
        }
        if (param_prop.first == "enable_shadows") {
          SHADOWS_ENABLED = param_prop.second.get_value<string>()
            == "true"?true:false;
          continue;
        }
        if (param_prop.first == "refraction_index") { 
          SCENE_REFR = 
            stof(param_prop.second.get_value<string>());
          continue;
        }
      }
      continue;
    }
  }
  
  return string(imgname);
}

