#pragma once

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <glm/glm.hpp>

#define USE_BVH 1
#define USE_MIS 1
#define MTBVH 1
#define OIDN_DENOISE 0
#define VIS_NORMAL 0
#define TONEMAPPING 1
#define STOCHASTIC_SAMPLING 0
#define DOF_ENABLED 0
#define SCATTER_ORIGIN_OFFSETMULT 0.001f
#define BOUNDING_BOX_EXPAND 0.001f
#define ALPHA_CUTOFF 0.01f
#define FIRST_INTERSECTION_CACHING 0
#define MAX_DEPTH 8
#define SORT_BY_MATERIAL_TYPE 0
#define MAX_NUM_PRIMS_IN_LEAF 2
#define SAH_BUCKET_SIZE 20
#define SAH_RAY_BOX_INTERSECTION_COST 0.1f


#define OCT_ENCODE_NORMAL 1
#define MAX_HIST_LEN 32.0f
#define ZDIFF_THRESHOLD 25.0f
#define NORMALDIFF_THRESHOLD 10.0f
#define LUMIN_BLEND_ALPHA 0.1f
#define MOMENTS_BLEND_ALPHA 0.1f
#define LUMIN_COEFF 3.0f
#define NORMAL_COEFF 10.0f
#define DEPTH_COEFF 5.0f

enum GeomType {
    SPHERE,
    CUBE,
    TRIANGLE_MESH
};

struct Ray {
    glm::vec3 origin;
    glm::vec3 direction;
};

struct ObjectTransform {
    glm::vec3 translation;
    glm::vec3 rotation;
    glm::vec3 scale;
    glm::mat4 transform;
    glm::mat4 inverseTransform;
    glm::mat4 invTranspose;
};

struct Object {
    enum GeomType type;
    int materialid;
    int triangleStart, triangleEnd;
    ObjectTransform Transform;
};

struct BoundingBox {
    glm::vec3 pMin, pMax;
    BoundingBox() :pMin(glm::vec3(1e38f)), pMax(glm::vec3(-1e38f)) {}
    glm::vec3 center() const { return (pMin + pMax) * 0.5f; }
};

BoundingBox Union(const BoundingBox& b1, const BoundingBox& b2);
BoundingBox Union(const BoundingBox& b1, const glm::vec3& p);
float BoxArea(const BoundingBox& b);

struct Primitive {
    int objID;
    int offset;//offset for triangles in model
    BoundingBox bbox;
    Primitive(const Object& obj, int objID, int triangleOffset = -1, const glm::ivec3* triangles = nullptr, const glm::vec3* vertices = nullptr);
};

struct BVHNode {
    int axis;
    BVHNode* left, * right;
    int startPrim, endPrim;
    BoundingBox bbox;
    BVHNode() :axis(-1), left(nullptr), right(nullptr), startPrim(-1), endPrim(-1) {}
};

struct BVHGPUNode
{
    int axis;
    BoundingBox bbox;
    int parent, left, right;
    int startPrim, endPrim;
    BVHGPUNode() :axis(-1), parent(-1), left(-1), right(-1), startPrim(-1), endPrim(-1){}
};


struct MTBVHGPUNode
{
    BoundingBox bbox;
    int hitLink, missLink;
    int startPrim, endPrim;
    MTBVHGPUNode():hitLink(-1), missLink(-1), startPrim(-1), endPrim(-1){}
};

const int dirs[] = {
    1,-1,2,-2,3,-3
};



enum MaterialType {
    diffuse, frenselSpecular, microfacet, metallicWorkflow, emitting
};

enum TextureType {
    color, normal, metallicroughness
};

struct GLTFTextureLoadInfo {
    char* buffer;
    int matIndex;
    TextureType texType;
    int width, height;
    int bits, component;
    GLTFTextureLoadInfo(char* buffer, int index, TextureType type, int width, int height, int bits, int component) :buffer(buffer), matIndex(index), texType(type), width(width), height(height), bits(bits), component(component){}
};

struct Material {
    glm::vec3 color = glm::vec3(0);
    float indexOfRefraction = 0;
    float emittance = 0;
    float metallic = -1.0;
    float roughness = -1.0;
    cudaTextureObject_t baseColorMap = 0, normalMap = 0, metallicRoughnessMap = 0;
    MaterialType type = diffuse;
};

struct Camera {
    glm::ivec2 resolution;
    glm::vec3 position;
    glm::vec3 lookAt;
    glm::vec3 view;
    glm::vec3 up;
    glm::vec3 right;
    glm::vec2 fov;
    glm::vec2 pixelLength;
    float fovAngle;
    float lensRadius = 0.001f;
    float focalLength = 1.0f;
};

struct LastCameraInfo {
    glm::vec3 position;
    glm::vec3 view;
    glm::vec3 up;
    glm::vec3 right;
};

struct RenderState {
    Camera camera;
    LastCameraInfo lastCamInfo;
    glm::mat4 prevViewProj;
    unsigned int iterations;
    int traceDepth;
    std::vector<glm::vec3> image;
    std::string imageName;
};

struct PathSegment {
    Ray ray;
    glm::vec3 color;
    int pixelIndex;
    int remainingBounces;
    float lastMatPdf;
};

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct ShadeableIntersection {
    float t = -1.0;
    glm::vec3 surfaceNormal = glm::vec3(0.0);
    glm::vec3 surfaceTangent = glm::vec3(0.0);
    float fsign = 1.0;
    glm::vec3 worldPos = glm::vec3(0.0);
    int materialId = -1;
    int primitiveId = -1;
    glm::vec2 uv = glm::vec2(0.0);
    MaterialType type = diffuse;
};

struct ModelInfoDev {
    glm::ivec3* dev_triangles;
    glm::vec3* dev_vertices;
    glm::vec2* dev_uvs;
    glm::vec3* dev_normals;
    glm::vec3* dev_tangents;
    float* dev_fsigns;
};

struct SceneInfoPtrs {
    Material* dev_materials;
    Object* dev_objs;
    int objectsSize;
    ModelInfoDev modelInfo;
    Primitive* dev_primitives;
    union {
        BVHGPUNode* dev_bvhArray;
        MTBVHGPUNode* dev_mtbvhArray;
    };
    int bvhDataSize;
    Primitive* dev_lights;
    int lightsSize;
    cudaTextureObject_t skyboxObj;
};

struct EAWParams {
    float sigmaCol;
    float sigmaAlbe;
    float sigmaPos;
    float sigmaNorm;
};

struct SVGFParams {

};

struct SceneGbufferPtrs {
    float* dev_t;
    float* dev_z;
    glm::vec3* dev_emission;
    glm::vec3* dev_albedo;
#if OCT_ENCODE_NORMAL
    glm::vec2* dev_normal;
#else
    glm::vec3* dev_normal;
#endif
//    glm::vec3* dev_position;
    glm::vec2* dev_velocity;
    glm::vec2* dev_znormalfwidth;
    int* dev_primID;
};

enum VisualizationType {
    render, gbuffer
};

enum GBufferVisualizationType {
    gTime, gPosition, gNormal, gVelocity, gAlbedo, gEmission, gDepth
};

enum DenoiserType {
    EAW, SVGF
};


struct SVGFBufferPtrs {
    glm::vec4* dev_illum;
    glm::vec2* dev_moments;
    float* dev_history_len;
    float* dev_z;
#if OCT_ENCODE_NORMAL
    glm::vec2* dev_normal;
#else
    glm::vec3* dev_normal;
#endif
    int* dev_primID;
};


