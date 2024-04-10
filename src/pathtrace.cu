#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"
#include "denoise.h"
#define ERRORCHECK 1


#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
void checkCUDAErrorFn(const char* msg, const char* file, int line) {
#if ERRORCHECK
	cudaDeviceSynchronize();
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess == err) {
		return;
	}

	fprintf(stderr, "CUDA error");
	if (file) {
		fprintf(stderr, " (%s:%d)", file, line);
	}
	fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
#  ifdef _WIN32
	getchar();
#  endif
	exit(EXIT_FAILURE);
#endif
}

__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth) {
	int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
	return thrust::default_random_engine(h);
}

__device__ inline bool util_math_is_nan(const glm::vec3& v)
{
	return (v.x != v.x) || (v.y != v.y) || (v.z != v.z);
}



//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution,
	int iter, glm::vec3* image) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < resolution.x && y < resolution.y) {
		int index = x + (y * resolution.x);
		glm::vec3 pix = image[index];

		glm::vec3 color;
#if TONEMAPPING
		color = pix / (float)iter;
		color = util_postprocess_gamma(util_postprocess_ACESFilm(color));
		color = color * 255.0f;
#else
		color.x = glm::clamp((int)(pix.x / iter * 255.0), 0, 255);
		color.y = glm::clamp((int)(pix.y / iter * 255.0), 0, 255);
		color.z = glm::clamp((int)(pix.z / iter * 255.0), 0, 255);
#endif
		if (util_math_is_nan(pix))
		{
			pbo[index].x = 255;
			pbo[index].y = 192;
			pbo[index].z = 203;
		}
		else
		{
			// Each thread writes one pixel location in the texture (textel)
			pbo[index].x = color.x;
			pbo[index].y = color.y;
			pbo[index].z = color.z;
		}
		pbo[index].w = 0;
	}
}

__global__ void vec4ToPBO(uchar4* pbo, glm::ivec2 resolution,
	int iter, glm::vec4* image) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < resolution.x && y < resolution.y) {
		int index = x + (y * resolution.x);
		glm::vec3 pix = glm::vec3(image[index]);

		glm::vec3 color;
#if TONEMAPPING
		color = pix / (float)iter;
		color = util_postprocess_gamma(util_postprocess_ACESFilm(color));
		color = color * 255.0f;
#else
		color.x = glm::clamp((int)(pix.x / iter * 255.0), 0, 255);
		color.y = glm::clamp((int)(pix.y / iter * 255.0), 0, 255);
		color.z = glm::clamp((int)(pix.z / iter * 255.0), 0, 255);
#endif
		if (util_math_is_nan(pix))
		{
			pbo[index].x = 255;
			pbo[index].y = 192;
			pbo[index].z = 203;
		}
		else
		{
			// Each thread writes one pixel location in the texture (textel)
			pbo[index].x = color.x;
			pbo[index].y = color.y;
			pbo[index].z = color.z;
		}
		pbo[index].w = 0;
	}
}


__global__ void gbufferToPBO(uchar4* pbo, glm::ivec2 resolution, SceneGbufferPtrs gBuffer, GBufferVisualizationType type, RenderState state) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < resolution.x && y < resolution.y) {
		int index = x + (y * resolution.x);
		float timeToIntersect = gBuffer.dev_t[index] * 256.0;
		int primID = gBuffer.dev_primID[index];
		if (primID == -1)
		{
			pbo[index].w = 0;
			pbo[index].x = 0;
			pbo[index].y = 0;
			pbo[index].z = 0;
			return;
		}
		glm::vec3 pos = glm::clamp(util_get_world_position(glm::vec2(x,y), gBuffer.dev_z[index], state.camera.position, state.camera.view, state.camera.up, state.camera.right, state.camera.resolution, state.camera.pixelLength) * 256.0f, 0.0f, 255.0f);
		glm::vec3 normal = glm::clamp((util_oct_to_vec3(gBuffer.dev_normal[index]) * 0.5f + 0.5f) * 256.0f, 0.0f, 255.0f);
		glm::vec3 velocity = glm::clamp((glm::vec3(glm::abs(gBuffer.dev_velocity[index]), 0.0f)) * 256.0f, 0.0f, 255.0f);
		glm::vec3 albedo = glm::clamp(gBuffer.dev_albedo[index] * 256.0f, 0.0f, 255.0f);
		glm::vec3 emission = glm::clamp(gBuffer.dev_emission[index] * 256.0f, 0.0f, 255.0f);
		glm::vec2 velocity_pixel = glm::vec2(gBuffer.dev_velocity[index]) * (glm::vec2)resolution;
		float depth = glm::clamp(abs(gBuffer.dev_z[index]) * 256.0f, 0.0f, 255.0f);
		if (type == gTime)
		{
			pbo[index].w = 0;
			pbo[index].x = timeToIntersect;
			pbo[index].y = timeToIntersect;
			pbo[index].z = timeToIntersect;
		}
		else if (type == gPosition)
		{
			pbo[index].w = 0;
			pbo[index].x = pos.x;
			pbo[index].y = pos.y;
			pbo[index].z = pos.z;
		}
		else if (type == gNormal)
		{
			pbo[index].w = 0;
			pbo[index].x = normal.x;
			pbo[index].y = normal.y;
			pbo[index].z = normal.z;
		}
		else if (type == gVelocity)
		{
			pbo[index].w = 0;
			pbo[index].x = velocity.x;
			pbo[index].y = velocity.y;
			pbo[index].z = velocity.z;
		}
		else if (type == gAlbedo)
		{
			pbo[index].w = 0;
			pbo[index].x = albedo.x;
			pbo[index].y = albedo.y;
			pbo[index].z = albedo.z;
		}
		else if(type == gEmission)
		{
			pbo[index].w = 0;
			pbo[index].x = emission.x;
			pbo[index].y = emission.y;
			pbo[index].z = emission.z;
		}
		else if (type == gDepth)
		{
			pbo[index].w = 0;
			pbo[index].x = depth;
			pbo[index].y = depth;
			pbo[index].z = depth;
		}
	}
}

static Scene* hst_scene = NULL;
static GuiDataContainer* guiData = NULL;
static glm::vec3* dev_image0 = NULL;
static glm::vec3* dev_image1 = NULL;
static Object* dev_objs = NULL;
static Material* dev_materials = NULL;
static BVHGPUNode* dev_bvhArray = NULL;
static MTBVHGPUNode* dev_mtbvhArray = NULL;
static Primitive* dev_primitives = NULL;
static glm::ivec3* dev_triangles = NULL;
static glm::vec3* dev_vertices = NULL;
static glm::vec2* dev_uvs = NULL;
static glm::vec3* dev_normals = NULL;
static glm::vec3* dev_tangents = NULL;
static float* dev_fsigns = NULL;
static Primitive* dev_lights = NULL;
static PathSegment* dev_paths1 = NULL;
static PathSegment* dev_paths2 = NULL;
static int* dev_rayValidCache = NULL;
static glm::vec3* dev_imageCache = NULL;
static ShadeableIntersection* dev_pathCache = NULL;
static ShadeableIntersection* dev_intersections1 = NULL;
static ShadeableIntersection* dev_intersections2 = NULL;
static ShadeableIntersection* dev_intersectionCache = NULL;
static int* dev_rayValid;
static int* dev_rayIndex;

static float* dev_g_t = NULL;
static glm::vec3* dev_g_emission = NULL;
static glm::vec3* dev_g_albedo = NULL;
#if OCT_ENCODE_NORMAL
static glm::vec2* dev_g_normal = NULL;
static glm::vec2* dev_prev_normal = NULL;
#else
static glm::vec3* dev_g_normal = NULL;
static glm::vec3* dev_prev_normal = NULL;
#endif
static glm::vec3* dev_g_position = NULL;
static glm::vec2* dev_g_velocity = NULL;
static float* dev_g_z = NULL;
static float* dev_prev_z = NULL;
static glm::vec2* dev_g_znormalfwidth = NULL;
static int* dev_g_primID = NULL;
static int* dev_prev_primID = NULL;

static glm::vec4* dev_illum0 = NULL;
static glm::vec2* dev_moments0 = NULL;
static float* dev_histLen0 = NULL;
static glm::vec4* dev_illum1 = NULL;
static glm::vec2* dev_moments1 = NULL;
static float* dev_histLen1 = NULL;
// TODO: static variables for device memory, any extra info you need, etc
// ...

void InitDataContainer(GuiDataContainer* imGuiData)
{
	guiData = imGuiData;
}

void pathtraceInit(Scene* scene) {
	hst_scene = scene;

	const Camera& cam = hst_scene->state.camera;
	const int pixelcount = cam.resolution.x * cam.resolution.y;

	cudaMalloc(&dev_image0, pixelcount * sizeof(glm::vec3));
	cudaMemset(dev_image0, 0, pixelcount * sizeof(glm::vec3));

	cudaMalloc(&dev_image1, pixelcount * sizeof(glm::vec3));
	cudaMemset(dev_image1, 0, pixelcount * sizeof(glm::vec3));

	cudaMalloc(&dev_g_t, pixelcount * sizeof(float));
	cudaMemset(dev_g_t, 0, pixelcount * sizeof(float));

	cudaMalloc(&dev_g_z, pixelcount * sizeof(float));
	cudaMemset(dev_g_z, 0, pixelcount * sizeof(float));

	cudaMalloc(&dev_prev_z, pixelcount * sizeof(float));
	cudaMemset(dev_prev_z, 0, pixelcount * sizeof(float));

	cudaMalloc(&dev_g_emission, pixelcount * sizeof(glm::vec3));
	cudaMemset(dev_g_emission, 0, pixelcount * sizeof(glm::vec3));

	cudaMalloc(&dev_g_albedo, pixelcount * sizeof(glm::vec3));
	cudaMemset(dev_g_albedo, 0, pixelcount * sizeof(glm::vec3));
#if OCT_ENCODE_NORMAL
	cudaMalloc(&dev_g_normal, pixelcount * sizeof(glm::vec2));
	cudaMemset(dev_g_normal, 0, pixelcount * sizeof(glm::vec2));
#else
	cudaMalloc(&dev_g_normal, pixelcount * sizeof(glm::vec3));
	cudaMemset(dev_g_normal, 0, pixelcount * sizeof(glm::vec3));
#endif

	cudaMalloc(&dev_prev_normal, pixelcount * sizeof(glm::vec3));
	cudaMemset(dev_prev_normal, 0, pixelcount * sizeof(glm::vec3));

	cudaMalloc(&dev_g_position, pixelcount * sizeof(glm::vec3));
	cudaMemset(dev_g_position, 0, pixelcount * sizeof(glm::vec3));

	cudaMalloc(&dev_g_velocity, pixelcount * sizeof(glm::vec2));
	cudaMemset(dev_g_velocity, 0, pixelcount * sizeof(glm::vec2));

	cudaMalloc(&dev_g_znormalfwidth, pixelcount * sizeof(glm::vec2));
	cudaMemset(dev_g_znormalfwidth, 0, pixelcount * sizeof(glm::vec2));

	cudaMalloc(&dev_g_znormalfwidth, pixelcount * sizeof(glm::vec2));
	cudaMemset(dev_g_znormalfwidth, 0, pixelcount * sizeof(glm::vec2));

	cudaMalloc(&dev_g_primID, pixelcount * sizeof(int));
	cudaMemset(dev_g_primID, -1, pixelcount * sizeof(int));

	cudaMalloc(&dev_prev_primID, pixelcount * sizeof(int));
	cudaMemset(dev_prev_primID, -1, pixelcount * sizeof(int));

	cudaMalloc(&dev_illum0, pixelcount * sizeof(glm::vec4));
	cudaMemset(dev_illum0, 0, pixelcount * sizeof(glm::vec4));

	cudaMalloc(&dev_moments0, pixelcount * sizeof(glm::vec2));
	cudaMemset(dev_moments0, 0, pixelcount * sizeof(glm::vec2));

	cudaMalloc(&dev_histLen0, pixelcount * sizeof(float));
	cudaMemset(dev_histLen0, 0, pixelcount * sizeof(float));

	cudaMalloc(&dev_illum1, pixelcount * sizeof(glm::vec4));
	cudaMemset(dev_illum1, 0, pixelcount * sizeof(glm::vec4));

	cudaMalloc(&dev_moments1, pixelcount * sizeof(glm::vec2));
	cudaMemset(dev_moments1, 0, pixelcount * sizeof(glm::vec2));

	cudaMalloc(&dev_histLen1, pixelcount * sizeof(float));
	cudaMemset(dev_histLen1, 0, pixelcount * sizeof(float));

	cudaMalloc(&dev_rayValid, sizeof(int) * pixelcount);
	cudaMalloc(&dev_rayIndex, sizeof(int) * pixelcount);

	cudaMalloc(&dev_paths1, pixelcount * sizeof(PathSegment));
	cudaMalloc(&dev_paths2, pixelcount * sizeof(PathSegment));

	cudaMalloc(&dev_objs, scene->objects.size() * sizeof(Object));
	cudaMemcpy(dev_objs, scene->objects.data(), scene->objects.size() * sizeof(Object), cudaMemcpyHostToDevice);

	if (scene->triangles.size())
	{
		cudaMalloc(&dev_triangles, scene->triangles.size() * sizeof(glm::ivec3));
		cudaMemcpy(dev_triangles, scene->triangles.data(), scene->triangles.size() * sizeof(glm::ivec3), cudaMemcpyHostToDevice);

		cudaMalloc(&dev_vertices, scene->verticies.size() * sizeof(glm::vec3));
		cudaMemcpy(dev_vertices, scene->verticies.data(), scene->verticies.size() * sizeof(glm::vec3), cudaMemcpyHostToDevice);

		cudaMalloc(&dev_uvs, scene->uvs.size() * sizeof(glm::vec2));
		cudaMemcpy(dev_uvs, scene->uvs.data(), scene->uvs.size() * sizeof(glm::vec2), cudaMemcpyHostToDevice);
		if (scene->normals.size())
		{
			cudaMalloc(&dev_normals, scene->normals.size() * sizeof(glm::vec3));
			cudaMemcpy(dev_normals, scene->normals.data(), scene->normals.size() * sizeof(glm::vec3), cudaMemcpyHostToDevice);
		}
		if (scene->tangents.size())
		{
			cudaMalloc(&dev_tangents, scene->tangents.size() * sizeof(glm::vec3));
			cudaMemcpy(dev_tangents, scene->tangents.data(), scene->tangents.size() * sizeof(glm::vec3), cudaMemcpyHostToDevice);
		}
		if (scene->fSigns.size())
		{
			cudaMalloc(&dev_fsigns, scene->fSigns.size() * sizeof(float));
			cudaMemcpy(dev_fsigns, scene->fSigns.data(), scene->fSigns.size() * sizeof(float), cudaMemcpyHostToDevice);
		}
	}
	
#if MTBVH
	cudaMalloc(&dev_mtbvhArray, scene->MTBVHArray.size() * sizeof(MTBVHGPUNode));
	cudaMemcpy(dev_mtbvhArray, scene->MTBVHArray.data(), scene->MTBVHArray.size() * sizeof(MTBVHGPUNode), cudaMemcpyHostToDevice);
#else
	cudaMalloc(&dev_bvhArray, scene->bvhArray.size() * sizeof(BVHGPUNode));
	cudaMemcpy(dev_bvhArray, scene->bvhArray.data(), scene->bvhArray.size() * sizeof(BVHGPUNode), cudaMemcpyHostToDevice);
#endif

	cudaMalloc(&dev_primitives, scene->primitives.size() * sizeof(Primitive));
	cudaMemcpy(dev_primitives, scene->primitives.data(), scene->primitives.size() * sizeof(Primitive), cudaMemcpyHostToDevice);

	if (scene->lights.size())
	{
		cudaMalloc(&dev_lights, scene->lights.size() * sizeof(Primitive));
		cudaMemcpy(dev_lights, scene->lights.data(), scene->lights.size() * sizeof(Primitive), cudaMemcpyHostToDevice);
	}

	cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
	cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_intersections1, pixelcount * sizeof(ShadeableIntersection));
	cudaMemset(dev_intersections1, 0, pixelcount * sizeof(ShadeableIntersection));

	cudaMalloc(&dev_intersections2, pixelcount * sizeof(ShadeableIntersection));

#if !STOCHASTIC_SAMPLING && FIRST_INTERSECTION_CACHING
	cudaMalloc(&dev_intersectionCache, pixelcount * sizeof(ShadeableIntersection));
	cudaMalloc(&dev_pathCache, pixelcount * sizeof(PathSegment));
	cudaMalloc(&dev_rayValidCache, pixelcount * sizeof(int));
	cudaMalloc(&dev_imageCache, pixelcount * sizeof(glm::vec3));
#endif
	// TODO: initialize any extra device memeory you need

	checkCUDAError("pathtraceInit");
}

void pathtraceClear()
{
	const Camera& cam = hst_scene->state.camera;
	const int pixelcount = cam.resolution.x * cam.resolution.y;
	cudaMemset(dev_image0, 0, pixelcount * sizeof(glm::vec3));
	cudaMemset(dev_g_t, 0, pixelcount * sizeof(float));
	cudaMemset(dev_g_z, 0, pixelcount * sizeof(float));
	cudaMemset(dev_g_emission, 0, pixelcount * sizeof(glm::vec3));
	cudaMemset(dev_g_albedo, 0, pixelcount * sizeof(glm::vec3));
#if OCT_ENCODE_NORMAL
	cudaMemset(dev_g_normal, 0, pixelcount * sizeof(glm::vec2));
#else
	cudaMemset(dev_g_normal, 0, pixelcount * sizeof(glm::vec3));
#endif
	cudaMemset(dev_g_position, 0, pixelcount * sizeof(glm::vec3));
	cudaMemset(dev_g_velocity, 0, pixelcount * sizeof(glm::vec2));
	cudaMemset(dev_g_znormalfwidth, 0, pixelcount * sizeof(glm::vec2));
	cudaMemset(dev_g_primID, -1, pixelcount * sizeof(int));
	cudaMemset(dev_prev_primID, -1, pixelcount * sizeof(int));
	cudaMemset(dev_intersections1, 0, pixelcount * sizeof(ShadeableIntersection));
	checkCUDAError("pathtraceClear");
}

void denoiseClear()
{
	const Camera& cam = hst_scene->state.camera;
	const int pixelcount = cam.resolution.x * cam.resolution.y;
	cudaMemset(dev_illum0, 0, pixelcount * sizeof(glm::vec4));
	cudaMemset(dev_moments0, 0, pixelcount * sizeof(glm::vec2));
	cudaMemset(dev_histLen0, 0, pixelcount * sizeof(float));
	cudaMemset(dev_illum1, 0, pixelcount * sizeof(glm::vec4));
	cudaMemset(dev_moments1, 0, pixelcount * sizeof(glm::vec2));
	cudaMemset(dev_histLen1, 0, pixelcount * sizeof(float));
#if OCT_ENCODE_NORMAL
	cudaMemset(dev_prev_normal, 0, pixelcount * sizeof(glm::vec2));
#else
	cudaMemset(dev_prev_normal, 0, pixelcount * sizeof(glm::vec3));
#endif
	cudaMemset(dev_prev_z, 0, pixelcount * sizeof(float));
	cudaMemset(dev_prev_primID, 0, pixelcount * sizeof(int));
	checkCUDAError("denoiseClear");
}

void pathtraceFree(Scene* scene) {
	cudaFree(dev_image0);  // no-op if dev_image is null

	cudaFree(dev_g_t);
	cudaFree(dev_g_z);
	cudaFree(dev_g_emission);
	cudaFree(dev_g_albedo);
	cudaFree(dev_g_normal);
	cudaFree(dev_g_position);
	cudaFree(dev_g_velocity);
	cudaFree(dev_g_znormalfwidth);

	cudaFree(dev_illum0);
	cudaFree(dev_illum1);
	cudaFree(dev_moments0);
	cudaFree(dev_moments1);
	cudaFree(dev_histLen0);
	cudaFree(dev_histLen1);
	cudaFree(dev_prev_normal);
	cudaFree(dev_prev_z);

	cudaFree(dev_rayIndex);
	cudaFree(dev_rayValid);
	cudaFree(dev_paths1);
	cudaFree(dev_paths2);
	cudaFree(dev_objs);
	if (scene->triangles.size())
	{
		cudaFree(dev_triangles);
		cudaFree(dev_vertices);
		cudaFree(dev_uvs);
		if (scene->normals.size())
		{
			cudaFree(dev_normals);
		}
		if (scene->tangents.size())
		{
			cudaFree(dev_tangents);
		}
		if (scene->fSigns.size())
		{
			cudaFree(dev_fsigns);
		}
	}
	cudaFree(dev_primitives);
	if (scene->lights.size())
	{
		cudaMalloc(&dev_lights, scene->lights.size() * sizeof(Primitive));
		cudaMemcpy(dev_lights, scene->lights.data(), scene->lights.size() * sizeof(Primitive), cudaMemcpyHostToDevice);
	}
#if MTBVH
	cudaFree(dev_mtbvhArray);
#else
	cudaFree(dev_bvhArray);
#endif
	cudaFree(dev_materials);
	cudaFree(dev_intersections1);
	cudaFree(dev_intersections2);
#if !STOCHASTIC_SAMPLING && FIRST_INTERSECTION_CACHING
	cudaFree(dev_intersectionCache);
	cudaFree(dev_pathCache);
	cudaFree(dev_rayValidCache);
	cudaFree(dev_imageCache);
#endif
	// TODO: clean up any extra device memory you created

	checkCUDAError("pathtraceFree");
}

__device__ inline glm::vec2 util_concentric_sample_disk(glm::vec2 rand)
{
	rand = 2.0f * rand - 1.0f;
	if (rand.x == 0 && rand.y == 0)
	{
		return glm::vec2(0);
	}
	const float pi_4 = PI / 4, pi_2 = PI / 2;
	bool x_g_y = abs(rand.x) > abs(rand.y);
	float theta = x_g_y ? pi_4 * rand.y / rand.x : pi_2 - pi_4 * rand.x / rand.y;
	float r = x_g_y ? rand.x : rand.y;
	return glm::vec2(cos(theta), sin(theta)) * r;
}

/**
* Generate PathSegments with rays from the camera through the screen into the
* scene, which is the first bounce of rays.
*
* Antialiasing - add rays for sub-pixel sampling
* motion blur - jitter rays "in time"
* lens effect - jitter ray origin positions based on a lens
*/
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	thrust::default_random_engine rng = makeSeededRandomEngine(x, y, iter);
	thrust::uniform_real_distribution<float> u01(0, 1);

	if (x < cam.resolution.x && y < cam.resolution.y) {
		int index = x + (y * cam.resolution.x);
		PathSegment& segment = pathSegments[index];

		segment.ray.origin = cam.position;
		segment.color = glm::vec3(1.0f, 1.0f, 1.0f);
#if STOCHASTIC_SAMPLING
		// TODO: implement antialiasing by jittering the ray
		glm::vec2 jitter = glm::vec2(0.5f * (u01(rng) * 2.0f - 1.0f), 0.5f * (u01(rng) * 2.0f - 1.0f));
		segment.ray.direction = glm::normalize(cam.view
			- cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f + jitter[0])
			- cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f + jitter[1])
		);
#if DOF_ENABLED
		float lensR = cam.lensRadius;
		glm::vec3 perpDir = glm::cross(cam.right, cam.up);
		perpDir = glm::normalize(perpDir);
		float focalLen = cam.focalLength;
		float tFocus = focalLen / glm::abs(glm::dot(segment.ray.direction, perpDir));
		glm::vec2 offset = lensR * util_concentric_sample_disk(glm::vec2(u01(rng), u01(rng)));
		glm::vec3 newOri = offset.x * cam.right + offset.y * cam.up + cam.position;
		glm::vec3 pFocus = segment.ray.direction * tFocus + segment.ray.origin;
		segment.ray.direction = glm::normalize(pFocus - newOri);
		segment.ray.origin = newOri;
#endif

#else
		segment.ray.direction = glm::normalize(cam.view
			- cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f)
			- cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f)
		);
#endif
		segment.pixelIndex = index;
		segment.remainingBounces = traceDepth;
		segment.lastMatPdf = -1;
	}
}

// TODO:
// computeIntersections handles generating ray intersections ONLY.
// Generating new rays is handled in your shader(s).
// Feel free to modify the code below.
__global__ void compute_intersection(
	int depth
	, int num_paths
	, PathSegment* pathSegments
	, Object* geoms
	, int objs_size
	, glm::ivec3* modelTriangles
	, glm::vec3* modelVertices
	, const glm::vec2* modelUVs
	, const glm::vec3* modelNormals
	, cudaTextureObject_t skyboxTex
	, ShadeableIntersection* intersections
	, int* rayValid
	, glm::vec3* image
)
{
	int path_index = blockIdx.x * blockDim.x + threadIdx.x;

	if (path_index < num_paths)
	{
		PathSegment& pathSegment = pathSegments[path_index];
		float t = -1.0;
		glm::vec3 intersect_point;
		glm::vec3 normal;
		float t_min = FLT_MAX;
		int hit_material_index = -1;
		bool outside = true;

		glm::vec3 tmp_intersect;
		glm::vec3 tmp_normal;

		// naive parse through global geoms

		for (int i = 0; i < objs_size; i++)
		{
			Object& geom = geoms[i];

			if (geom.type == CUBE)
			{
				t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal);
			}
			else if (geom.type == SPHERE)
			{
				t = util_geometry_ray_sphere_intersection(geom, pathSegment.ray, tmp_intersect, tmp_normal);
			}
			else if (geom.type == TRIANGLE_MESH)
			{
				glm::vec3 baryCoord;
				for (int i = geom.triangleStart; i != geom.triangleEnd; i++)
				{
					const glm::ivec3& tri = modelTriangles[i];
					const glm::vec3& v0 = modelVertices[tri[0]];
					const glm::vec3& v1 = modelVertices[tri[1]];
					const glm::vec3& v2 = modelVertices[tri[2]];
					t = triangleIntersectionTest(geom.Transform, v0, v1, v2, pathSegment.ray, tmp_intersect, tmp_normal, baryCoord);
					if (t > 0.0f && t_min > t)
					{
						t_min = t;
						hit_material_index = geom.materialid;
						intersect_point = tmp_intersect;
						normal = tmp_normal;
					}
				}
			}
			// TODO: add more intersection tests here... triangle? metaball? CSG?

			// Compute the minimum t from the intersection tests to determine what
			// scene geometry object was hit first.
			if (t > 0.0f && t_min > t)
			{
				t_min = t;
				hit_material_index = geom.materialid;
				intersect_point = tmp_intersect;
				normal = tmp_normal;
			}
		}

		if (t_min == FLT_MAX)//hits nothing
		{
			rayValid[path_index] = 0;
			intersections[path_index].t = -1.0f;
			if (skyboxTex)
			{
				glm::vec2 uv = util_sample_spherical_map(glm::normalize(pathSegment.ray.direction));
				float4 skyColorRGBA = tex2D<float4>(skyboxTex, uv.x, uv.y);
				glm::vec3 skyColor = glm::vec3(skyColorRGBA.x, skyColorRGBA.y, skyColorRGBA.z);
				image[pathSegment.pixelIndex] += pathSegment.color * skyColor;
			}
		}
		else
		{
			//The ray hits something
			intersections[path_index].t = t_min;
			intersections[path_index].materialId = hit_material_index;
			intersections[path_index].surfaceNormal = normal;
			intersections[path_index].worldPos = intersect_point;
			rayValid[path_index] = 1;
		}
	}
}

__device__ inline int util_bvh_get_sibling(const BVHGPUNode* bvhArray, int curr)
{
	int parent = bvhArray[curr].parent;
	if (parent == -1) return -1;
	return bvhArray[parent].left == curr ? bvhArray[parent].right : bvhArray[parent].left;
}

__device__ inline int util_bvh_get_near_child(const BVHGPUNode* bvhArray, int curr, const glm::vec3& rayDir)
{
	return rayDir[bvhArray[curr].axis] > 0.0 ? bvhArray[curr].left : bvhArray[curr].right;
}

__device__ inline bool util_bvh_is_leaf(const BVHGPUNode* bvhArray, int curr)
{
	return bvhArray[curr].left == -1 && bvhArray[curr].right == -1;
}



enum bvh_traverse_state {
	fromChild,fromParent,fromSibling
};

__global__ void compute_intersection_bvh_stackless(
	int depth
	, int num_paths
	, PathSegment* pathSegments
	, SceneInfoPtrs dev_sceneInfo
	, ShadeableIntersection* intersections
	, int* rayValid
	, glm::vec3* image
)
{
	int path_index = blockIdx.x * blockDim.x + threadIdx.x;
	if (path_index >= num_paths) return;
	PathSegment& pathSegment = pathSegments[path_index];
	Ray& ray = pathSegment.ray;
	glm::vec3 rayDir = pathSegment.ray.direction;
	glm::vec3 rayOri = pathSegment.ray.origin;
	int curr = util_bvh_get_near_child(dev_sceneInfo.dev_bvhArray, 0, rayOri);
	bvh_traverse_state state = fromParent;
	ShadeableIntersection tmpIntersection;
	tmpIntersection.t = 1e37f;
	bool intersected = false;
	while (curr >= 0 && curr < dev_sceneInfo.bvhDataSize)
	{
		if (state == fromChild)
		{
			if (curr == 0) break;
			int parent = dev_sceneInfo.dev_bvhArray[curr].parent;
			if (curr == util_bvh_get_near_child(dev_sceneInfo.dev_bvhArray, parent, rayOri))
			{
				curr = util_bvh_get_sibling(dev_sceneInfo.dev_bvhArray, curr);
				state = fromSibling;
			}
			else
			{
				curr = parent;
				state = fromChild;
			}
		}
		else if (state == fromSibling)
		{
			bool outside = true;
			float boxt = boundingBoxIntersectionTest(dev_sceneInfo.dev_bvhArray[curr].bbox, ray, outside);
			if (!outside) boxt = EPSILON;
			if (!(boxt > 0 && boxt < tmpIntersection.t))
			{
				curr = dev_sceneInfo.dev_bvhArray[curr].parent;
				state = fromChild;
			}
			else if (util_bvh_is_leaf(dev_sceneInfo.dev_bvhArray, curr))
			{
				int start = dev_sceneInfo.dev_bvhArray[curr].startPrim, end = dev_sceneInfo.dev_bvhArray[curr].endPrim;
				if (util_bvh_leaf_intersect(start, end, dev_sceneInfo, ray, &tmpIntersection))
				{
					intersected = true;
				}
				curr = dev_sceneInfo.dev_bvhArray[curr].parent;
				state = fromChild;
			}
			else
			{
				curr = util_bvh_get_near_child(dev_sceneInfo.dev_bvhArray, curr, rayOri);
				state = fromParent;
			}
		}
		else// from parent
		{
			bool outside = true;
			float boxt = boundingBoxIntersectionTest(dev_sceneInfo.dev_bvhArray[curr].bbox, ray, outside);
			if (!outside) boxt = EPSILON;
			if (!(boxt > 0 && boxt < tmpIntersection.t))
			{
				curr = util_bvh_get_sibling(dev_sceneInfo.dev_bvhArray, curr);
				state = fromSibling;
			}
			else if (util_bvh_is_leaf(dev_sceneInfo.dev_bvhArray, curr))
			{
				int start = dev_sceneInfo.dev_bvhArray[curr].startPrim, end = dev_sceneInfo.dev_bvhArray[curr].endPrim;
				if (util_bvh_leaf_intersect(start, end, dev_sceneInfo, ray, &tmpIntersection))
				{
					intersected = true;
				}
				curr = util_bvh_get_sibling(dev_sceneInfo.dev_bvhArray, curr);
				state = fromSibling;
			}
			else
			{
				curr = util_bvh_get_near_child(dev_sceneInfo.dev_bvhArray, curr, pathSegment.ray.origin);
				state = fromParent;
			}
		}
	}
	rayValid[path_index] = intersected;
	intersections[path_index].t = -1.0f;
	if (intersected)
	{
		intersections[path_index] = tmpIntersection;
		intersections[path_index].type = dev_sceneInfo.dev_materials[tmpIntersection.materialId].type;
	}
	else if(dev_sceneInfo.skyboxObj)
	{
		glm::vec2 uv = util_sample_spherical_map(glm::normalize(rayDir));
		float4 skyColorRGBA = tex2D<float4>(dev_sceneInfo.skyboxObj, uv.x, uv.y);
		glm::vec3 skyColor = glm::vec3(skyColorRGBA.x, skyColorRGBA.y, skyColorRGBA.z);
		image[pathSegment.pixelIndex] += pathSegment.color * skyColor;
	}
}



__global__ void compute_intersection_bvh_stackless_mtbvh(
	int depth
	, int num_paths
	, PathSegment* pathSegments
	, SceneInfoPtrs dev_sceneInfo
	, ShadeableIntersection* intersections
	, int* rayValid
	, glm::vec3* image
)
{
	int path_index = blockIdx.x * blockDim.x + threadIdx.x;
	if (path_index >= num_paths) return;
	PathSegment& pathSegment = pathSegments[path_index];
	Ray& ray = pathSegment.ray;
	glm::vec3 rayDir = pathSegment.ray.direction;
	glm::vec3 rayOri = pathSegment.ray.origin;
	float x = fabs(rayDir.x), y = fabs(rayDir.y), z = fabs(rayDir.z);
	int axis = x > y && x > z ? 0 : (y > z ? 1 : 2);
	int sgn = rayDir[axis] > 0 ? 0 : 1;
	int d = (axis << 1) + sgn;
	const MTBVHGPUNode* currArray = dev_sceneInfo.dev_mtbvhArray + d * dev_sceneInfo.bvhDataSize;
	int curr = 0;
	ShadeableIntersection tmpIntersection;
	tmpIntersection.t = 1e37f;
	bool intersected = false;
	while (curr >= 0 && curr < dev_sceneInfo.bvhDataSize)
	{
		bool outside = true;
		float boxt = boundingBoxIntersectionTest(currArray[curr].bbox, ray, outside);
		if (!outside) boxt = EPSILON;
		if (boxt > 0 && boxt < tmpIntersection.t)
		{
			if (currArray[curr].startPrim != -1)//leaf node
			{
				int start = currArray[curr].startPrim, end = currArray[curr].endPrim;
				bool intersect = util_bvh_leaf_intersect(start, end, dev_sceneInfo, ray, &tmpIntersection);
				intersected = intersected || intersect;
			}
			curr = currArray[curr].hitLink;
		}
		else
		{
			curr = currArray[curr].missLink;
		}
	}
	
	rayValid[path_index] = intersected;
	intersections[path_index].t = -1.0f;
	if (intersected)
	{
		intersections[path_index] = tmpIntersection;
		intersections[path_index].type = dev_sceneInfo.dev_materials[tmpIntersection.materialId].type;
		pathSegment.remainingBounces--;
	}
	else if (dev_sceneInfo.skyboxObj)
	{
		glm::vec2 uv = util_sample_spherical_map(glm::normalize(rayDir));
		float4 skyColorRGBA = tex2D<float4>(dev_sceneInfo.skyboxObj, uv.x, uv.y);
		glm::vec3 skyColor = glm::vec3(skyColorRGBA.x, skyColorRGBA.y, skyColorRGBA.z);
		image[pathSegment.pixelIndex] += pathSegment.color * skyColor;
	}
}



__global__ void generateGBuffer(
	int num_pixels,
	PathSegment* pathSegments,
	ShadeableIntersection* shadeableIntersections,
	SceneInfoPtrs dev_sceneInfo,
	SceneGbufferPtrs gbuffer,
	RenderState state
	)
{
	int path_index = blockIdx.x * blockDim.x + threadIdx.x;
	if (path_index >= num_pixels) return;
	ShadeableIntersection tmp_intersection = shadeableIntersections[path_index];
	PathSegment path = pathSegments[path_index];
	int pixelIndex = path.pixelIndex;
	gbuffer.dev_t[pixelIndex] = tmp_intersection.t;
	if (tmp_intersection.t < 0) return;

	gbuffer.dev_normal[pixelIndex] = util_vec3_to_oct(tmp_intersection.surfaceNormal);

	Material mat = dev_sceneInfo.dev_materials[tmp_intersection.materialId];
	glm::vec3 materialColor = mat.color;
	if (mat.baseColorMap)
	{
		float4 color = tex2D<float4>(mat.baseColorMap, tmp_intersection.uv.x, tmp_intersection.uv.y);
		materialColor.x = color.x;
		materialColor.y = color.y;
		materialColor.z = color.z;
	}

	if (mat.emittance > 0.0f) {
		gbuffer.dev_emission[pixelIndex] = mat.emittance * materialColor;
	}
	else {
		gbuffer.dev_albedo[pixelIndex] = materialColor;
	}

	glm::vec2 currUV = glm::vec2(pixelIndex % state.camera.resolution.x, pixelIndex / state.camera.resolution.x) / glm::vec2(state.camera.resolution);
	LastCameraInfo& lastCam = state.lastCamInfo;
	glm::vec2 lastUV = util_get_last_uv(tmp_intersection.worldPos, lastCam.position, lastCam.view, lastCam.up, lastCam.right, state.camera.resolution, state.camera.pixelLength);
	gbuffer.dev_velocity[pixelIndex] = currUV - lastUV;
	gbuffer.dev_z[pixelIndex] = glm::dot(state.camera.view, tmp_intersection.worldPos - state.camera.position);
	gbuffer.dev_primID[pixelIndex] = tmp_intersection.primitiveId;

}



__global__ void scatter_on_intersection(
	int iter
	, int num_paths
	, ShadeableIntersection* shadeableIntersections
	, PathSegment* pathSegments
	, SceneInfoPtrs sceneInfo
	, int* rayValid
	, glm::vec3* image
)
{
	Material* materials = sceneInfo.dev_materials;
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= num_paths) return;
	ShadeableIntersection intersection = shadeableIntersections[idx];
	// Set up the RNG
	// LOOK: this is how you use thrust's RNG! Please look at
	// makeSeededRandomEngine as well.
	thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
	thrust::uniform_real_distribution<float> u01(0, 1);

	Material material = materials[intersection.materialId];
	glm::vec3 materialColor = material.color;
#if VIS_NORMAL
	image[pathSegments[idx].pixelIndex] += (glm::normalize(intersection.surfaceNormal));
	rayValid[idx] = 0;
	return;
#endif

	// If the material indicates that the object was a light, "light" the ray
	if (material.type == MaterialType::emitting) {
		pathSegments[idx].color *= (materialColor * material.emittance);
		rayValid[idx] = 0;
		if (!util_math_is_nan(pathSegments[idx].color))
			image[pathSegments[idx].pixelIndex] += pathSegments[idx].color;
	}
	else {
		glm::vec3& woInWorld = pathSegments[idx].ray.direction;
		glm::vec3 nMap = glm::vec3(0, 0, 1);
		if (material.normalMap != 0)
		{
			float4 nMapCol = tex2D<float4>(material.normalMap, intersection.uv.x, intersection.uv.y);
			nMap.x = nMapCol.x;
			nMap.y = nMapCol.y;
			nMap.z = nMapCol.z;
			nMap = glm::pow(nMap, glm::vec3(1 / 2.2f));
			nMap = nMap * 2.0f - 1.0f;
			nMap = glm::normalize(nMap);
		}
		glm::vec3 N = glm::normalize(intersection.surfaceNormal);
		glm::vec3 B, T;
		if (material.normalMap != 0)
		{
			T = intersection.surfaceTangent;
			T = glm::normalize(T - N * glm::dot(N, T));
			B = glm::cross(N, T);
			N = glm::normalize(T * nMap.x + B * nMap.y + N * nMap.z);
		}
		else
		{
			util_math_get_TBN_pixar(N, &T, &B);
		}
		glm::mat3 TBN(T, B, N);
		glm::vec3 wo = glm::transpose(TBN) * (-woInWorld);
		wo = glm::normalize(wo);
		float pdf = 0;
		glm::vec3 wi, bxdf;
		glm::vec3 random = glm::vec3(u01(rng), u01(rng), u01(rng));
		if (material.type == MaterialType::metallicWorkflow)
		{
			float4 color = { 0,0,0,1 };
			float roughness = material.roughness, metallic = material.metallic;
			if (material.baseColorMap != 0)
			{
				color = tex2D<float4>(material.baseColorMap, intersection.uv.x, intersection.uv.y);
				materialColor.x = color.x;
				materialColor.y = color.y;
				materialColor.z = color.z;
			}
			if (material.metallicRoughnessMap != 0)
			{
				color = tex2D<float4>(material.metallicRoughnessMap, intersection.uv.x, intersection.uv.y);
				roughness = color.y;
				metallic = color.z;
			}

			bxdf = bxdf_metallic_workflow_sample_f(wo, &wi, random, &pdf, materialColor, metallic, roughness);
		}
		else if (material.type == MaterialType::frenselSpecular)
		{
			glm::vec2 iors = glm::dot(woInWorld, N) < 0 ? glm::vec2(1.0, material.indexOfRefraction) : glm::vec2(material.indexOfRefraction, 1.0);
			bxdf = bxdf_frensel_specular_sample_f(wo, &wi, glm::vec2(random.x, random.y), &pdf, materialColor, materialColor, iors);
		}
		else if (material.type == MaterialType::microfacet)
		{
			bxdf = bxdf_microfacet_sample_f(wo, &wi, glm::vec2(random.x, random.y), &pdf, materialColor, material.roughness);
		}
		else//diffuse
		{
			float4 color = { 0,0,0,1 };
			if (material.baseColorMap != 0)
			{
				color = tex2D<float4>(material.baseColorMap, intersection.uv.x, intersection.uv.y);
				materialColor.x = color.x;
				materialColor.y = color.y;
				materialColor.z = color.z;
				
			}
			
			if (color.w <= ALPHA_CUTOFF)
			{
				bxdf = pathSegments[idx].remainingBounces == 0 ? glm::vec3(0, 0, 0) : glm::vec3(1, 1, 1);
				wi = -wo;
				pdf = util_math_tangent_space_abscos(wi);
			}
			else
			{
				bxdf = bxdf_diffuse_sample_f(wo, &wi, glm::vec2(random.x, random.y), &pdf, materialColor);
			}

		}
		if (pdf > 0)
		{
			pathSegments[idx].color *= bxdf * util_math_tangent_space_abscos(wi) / pdf;
			glm::vec3 newDir = glm::normalize(TBN * wi);
			glm::vec3 offset = glm::dot(newDir, N) < 0 ? -N : N;
			float offsetMult = material.type != MaterialType::frenselSpecular ? SCATTER_ORIGIN_OFFSETMULT : SCATTER_ORIGIN_OFFSETMULT * 100.0f;
			pathSegments[idx].ray.origin = intersection.worldPos + offset * offsetMult;
			pathSegments[idx].ray.direction = newDir;
			rayValid[idx] = 1;
		}
		else
		{
			rayValid[idx] = 0;
		}

	}
}

__global__ void scatter_on_intersection_mis(
	int iter
	, int num_paths
	, ShadeableIntersection* shadeableIntersections
	, PathSegment* pathSegments
	, SceneInfoPtrs sceneInfo
	, int* rayValid
	, glm::vec3* image
)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= num_paths) return;
	ShadeableIntersection intersection = shadeableIntersections[idx];
	thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
	thrust::uniform_real_distribution<float> u01(0, 1);

	Material* materials = sceneInfo.dev_materials;
	Material material = materials[intersection.materialId];
	glm::vec3 materialColor = material.color;
#if VIS_NORMAL
	image[pathSegments[idx].pixelIndex] += (glm::normalize(intersection.surfaceNormal));
	rayValid[idx] = 0;
	return;
#endif

	// If the material indicates that the object was a light, "light" the ray
	if (material.type == MaterialType::emitting) {
		int lightPrimId = intersection.primitiveId;
		
		float matPdf = pathSegments[idx].lastMatPdf;
		if (matPdf > 0.0)
		{
			float G = util_math_solid_angle_to_area(intersection.worldPos, intersection.surfaceNormal, pathSegments[idx].ray.origin);
			float lightPdf = lights_sample_pdf(sceneInfo, lightPrimId);
			float misW = util_mis_weight_balanced(matPdf * G, lightPdf);
			pathSegments[idx].color *= (materialColor * material.emittance * misW);
		}
		else
		{
			pathSegments[idx].color *= (materialColor * material.emittance);
		}
		rayValid[idx] = 0;
		if (!util_math_is_nan(pathSegments[idx].color))
			image[pathSegments[idx].pixelIndex] += pathSegments[idx].color;
	}
	else {
		//Prepare normal and wo for sample
		glm::vec3& woInWorld = pathSegments[idx].ray.direction;
		glm::vec3 nMap = glm::vec3(0, 0, 1);
		if (material.normalMap != 0)
		{
			float4 nMapCol = tex2D<float4>(material.normalMap, intersection.uv.x, intersection.uv.y);
			nMap.x = nMapCol.x;
			nMap.y = nMapCol.y;
			nMap.z = nMapCol.z;
			nMap = glm::pow(nMap, glm::vec3(1 / 2.2f));
			nMap = nMap * 2.0f - 1.0f;
			nMap = glm::normalize(nMap);
		}
		glm::vec3 N = glm::normalize(intersection.surfaceNormal);
		glm::vec3 B, T;
		if (material.normalMap != 0)
		{
			T = intersection.surfaceTangent;
			T = glm::normalize(T - N * glm::dot(N, T));
			B = glm::cross(N, T);
			N = glm::normalize(T * nMap.x + B * nMap.y + N * nMap.z);
		}
		else
		{
			util_math_get_TBN_pixar(N, &T, &B);
		}
		glm::mat3 TBN(T, B, N);
		glm::vec3 wo = glm::transpose(TBN) * (-woInWorld);
		wo = glm::normalize(wo);
		float pdf = 0;
		glm::vec3 wi, bxdf;
		glm::vec3 random = glm::vec3(u01(rng), u01(rng), u01(rng));
		if (material.type == MaterialType::frenselSpecular)
		{
			glm::vec2 iors = glm::dot(woInWorld, N) < 0 ? glm::vec2(1.0, material.indexOfRefraction) : glm::vec2(material.indexOfRefraction, 1.0);
			bxdf = bxdf_frensel_specular_sample_f(wo, &wi, glm::vec2(random.x, random.y), &pdf, materialColor, materialColor, iors);
		}
		else
		{
			float roughness = material.roughness, metallic = material.metallic;
			float4 color = { 0,0,0,1 };
			float alpha = 1.0f;
			//Texture mapping
			if (material.baseColorMap != 0)
			{
				color = tex2D<float4>(material.baseColorMap, intersection.uv.x, intersection.uv.y);
				materialColor.x = color.x;
				materialColor.y = color.y;
				materialColor.z = color.z;
				alpha = color.w;
			}
			if (material.metallicRoughnessMap != 0)
			{
				color = tex2D<float4>(material.metallicRoughnessMap, intersection.uv.x, intersection.uv.y);
				roughness = color.y;
				metallic = color.z;
			}
			//Sampling lights
			glm::vec3 lightPos, lightNormal, emissive = glm::vec3(0);
			float lightPdf = -1.0;
			lights_sample(sceneInfo, glm::vec3(u01(rng), u01(rng), u01(rng)), intersection.worldPos, N, &lightPos, &lightNormal, &emissive, &lightPdf);
			
			if (emissive.x > 0.0 || emissive.y > 0.0 || emissive.z > 0.0)
			{
				glm::vec3 light_wi = lightPos - intersection.worldPos;
				light_wi = glm::normalize(glm::transpose(TBN) * (light_wi));
				float G = util_math_solid_angle_to_area(lightPos, glm::normalize(lightNormal), intersection.worldPos);
				float matPdf = -1.0f;
				if (material.type == MaterialType::metallicWorkflow)
				{
					matPdf = bxdf_metallic_workflow_pdf(wo, light_wi, materialColor, metallic, roughness);
					bxdf = bxdf_metallic_workflow_eval(wo, light_wi, materialColor, metallic, roughness);
				}
				else if (material.type == MaterialType::microfacet)
				{
					matPdf = bxdf_microfacet_pdf(wo, light_wi, roughness);
					bxdf = bxdf_microfacet_eval(wo, light_wi, materialColor, roughness);
				}
				else
				{
					matPdf = bxdf_diffuse_pdf(wo, light_wi);
					bxdf = bxdf_diffuse_eval(wo, light_wi, materialColor);
				}
				if (matPdf > 0.0)
				{
					float misW = util_mis_weight_balanced(lightPdf, matPdf * G);
					image[pathSegments[idx].pixelIndex] += pathSegments[idx].color * bxdf * util_math_tangent_space_abscos(light_wi) * emissive * misW * G / (lightPdf);
				}
			}
			//Sampling material bsdf
			if (material.type == MaterialType::metallicWorkflow)
			{	
				bxdf = bxdf_metallic_workflow_sample_f(wo, &wi, random, &pdf, materialColor, metallic, roughness);
			}
			else if (material.type == MaterialType::microfacet)
			{
				bxdf = bxdf_microfacet_sample_f(wo, &wi, glm::vec2(random.x, random.y), &pdf, materialColor, roughness);
			}
			else//diffuse
			{
				if (alpha <= ALPHA_CUTOFF)
				{
					bxdf = pathSegments[idx].remainingBounces == 0 ? glm::vec3(0, 0, 0) : glm::vec3(1, 1, 1);
					wi = -wo;
					pdf = util_math_tangent_space_abscos(wi);
				}
				else
				{
					bxdf = bxdf_diffuse_sample_f(wo, &wi, glm::vec2(random.x, random.y), &pdf, materialColor);
				}

			}
		}
		if (pdf > 0)
		{
			pathSegments[idx].color *= bxdf * util_math_tangent_space_abscos(wi) / pdf;
			glm::vec3 newDir = glm::normalize(TBN * wi);
			glm::vec3 offset = glm::dot(newDir, N) < 0 ? -N : N;
			float offsetMult = material.type != MaterialType::frenselSpecular ? SCATTER_ORIGIN_OFFSETMULT : SCATTER_ORIGIN_OFFSETMULT * 100.0f;
			pathSegments[idx].ray.origin = intersection.worldPos + offset * offsetMult;
			pathSegments[idx].ray.direction = newDir;
			pathSegments[idx].lastMatPdf = pdf;
			rayValid[idx] = 1;
		}
		else
		{
			rayValid[idx] = 0;
		}

	}
}


__global__ void addBackground(glm::vec3* dev_image, glm::vec3* dev_imageCache, int numPixels)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index >= numPixels) return;
	dev_image[index] += dev_imageCache[index];
}


// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3* image, PathSegment* iterationPaths)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index < nPaths)
	{
		PathSegment iterationPath = iterationPaths[index];
		if (!util_math_is_nan(iterationPath.color))
			image[iterationPath.pixelIndex] += iterationPath.color;
	}
}

struct mat_comp {
	__host__ __device__ bool operator()(const ShadeableIntersection& a, const ShadeableIntersection& b) const {
		return a.type < b.type;
	}
};

int compact_rays(int* rayValid,int* rayIndex,int numRays, bool sortByMat=false)
{
	thrust::device_ptr<PathSegment> dev_thrust_paths1(dev_paths1), dev_thrust_paths2(dev_paths2);
	thrust::device_ptr<ShadeableIntersection> dev_thrust_intersections1(dev_intersections1), dev_thrust_intersections2(dev_intersections2);
	thrust::device_ptr<int> dev_thrust_rayValid(rayValid), dev_thrust_rayIndex(rayIndex);
	thrust::exclusive_scan(dev_thrust_rayValid, dev_thrust_rayValid + numRays, dev_thrust_rayIndex);
	int nextNumRays, tmp;
	cudaMemcpy(&tmp, rayIndex + numRays - 1, sizeof(int), cudaMemcpyDeviceToHost);
	nextNumRays = tmp;
	cudaMemcpy(&tmp, rayValid + numRays - 1, sizeof(int), cudaMemcpyDeviceToHost);
	nextNumRays += tmp;
	thrust::scatter_if(dev_thrust_paths1, dev_thrust_paths1 + numRays, dev_thrust_rayIndex, dev_thrust_rayValid, dev_thrust_paths2);
	thrust::scatter_if(dev_thrust_intersections1, dev_thrust_intersections1 + numRays, dev_thrust_rayIndex, dev_thrust_rayValid, dev_thrust_intersections2);
	if (sortByMat)
	{
		mat_comp cmp;
		thrust::sort_by_key(dev_thrust_intersections2, dev_thrust_intersections2 + nextNumRays, dev_thrust_paths2, cmp);
	}
	std::swap(dev_paths1, dev_paths2);
	std::swap(dev_intersections1, dev_intersections2);
	return nextNumRays;
}

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(int iter, bool drawGbuffer) {
	const Camera& cam = hst_scene->state.camera;
	const int pixelcount = cam.resolution.x * cam.resolution.y;

	// 2D block for generating ray from camera
	const dim3 blockSize2d(8, 8);
	const dim3 blocksPerGrid2d(
		(cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
		(cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

	// 1D block for path tracing
	const int blockSize1d = 128;

	///////////////////////////////////////////////////////////////////////////

	// Recap:
	// * Initialize array of path rays (using rays that come out of the camera)
	//   * You can pass the Camera object to that kernel.
	//   * Each path ray must carry at minimum a (ray, color) pair,
	//   * where color starts as the multiplicative identity, white = (1, 1, 1).
	//   * This has already been done for you.
	// * For each depth:
	//   * Compute an intersection in the scene for each path ray.
	//     A very naive version of this has been implemented for you, but feel
	//     free to add more primitives and/or a better algorithm.
	//     Currently, intersection distance is recorded as a parametric distance,
	//     t, or a "distance along the ray." t = -1.0 indicates no intersection.
	//     * Color is attenuated (multiplied) by reflections off of any object
	//   * TODO: Stream compact away all of the terminated paths.
	//     You may use either your implementation or `thrust::remove_if` or its
	//     cousins.
	//     * Note that you can't really use a 2D kernel launch any more - switch
	//       to 1D.
	//   * TODO: Shade the rays that intersected something or didn't bottom out.
	//     That is, color the ray by performing a color computation according
	//     to the shader, then generate a new ray to continue the ray path.
	//     We recommend just updating the ray's PathSegment in place.
	//     Note that this step may come before or after stream compaction,
	//     since some shaders you write may also cause a path to terminate.
	// * Finally, add this iteration's results to the image. This has been done
	//   for you.

	// TODO: perform one iteration of path tracing
	SceneInfoPtrs dev_sceneInfo{};
	dev_sceneInfo.dev_materials = dev_materials;
	dev_sceneInfo.dev_objs = dev_objs;
	dev_sceneInfo.objectsSize = hst_scene->objects.size();
	dev_sceneInfo.modelInfo.dev_triangles = dev_triangles;
	dev_sceneInfo.modelInfo.dev_vertices = dev_vertices;
	dev_sceneInfo.modelInfo.dev_normals = dev_normals;
	dev_sceneInfo.modelInfo.dev_uvs = dev_uvs;
	dev_sceneInfo.modelInfo.dev_tangents = dev_tangents;
	dev_sceneInfo.modelInfo.dev_fsigns = dev_fsigns;
	dev_sceneInfo.dev_primitives = dev_primitives;
#if USE_BVH
#if MTBVH
	dev_sceneInfo.dev_mtbvhArray = dev_mtbvhArray;
	dev_sceneInfo.bvhDataSize = hst_scene->MTBVHArray.size() / 6;
#else
	dev_sceneInfo.dev_bvhArray = dev_bvhArray;
	dev_sceneInfo.bvhDataSize = hst_scene->bvhTreeSize;
#endif
#endif // 
	dev_sceneInfo.skyboxObj = hst_scene->skyboxTextureObj;
	dev_sceneInfo.dev_lights = dev_lights;
	dev_sceneInfo.lightsSize = hst_scene->lights.size();

	

	generateRayFromCamera << <blocksPerGrid2d, blockSize2d >> > (cam, iter, MAX_DEPTH, dev_paths1);
	checkCUDAError("generate camera ray");

	int depth = 0;
	PathSegment* dev_path_end = dev_paths1 + pixelcount;
	int num_paths = pixelcount;
	
	
	int numRays = num_paths;
	
	
	cudaDeviceSynchronize();
	// --- PathSegment Tracing Stage ---
	// Shoot ray into scene, bounce between objects, push shading chunks

	bool iterationComplete = false;
	while (numRays && depth < MAX_DEPTH) {

		// clean shading chunks
		cudaMemset(dev_intersections1, 0, pixelcount * sizeof(ShadeableIntersection));
		dim3 numblocksPathSegmentTracing = (numRays + blockSize1d - 1) / blockSize1d;
#if !STOCHASTIC_SAMPLING && FIRST_INTERSECTION_CACHING
		if (iter != 1 && depth == 0)
		{
			cudaMemcpy(dev_intersections1, dev_intersectionCache, pixelcount * sizeof(ShadeableIntersection), cudaMemcpyHostToHost);
			cudaMemcpy(dev_paths1, dev_pathCache, pixelcount * sizeof(PathSegment), cudaMemcpyHostToHost);
			cudaMemcpy(rayValid, dev_rayValidCache, sizeof(int) * pixelcount, cudaMemcpyHostToHost);
			addBackground << < numblocksPathSegmentTracing, blockSize1d >> > (dev_image, dev_imageCache, pixelcount);
		}
		if (iter == 1||(iter!=1&&depth>0))
		{
#endif
			// tracing
#if USE_BVH
#if MTBVH
			compute_intersection_bvh_stackless_mtbvh << <numblocksPathSegmentTracing, blockSize1d >> > (
				depth
				, numRays
				, dev_paths1
				, dev_sceneInfo
				, dev_intersections1
				, dev_rayValid
				, dev_image0
				);
#else
			compute_intersection_bvh_stackless << <numblocksPathSegmentTracing, blockSize1d >> > (
				depth
				, numRays
				, dev_paths1
				, dev_sceneInfo
				, dev_intersections1
				, rayValid
				, dev_image
				);
#endif
#else
			compute_intersection << <numblocksPathSegmentTracing, blockSize1d >> > (
				depth
				, numRays
				, dev_paths1
				, dev_objs
				, hst_scene->objects.size()
				, dev_triangles
				, dev_vertices
				, dev_uvs
				, dev_normals
				, hst_scene->skyboxTextureObj
				, dev_intersections1
				, rayValid
				, dev_image
				);
#endif
#if !STOCHASTIC_SAMPLING && FIRST_INTERSECTION_CACHING
		}
		if (iter == 1 && depth == 0)
		{
			cudaMemcpy(dev_intersectionCache, dev_intersections1, pixelcount * sizeof(ShadeableIntersection), cudaMemcpyHostToHost);
			cudaMemcpy(dev_pathCache, dev_paths1, pixelcount * sizeof(PathSegment), cudaMemcpyHostToHost);
			cudaMemcpy(dev_rayValidCache, rayValid, sizeof(int) * pixelcount, cudaMemcpyHostToHost);
			cudaMemcpy(dev_imageCache, dev_image, sizeof(glm::vec3) * pixelcount, cudaMemcpyHostToHost);
		}
#endif

		cudaDeviceSynchronize();
		checkCUDAError("trace one bounce");
		if (depth == 0 && drawGbuffer)
		{
			SceneGbufferPtrs dev_gbuffer;
			dev_gbuffer.dev_t = dev_g_t;
			dev_gbuffer.dev_albedo = dev_g_albedo;
			dev_gbuffer.dev_emission = dev_g_emission;
			dev_gbuffer.dev_normal = dev_g_normal;
			dev_gbuffer.dev_velocity = dev_g_velocity;
//			dev_gbuffer.dev_position = dev_g_position;
			dev_gbuffer.dev_z = dev_g_z;
			dev_gbuffer.dev_primID = dev_g_primID;
			generateGBuffer << <numblocksPathSegmentTracing, blockSize1d >> > (pixelcount, dev_paths1, dev_intersections1, dev_sceneInfo, dev_gbuffer, hst_scene->state);
		}
		cudaDeviceSynchronize();
		checkCUDAError("generate gbuffer");
		depth++;

#if SORT_BY_MATERIAL_TYPE
		numRays = compact_rays(rayValid, rayIndex, numRays, true);
#else
		numRays = compact_rays(dev_rayValid, dev_rayIndex, numRays);
#endif

		// TODO:
		// --- Shading Stage ---
		// Shade path segments based on intersections and generate new rays by
		// evaluating the BSDF.
		// Start off with just a big kernel that handles all the different
		// materials you have in the scenefile.
		// TODO: compare between directly shading the path segments and shading
		// path segments that have been reshuffled to be contiguous in memory.
		if (!numRays) break;
		dim3 numblocksLightScatter = (numRays + blockSize1d - 1) / blockSize1d;
#if USE_MIS
		scatter_on_intersection_mis << <numblocksPathSegmentTracing, blockSize1d >> > (
			iter,
			numRays,
			dev_intersections1,
			dev_paths1,
			dev_sceneInfo,
			dev_rayValid,
			dev_image0
			);
#else
		scatter_on_intersection << <numblocksPathSegmentTracing, blockSize1d >> > (
			iter,
			numRays,
			dev_intersections1,
			dev_paths1,
			dev_sceneInfo,
			dev_rayValid,
			dev_image0
			);
#endif
		cudaDeviceSynchronize();
		checkCUDAError("scatter_on_intersection_mis");

		numRays = compact_rays(dev_rayValid, dev_rayIndex, numRays);

		if (guiData != NULL)
		{
			guiData->TracedDepth = depth;
		}
	}

	//if (0)
	//{
	//	// Assemble this iteration and apply it to the image
	//	dim3 numBlocksPixels = (numRays + blockSize1d - 1) / blockSize1d;
	//	finalGather << <numBlocksPixels, blockSize1d >> > (numRays, dev_image0, dev_paths1);
	//}

	
	// Retrieve image from GPU
	cudaMemcpy(hst_scene->state.image.data(), dev_image0,
		pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
	
	checkCUDAError("pathtrace");
}

void showRenderedImage(uchar4* pbo, glm::ivec2 resolution, int iter, bool denoise)
{
	const dim3 blockSize2d(8, 8);
	const dim3 blocksPerGrid2d(
		(resolution.x + blockSize2d.x - 1) / blockSize2d.x,
		(resolution.y + blockSize2d.y - 1) / blockSize2d.y);
	// Send results to OpenGL buffer for rendering
	sendImageToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, resolution, iter, dev_image0);
}

void showGBuffer(uchar4* pbo, glm::ivec2 resolution, GBufferVisualizationType gbufType, int iter)
{
	SceneGbufferPtrs dev_gbuffer;
	dev_gbuffer.dev_t = dev_g_t;
	dev_gbuffer.dev_albedo = dev_g_albedo;
	dev_gbuffer.dev_emission = dev_g_emission;
	dev_gbuffer.dev_normal = dev_g_normal;
	dev_gbuffer.dev_velocity = dev_g_velocity;
//	dev_gbuffer.dev_position = dev_g_position;
	dev_gbuffer.dev_z = dev_g_z;
	dev_gbuffer.dev_primID = dev_g_primID;
	const dim3 blockSize2d(8, 8);
	const dim3 blocksPerGrid2d(
		(resolution.x + blockSize2d.x - 1) / blockSize2d.x,
		(resolution.y + blockSize2d.y - 1) / blockSize2d.y);
	int pixelCount = resolution.x * resolution.y;
	gbufferToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, resolution, dev_gbuffer, gbufType, hst_scene->state);
}

void svgfDenoise(int iter)
{
	const glm::ivec2& resolution = hst_scene->state.camera.resolution;
	SceneGbufferPtrs dev_gbuffer;
	dev_gbuffer.dev_t = dev_g_t;
	dev_gbuffer.dev_albedo = dev_g_albedo;
	dev_gbuffer.dev_emission = dev_g_emission;
	dev_gbuffer.dev_normal = dev_g_normal;
	dev_gbuffer.dev_velocity = dev_g_velocity;
	dev_gbuffer.dev_z = dev_g_z;
	dev_gbuffer.dev_znormalfwidth = dev_g_znormalfwidth;
	dev_gbuffer.dev_primID = dev_g_primID;
	computeDepthNormalFwidth(resolution, dev_gbuffer);
	cudaDeviceSynchronize();
	checkCUDAError("Compute fwidth");
	SVGFBufferPtrs dev_prevSVGFBuffer;
	dev_prevSVGFBuffer.dev_history_len = dev_histLen0;
	dev_prevSVGFBuffer.dev_illum = dev_illum0;
	dev_prevSVGFBuffer.dev_moments = dev_moments0;
	dev_prevSVGFBuffer.dev_normal = dev_prev_normal;
	dev_prevSVGFBuffer.dev_z = dev_prev_z;
	dev_prevSVGFBuffer.dev_primID = dev_prev_primID;
	SVGFBufferPtrs dev_currSVGFBuffer;
	dev_currSVGFBuffer.dev_history_len = dev_histLen1;
	dev_currSVGFBuffer.dev_illum = dev_illum1;
	dev_currSVGFBuffer.dev_moments = dev_moments1;
	SVGFAccumulation(resolution, dev_gbuffer, dev_prevSVGFBuffer, dev_currSVGFBuffer, dev_image0, iter);
	cudaDeviceSynchronize();
	checkCUDAError("SVGFAccumulation");
	//std::swap(dev_illum0, dev_illum1);
	
	
	
	SVGFBilateral(resolution, dev_gbuffer, dev_currSVGFBuffer, dev_prevSVGFBuffer.dev_illum);
	cudaDeviceSynchronize();
	checkCUDAError("SVGFBilateral");
	
	
	for (int i = 0; i < 5; i++)
	{
		dev_currSVGFBuffer.dev_illum = dev_illum0;
		SVGFWavelet(resolution, dev_gbuffer, dev_currSVGFBuffer, i, dev_illum1);
		std::swap(dev_illum0, dev_illum1);
	}
	cudaDeviceSynchronize();
	checkCUDAError("SVGFWavelet");
	
	
	SVGFCombine(resolution, dev_gbuffer, dev_illum0, dev_image0);
#if OCT_ENCODE_NORMAL
	cudaMemcpy(dev_prev_normal, dev_g_normal, resolution.x * resolution.y * sizeof(glm::vec2), cudaMemcpyDeviceToDevice);
#else
	cudaMemcpy(dev_prev_normal, dev_g_normal, resolution.x * resolution.y * sizeof(glm::vec3), cudaMemcpyDeviceToDevice);
#endif
	cudaMemcpy(dev_prev_z, dev_g_z, resolution.x * resolution.y * sizeof(float), cudaMemcpyDeviceToDevice);
	cudaMemcpy(dev_prev_primID, dev_g_primID, resolution.x * resolution.y * sizeof(int), cudaMemcpyDeviceToDevice);
	std::swap(dev_moments0, dev_moments1);
	std::swap(dev_histLen0, dev_histLen1);
}

void eawDenoise(int iter, EAWParams params, bool edgeAwared)
{
	const glm::ivec2 resolution = hst_scene->state.camera.resolution;
	cudaMemset(dev_image1, 0, resolution.x * resolution.y * sizeof(glm::ivec2));
	SceneGbufferPtrs dev_gbuffer;
	dev_gbuffer.dev_albedo   = dev_g_albedo;
	dev_gbuffer.dev_emission = dev_g_emission;
	dev_gbuffer.dev_normal   = dev_g_normal;
	dev_gbuffer.dev_z        = dev_g_z;
	dev_gbuffer.dev_primID   = dev_g_primID;
	for (int i = 0; i < 5; i++) {
		EAWFilter(resolution, dev_image0, dev_image1, dev_gbuffer, params, edgeAwared, i, i==0? iter:1, hst_scene->state.camera);
		std::swap(dev_image0, dev_image1);
	}
	//printf("now filter finished\n");
	
}

//void DrawGbuffer(int numIter)
//{
//	if (!USE_BVH) throw;
//
//	const Camera& cam = hst_scene->state.camera;
//	const int pixelcount = cam.resolution.x * cam.resolution.y;
//
//	SceneInfoPtrs dev_sceneInfo{};
//	dev_sceneInfo.dev_materials = dev_materials;
//	dev_sceneInfo.dev_objs = dev_objs;
//	dev_sceneInfo.objectsSize = hst_scene->objects.size();
//	dev_sceneInfo.modelInfo.dev_triangles = dev_triangles;
//	dev_sceneInfo.modelInfo.dev_vertices = dev_vertices;
//	dev_sceneInfo.modelInfo.dev_normals = dev_normals;
//	dev_sceneInfo.modelInfo.dev_uvs = dev_uvs;
//	dev_sceneInfo.modelInfo.dev_tangents = dev_tangents;
//	dev_sceneInfo.modelInfo.dev_fsigns = dev_fsigns;
//	dev_sceneInfo.dev_primitives = dev_primitives;
//#if USE_BVH
//#if MTBVH
//	dev_sceneInfo.dev_mtbvhArray = dev_mtbvhArray;
//	dev_sceneInfo.bvhDataSize = hst_scene->MTBVHArray.size() / 6;
//#else
//	dev_sceneInfo.dev_bvhArray = dev_bvhArray;
//	dev_sceneInfo.bvhDataSize = hst_scene->bvhTreeSize;
//#endif
//#endif // 
//	dev_sceneInfo.skyboxObj = hst_scene->skyboxTextureObj;
//
//	const dim3 blockSize2d(8, 8);
//	const dim3 blocksPerGrid2d(
//		(cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
//		(cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);
//	
//	const int blockSize1d = 128;
//	dim3 numblocksPathSegmentTracing = (pixelcount + blockSize1d - 1) / blockSize1d;
//	SceneGbufferPtrs dev_gbuffer;
//	glm::vec3* dev_albedo,*dev_normal;
//	cudaMalloc(&dev_albedo, pixelcount * sizeof(glm::vec3));
//	cudaMalloc(&dev_normal, pixelcount * sizeof(glm::vec3));
//	dev_gbuffer.dev_albedo = dev_albedo;
//	dev_gbuffer.dev_normal = dev_normal;
//	for (int i = 0; i < numIter; i++)
//	{
//		generateRayFromCamera << <blocksPerGrid2d, blockSize2d >> > (cam, i, MAX_DEPTH, dev_paths1);
//		draw_gbuffer_from_scratch << <numblocksPathSegmentTracing, blockSize1d >> > (pixelcount, dev_paths1, dev_sceneInfo, dev_gbuffer);
//	}
//	hst_scene->state.albedo.resize(pixelcount);
//	hst_scene->state.normal.resize(pixelcount);
//	cudaMemcpy(hst_scene->state.albedo.data(), dev_albedo, pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
//	cudaMemcpy(hst_scene->state.normal.data(), dev_normal, pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
//	cudaFree(dev_albedo);
//	cudaFree(dev_normal);
//	for (int i = 0; i < pixelcount; i++)
//	{
//		hst_scene->state.albedo[i] /= (float)numIter;
//		hst_scene->state.normal[i] /= (float)numIter;
//	}
//}
