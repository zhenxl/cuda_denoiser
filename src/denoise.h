#pragma once

#include "glm/glm.hpp"
void EAWFilter(glm::ivec2 resolution, const glm::vec3* inImage, glm::vec3* outImage, SceneGbufferPtrs gbuffer, EAWParams params, bool edgeAwared, int iter, int luminDiv, Camera cam);
void computeDepthNormalFwidth(glm::ivec2 resolution, SceneGbufferPtrs gbuffer);
void SVGFAccumulation(glm::ivec2 resolution, SceneGbufferPtrs gbuffer, SVGFBufferPtrs prevSVGFBuffer, SVGFBufferPtrs currSVGFBuffer, glm::vec3* radiance, int iter);
void SVGFBilateral(glm::ivec2 resolution, SceneGbufferPtrs gbuffer, SVGFBufferPtrs currSVGFBuffer, glm::vec4* filteredIllum);
void SVGFWavelet(glm::ivec2 resolution, SceneGbufferPtrs gbuffer, SVGFBufferPtrs currSVGFBuffer, int iter, glm::vec4* filteredIllum);
void SVGFCombine(glm::ivec2 resolution, SceneGbufferPtrs gbuffer, glm::vec4* filteredIllum, glm::vec3* image);

__device__ inline glm::vec2 util_get_last_uv(glm::vec3 worldPos, glm::vec3 lastCamPos, glm::vec3 lastCamView, glm::vec3 lastCamUp, glm::vec3 lastCamRight, glm::ivec2 resolution, glm::vec2 pixelLength)
{
	glm::vec3 ori = lastCamPos;
	glm::vec3 dir = glm::normalize(worldPos - ori);
	float NdotDir = glm::dot(dir, lastCamView);
	if (abs(NdotDir) < 1e-10f) return glm::vec2(-1.0f);
	glm::vec3 screenCenter = ori + lastCamView;
	float d = glm::dot(-lastCamView, screenCenter);
	float t = -(glm::dot(lastCamView, ori) + d) / NdotDir;
	if (t < 0.0f) return glm::vec2(-1.0f);
	glm::vec3 screenPos = ori + dir * t;
	glm::vec3 screenOri = screenCenter + 0.5f * resolution.x * pixelLength.x * lastCamRight + 0.5f * resolution.y * pixelLength.y * lastCamUp;
	glm::vec2 uv;
	uv.x = glm::dot(screenPos - screenOri, lastCamRight);
	uv.y = glm::dot(screenPos - screenOri, lastCamUp);
	uv /= (glm::vec2(resolution) * pixelLength);
	return -uv;
}

__device__ inline glm::vec3 util_get_world_position(const glm::vec2& pixel, float z, const glm::vec3& camPos, const  glm::vec3& camView, const  glm::vec3& camUp, const  glm::vec3& camRight, const  glm::ivec2& resolution, const glm::vec2& pixelLength)
{
	glm::vec3 dir = glm::normalize(camView
		- camRight * pixelLength.x * (pixel.x - (float)resolution.x * 0.5f)
		- camUp * pixelLength.y * (pixel.y - (float)resolution.y * 0.5f));
	float costheta = glm::dot(glm::normalize(camView), dir);
	float len = z / costheta;
	return camPos + dir * len;
}

__device__ inline glm::vec2 util_vec2_sgn(glm::vec2 v)
{
	return glm::vec2(v.x >= 0.0 ? 1.0f : -1.0f, v.y >= 0.0 ? 1.0f : -1.0f);
}

__device__  inline glm::vec2 util_vec3_to_oct(glm::vec3 v)
{
	glm::vec2 oct = glm::vec2(v.x, v.y) / (abs(v.x) + abs(v.y) + abs(v.z));
	return v.z > 0.0 ? oct : ((1.0f - glm::abs(glm::vec2(oct.y, oct.x))) * util_vec2_sgn(oct));
}

__device__ inline glm::vec3 util_oct_to_vec3(glm::vec2 oct)
{
	glm::vec3 v = glm::vec3(oct, 1.0 - abs(oct.x) - abs(oct.y));
	if (v.z<0.0f)
	{
		glm::vec2 sgn = util_vec2_sgn(glm::vec2(v));
		v.x = 1.0f - abs(v.y) * sgn.x;
		v.y = 1.0f - abs(v.x) * sgn.y;
	}
	return glm::normalize(v);
}

