#include "sceneStructs.h"
#include <cstdio>
#include <cuda.h>
#include <cmath>
#include "utilities.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "denoise.h"

__device__ constexpr float gaussian3x3[] = {
	1.0f / 16.0f, 2.0f / 16.0f, 1.0f / 16.0f,
	2.0f / 16.0f, 4.0f / 16.0f, 2.0f / 16.0f,
	1.0f / 16.0f, 2.0f / 16.0f, 1.0f / 16.0f
};

__device__ constexpr float gaussian5x5[] = {
		1.0 / 256.0, 4.0 / 256.0 , 6.0 / 256.0, 4.0 / 256.0, 1.0 / 256.0,
		4.0 / 256.0, 16.0 / 256.0 , 24.0 / 256.0, 16.0 / 256.0, 4.0 / 256.0,
		6.0 / 256.0, 24.0 / 256.0 , 36.0 / 256.0, 24.0 / 256.0, 6.0 / 256.0,
		4.0 / 256.0, 16.0 / 256.0 , 24.0 / 256.0, 16.0 / 256.0, 4.0 / 256.0,
		1.0 / 256.0, 4.0 / 256.0 , 6.0 / 256.0, 4.0 / 256.0, 1.0 / 256.0
};

__global__ void eawFilter(glm::ivec2 resolution, const glm::vec3* inImage, glm::vec3* outImage, SceneGbufferPtrs gBuffer, EAWParams params, bool edgeAwared, int iter, int luminDiv, Camera cam)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int stepSize = 1 << iter;
	
	if (x < resolution.x && y < resolution.y) {
		
		glm::vec3 color   = glm::vec3(0.0);
		float weightSum   = 0.0f;
		int curIdx        = x + y * resolution.x;
		glm::vec3 cColor  = inImage[curIdx] / (float)luminDiv;
		glm::vec3 cAlbedo = gBuffer.dev_albedo[curIdx];
		glm::vec3 cNormal = util_oct_to_vec3(gBuffer.dev_normal[curIdx]);
		glm::vec3 cPos    = util_get_world_position(glm::vec2(x, y), gBuffer.dev_z[curIdx], cam.position, cam.view, cam.up, cam.right, resolution, cam.pixelLength);
		int curPrimID = gBuffer.dev_primID[curIdx];
		for (int dx = -2; dx < 3; dx++) {
			for (int dy = -2; dy < 3; dy++) {
				int nx = x + dx * stepSize;
				int ny = y + dy * stepSize;
				if (nx < 0 || nx >= resolution.x || ny < 0 || ny >= resolution.y)continue;
				int nIdx = nx + ny * resolution.x;
				int i = dx + 2 + (dy + 2) * 5;
				//printf("edgeAwared: %d\n", edgeAwared == false);
				if (edgeAwared == false) {
					//printf("just gaussian blur\n");
					color += inImage[nIdx] / (float)luminDiv * gaussian5x5[i];
					weightSum += gaussian5x5[i];
				}
				else {
					glm::vec3 nColor  = inImage[nIdx] / (float)luminDiv;
					glm::vec3 nAlbedo = gBuffer.dev_albedo[nIdx];
					glm::vec3 nNormal = util_oct_to_vec3(gBuffer.dev_normal[nIdx]);
					glm::vec3 nPos    = util_get_world_position(glm::vec2(x, y), gBuffer.dev_z[nIdx], cam.position, cam.view, cam.up, cam.right, resolution, cam.pixelLength);
					int nPrimID = gBuffer.dev_primID[nIdx];
					if (nPrimID != curPrimID) {
						continue;
					}
					float dist2;
					dist2 = glm::distance2(nColor, cColor);
					float cw = i == 0 ? 1.0 : min(exp(-dist2 * params.sigmaCol * stepSize), 1.0);
					dist2 = glm::distance2(nAlbedo, cAlbedo);
					float aw = min(exp(-dist2 * params.sigmaAlbe * stepSize), 1.0);
					dist2 = glm::distance2(nNormal, cNormal);
					float nw = min(exp(-dist2 * params.sigmaNorm * stepSize), 1.0);
					dist2 = glm::distance2(cPos, nPos);
					float pw = min(exp(-dist2 * params.sigmaPos * stepSize), 1.0);
					float eaW = cw * aw * nw * pw;
					color += nColor * eaW * gaussian5x5[i];
					weightSum += eaW * gaussian5x5[i];

				}

			}
		}
		outImage[curIdx] = color / max(0.000001f, weightSum);
	}
}

void EAWFilter(glm::ivec2 resolution, const glm::vec3* inImage, glm::vec3* outImage, SceneGbufferPtrs gbuffer, EAWParams params, bool edgeAwared, int iter, int luminDiv, Camera cam)
{
	const dim3 blockSize2d(8, 8);
	const dim3 blocksPerGrid2d(
		(resolution.x + blockSize2d.x -1) / blockSize2d.x,
		(resolution.y + blockSize2d.y - 1) / blockSize2d.y
	);
	//printf("now in filter %d\n", edgeAwared);
	eawFilter << <blocksPerGrid2d, blockSize2d >> > (resolution, inImage, outImage, gbuffer, params, edgeAwared, iter, luminDiv, cam);
}


__global__ void ComputeDepthNormalFwidth(glm::ivec2 resolution, SceneGbufferPtrs gbuffer)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x < resolution.x && y < resolution.y) {
		float zddx, zddy;
		glm::vec2 znormalfwidth;
		if (x < resolution.x)
			zddx = abs(gbuffer.dev_z[x + 1 + (y * resolution.x)] - gbuffer.dev_z[x + (y * resolution.x)]);
		else
			zddx = abs(gbuffer.dev_z[x + (y * resolution.x)] - gbuffer.dev_z[x - 1 + (y * resolution.x)]);
		if (y < resolution.y)
			zddy = abs(gbuffer.dev_z[x + ((y + 1) * resolution.x)] - gbuffer.dev_z[x + (y * resolution.x)]);
		else
			zddy = abs(gbuffer.dev_z[x + (y * resolution.x)] - gbuffer.dev_z[x + (y - 1) * resolution.x]);
		znormalfwidth.x = max(zddx, zddy);
		glm::vec3 nddx = glm::vec3(0);
#if OCT_ENCODE_NORMAL
		if (x < resolution.x)
			nddx += abs(util_oct_to_vec3(gbuffer.dev_normal[x + 1 + (y * resolution.x)]) - util_oct_to_vec3(gbuffer.dev_normal[x + (y * resolution.x)]));
		else
			nddx += abs(util_oct_to_vec3(gbuffer.dev_normal[x + (y * resolution.x)]) - util_oct_to_vec3(gbuffer.dev_normal[x - 1 + (y * resolution.x)]));
		if (y < resolution.y)
			nddx += abs(util_oct_to_vec3(gbuffer.dev_normal[x + ((y + 1) * resolution.x)]) - util_oct_to_vec3(gbuffer.dev_normal[x + (y * resolution.x)]));
		else
			nddx += abs(util_oct_to_vec3(gbuffer.dev_normal[x + (y * resolution.x)]) - util_oct_to_vec3(gbuffer.dev_normal[x + (y - 1) * resolution.x]));
#else
		if (x < resolution.x)
			nddx += abs(gbuffer.dev_normal[x + 1 + (y * resolution.x)] - gbuffer.dev_normal[x + (y * resolution.x)]);
		else
			nddx += abs(gbuffer.dev_normal[x + (y * resolution.x)] - gbuffer.dev_normal[x - 1 + (y * resolution.x)]);
		if (y < resolution.y)
			nddx += abs(gbuffer.dev_normal[x + ((y + 1) * resolution.x)] - gbuffer.dev_normal[x + (y * resolution.x)]);
		else
			nddx += abs(gbuffer.dev_normal[x + (y * resolution.x)] - gbuffer.dev_normal[x + (y - 1) * resolution.x]);
#endif
		znormalfwidth.y = length(nddx);
		gbuffer.dev_znormalfwidth[x + (y * resolution.x)] = znormalfwidth;
	}
}

void computeDepthNormalFwidth(glm::ivec2 resolution, SceneGbufferPtrs gbuffer)
{
	const dim3 blockSize2d(8, 8);
	const dim3 blocksPerGrid2d(
		(resolution.x + blockSize2d.x - 1) / blockSize2d.x,
		(resolution.y + blockSize2d.y - 1) / blockSize2d.y);
	ComputeDepthNormalFwidth << <blocksPerGrid2d, blockSize2d >> > (resolution, gbuffer);
}

__device__ inline bool util_check_reproj_valid(float z, float zPrev, float zfwidth, glm::vec3 normal, glm::vec3 prevNormal, float normalfwidth)
{
	if (abs(z - zPrev) / (zfwidth + 1e-2f) > ZDIFF_THRESHOLD) return false;
	if (glm::distance(normal, prevNormal) / (normalfwidth + 1e-3f) > NORMALDIFF_THRESHOLD) return false;
	return true;
}

__device__ bool util_load_filter_prev_frame(glm::ivec2 pos, SceneGbufferPtrs& gbuffer, SVGFBufferPtrs& prevSVGFBuffer, glm::ivec2 resolution, glm::vec3* filteredIllum, glm::vec2* filteredMome, float* historyLen)
{
	int currPosIdx = pos.x + pos.y * resolution.x;
	glm::vec2 prevpos = glm::vec2(pos) - gbuffer.dev_velocity[currPosIdx] * glm::vec2(resolution);
	glm::vec2 znormalfw = gbuffer.dev_znormalfwidth[currPosIdx];
	float z = gbuffer.dev_z[currPosIdx];
	glm::vec3 normal = util_oct_to_vec3(gbuffer.dev_normal[currPosIdx]);
	int id = gbuffer.dev_primID[currPosIdx];

	bool valid[4] = { false,false,false,false };
	bool anyValid = false;
	for (int i = 0; i < 4; i++)
	{
		glm::ivec2 npos = prevpos + glm::vec2(i & 1, i >> 1);
		int nPosIdx = npos.x + npos.y * resolution.x;
		if (npos.x < 0 || npos.x >= resolution.x || npos.y < 0 || npos.y >= resolution.y)
		{
			valid[i] = false;
			continue;
		}
		/*int idprev = prevSVGFBuffer.dev_primID[nPosIdx];
		if (idprev != id)
		{
			valid[i] = false;
			continue;
		}*/
		float zprev = prevSVGFBuffer.dev_z[nPosIdx];
		glm::vec3 prevnormal = util_oct_to_vec3(prevSVGFBuffer.dev_normal[nPosIdx]);
		valid[i] = util_check_reproj_valid(z, zprev, znormalfw.x, normal, prevnormal, znormalfw.y);
		anyValid = valid[i] || anyValid;
	}
	glm::vec3 prevIllum = glm::vec3(0);
	glm::vec2 prevMoments = glm::vec2(0);
	if (anyValid)
	{
		float xf = prevpos.x - (int)prevpos.x;
		float yf = prevpos.y - (int)prevpos.y;
		const float w[4] = { (1 - xf) * (1 - yf), xf * (1 - yf), (1 - xf) * yf, xf * yf };
		float totalW = 0.0f;
		for (int i = 0; i < 4; i++)
		{
			glm::ivec2 npos = prevpos + glm::vec2(i & 1, i >> 1);
			int nPosIdx = npos.x + npos.y * resolution.x;
			if (npos.x < 0 || npos.x >= resolution.x || npos.y < 0 || npos.y >= resolution.y) continue;
			if (valid[i])
			{
				prevIllum += glm::vec3(prevSVGFBuffer.dev_illum[nPosIdx]) * w[i];
				prevMoments += prevSVGFBuffer.dev_moments[nPosIdx] * w[i];
				totalW += w[i];
			}
		}

		anyValid = totalW > 0.0f ? 1 : 0;
		if (anyValid)
		{
			prevIllum /= totalW;
			prevMoments /= totalW;
		}
	}
	
	if (!anyValid)
	{
		float validCnt = 0;
		for(int dx=-1;dx<2;dx++)
			for (int dy = -1; dy < 2; dy++)
			{
				glm::ivec2 npos = prevpos + glm::vec2(dx, dy);
				int nPosIdx = npos.x + npos.y * resolution.x;
				if (npos.x < 0 || npos.x >= resolution.x || npos.y < 0 || npos.y >= resolution.y) continue;
				/*int idprev = prevSVGFBuffer.dev_primID[nPosIdx];
				if (idprev != id) continue;*/
				float zprev = prevSVGFBuffer.dev_z[nPosIdx];
				glm::vec3 prevnormal = util_oct_to_vec3(prevSVGFBuffer.dev_normal[nPosIdx]);
				if (util_check_reproj_valid(z, zprev, znormalfw.x, normal, prevnormal, znormalfw.y))
				{
					validCnt += 1;
					prevIllum += glm::vec3(prevSVGFBuffer.dev_illum[nPosIdx]);
					prevMoments += prevSVGFBuffer.dev_moments[nPosIdx];
				}
			}
		if (validCnt > 0.0f)
		{
			anyValid = true;
			prevIllum /= validCnt;
			prevMoments /= validCnt;
		}
	}
	
	*filteredIllum = anyValid ? prevIllum : glm::vec3(0.0f);
	*filteredMome = anyValid ? prevMoments : glm::vec2(0.0f);
	anyValid = anyValid && ((int)prevpos.x >= 0 && (int)prevpos.x < resolution.x && (int)prevpos.y >= 0 && (int)prevpos.y < resolution.y);
	*historyLen = anyValid ? prevSVGFBuffer.dev_history_len[(int)prevpos.x + (int)prevpos.y * resolution.x] : 0;
	
	return anyValid;
}

//TODO: remove duplicate
__device__ inline float util_math_luminance1(glm::vec3 col)
{ 
	return 0.299f * col.r + 0.587f * col.g + 0.114f * col.b;
}

__device__ inline float util_compute_stopping_func(float depth, float depthN, float depthCoeff, glm::vec3 normal, glm::vec3 normalN, float normalCoeff, glm::vec3 illum, glm::vec3 illumN, float illumCoeff, int iter)
{
	float zwexp = glm::abs(depth - depthN) / glm::max(1e-3f, depthCoeff);
	float iwexp = glm::pow(glm::length(illum - illumN), 1.0f) / glm::max(1e-3f, illumCoeff);
	float nw = glm::pow(glm::clamp(glm::dot(normal, normalN), 0.0f, 1.0f), normalCoeff);
	return glm::exp(-zwexp - iwexp) * nw;
}

__global__ void svgfAccumulation(glm::ivec2 resolution, SceneGbufferPtrs gbuffer, SVGFBufferPtrs prevSVGFBuffer, SVGFBufferPtrs currSVGFBuffer, glm::vec3* radiance, int iter)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x >= resolution.x && y >= resolution.y) return;
	int curIndex = x + y * resolution.x;
	glm::vec3 albedo    = gbuffer.dev_albedo[curIndex];
	glm::vec3 emission  = gbuffer.dev_emission[curIndex];
	glm::vec3 currIllum = (radiance[curIndex] / (float)iter - emission) / max(albedo, glm::vec3(1e-3f));
	glm::vec3 prevIllum = glm::vec3(0.0, 0.0, 0.0);
	glm::vec2 prevMome  = glm::vec2(0.0, 0.0);
	float histLen       = 0.0f;
	bool usePrev = util_load_filter_prev_frame(glm::ivec2(x, y), gbuffer, prevSVGFBuffer, resolution, &prevIllum, &prevMome, &histLen);
	histLen = usePrev ? histLen + 1.0f : 1.0f;
	histLen = glm::min(histLen, MAX_HIST_LEN);
	float alphaLum = usePrev ? glm::max(LUMIN_BLEND_ALPHA, 1.0f / histLen) : 1.0f;
	float alphaMome = usePrev ? glm::max(MOMENTS_BLEND_ALPHA, 1.0f / histLen) : 1.0f;
	float lumin = util_math_luminance1(currIllum);
	glm::vec2 currMome;
	currMome.x = lumin;
	currMome.y = lumin * lumin;

	currMome  = alphaMome * currMome + (1 - alphaMome) * prevMome;
	currIllum = alphaLum * currIllum + (1 - alphaLum) * prevIllum;
	float sigma = glm::max(0.0f, currMome.y - currMome.x * currMome.x);
	currSVGFBuffer.dev_illum[curIndex]   = glm::vec4(currIllum, sigma);
	currSVGFBuffer.dev_moments[curIndex] = currMome;
	currSVGFBuffer.dev_history_len[curIndex] = histLen;
}

void SVGFAccumulation(glm::ivec2 resolution, SceneGbufferPtrs gbuffer, SVGFBufferPtrs prevSVGFBuffer, SVGFBufferPtrs currSVGFBuffer, glm::vec3* radiance, int iter)
{
	const dim3 blockSized2d(8, 8);
	const dim3 blocksPerGrid2d(
		(resolution.x + blockSized2d.x -1) / blockSized2d.x,
		(resolution.y + blockSized2d.y - 1) / blockSized2d.y
	);
	svgfAccumulation << <blocksPerGrid2d, blockSized2d >> > (resolution, gbuffer, prevSVGFBuffer, currSVGFBuffer, radiance, iter);
}

__global__ void svgfBilateralFilter(glm::ivec2 resolution, SceneGbufferPtrs gbuffer, SVGFBufferPtrs currSVGFBuffer, glm::vec4* filteredIllum)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x >= resolution.x && y >= resolution.y) return;
	int currIdx = x + y * resolution.x;
	float histLen = currSVGFBuffer.dev_history_len[currIdx];
	float depthC = gbuffer.dev_z[currIdx];
	glm::vec3 normalC = util_oct_to_vec3(gbuffer.dev_normal[currIdx]);
	glm::vec4 illumC = currSVGFBuffer.dev_illum[currIdx];
	glm::vec2 znormalwidth = gbuffer.dev_znormalfwidth[currIdx];
	float depthCoeff  = glm::max(znormalwidth.x, 1e-8f) * 3.0f;
	float normalCoeff = glm::max(znormalwidth.y, 1e-8f) * 3.0f;
	float luminCoeff = LUMIN_COEFF;
	float totalW = 0.0f;
	glm::vec3 totalIllum = glm::vec3(0.0f);
	glm::vec2 totalMoments = glm::vec2(0.0f);
	if (histLen >= 4.0f) {
		filteredIllum[currIdx] = currSVGFBuffer.dev_illum[currIdx];
		return;
	}
	else {
		for (int dx = -3; dx < 4; dx++) {
			for (int dy = -3; dy < 4; dy++) {
				int nx = x + dx, ny = y + dy;
				int nIdx = nx + ny * resolution.x;
				if (nx >= 0 && nx < resolution.x && ny >= 0 && ny < resolution.y) {

					float depthN = gbuffer.dev_z[nIdx];
					glm::vec3 normalN  = util_oct_to_vec3(gbuffer.dev_normal[nIdx]);
					glm::vec3 illumN   = glm::vec3(currSVGFBuffer.dev_illum[nIdx]);
					glm::vec2 momentsN = currSVGFBuffer.dev_moments[nIdx];
					float w = util_compute_stopping_func(depthC, depthN, depthCoeff, normalC, normalN, normalCoeff, 
						                                 glm::vec3(illumC), glm::vec3(illumN), luminCoeff, 1);
					totalW += w;
					totalIllum += w * illumN;
					totalMoments += w * momentsN;
				}
			}
		}
		totalW = glm::max(totalW, 1e-9f);
		totalIllum /= totalW;
		totalMoments /= totalW;
		float var = totalMoments.y - totalMoments.x * totalMoments.x;
		var *= 4.0f / histLen;
		filteredIllum[currIdx] = glm::vec4(totalIllum, var);
	}
}

void SVGFBilateral(glm::ivec2 resolution, SceneGbufferPtrs gbuffer, SVGFBufferPtrs currSVGFBuffer, glm::vec4* filteredIllum)
{
	const dim3 blockSize2d(8, 8);
	const dim3 blocksPerGrid2d(
		(resolution.x + blockSize2d.x - 1) / blockSize2d.x,
		(resolution.y + blockSize2d.y - 1) / blockSize2d.y);
	svgfBilateralFilter << <blocksPerGrid2d, blockSize2d >> > (resolution, gbuffer, currSVGFBuffer, filteredIllum);
}

__global__ void svgfWaveletFilter(glm::ivec2 resolution, SceneGbufferPtrs gbuffer, SVGFBufferPtrs currSVGFBuffer, int iter, glm::vec4* filteredIllum)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x >= resolution.x && y >= resolution.y) return;

	int currIdx = x + y * resolution.x;
	int stepSize = 1 << iter;
	glm::vec4 illumC = currSVGFBuffer.dev_illum[currIdx];
	float totalVar = 0.0, totalW = 0.0;
	for (int dx = -1; dx < 2; dx++) {
		for (int dy = -1; dy < 2; dy++) {
			int nx = x + dx, ny = y + dy;
			if (nx >= 0 && nx < resolution.x && ny >= 0 && ny < resolution.y) {
				totalVar += currSVGFBuffer.dev_illum[nx + ny * resolution.x].w * gaussian3x3[(dx + 1) + (dy + 1) * 3];
				totalW += gaussian3x3[(dx + 1) + (dy + 1) * 3];
			}
		}
	}
	totalVar /= max(totalW, 1e-10f);
	glm::vec3 normalC = util_oct_to_vec3(gbuffer.dev_normal[currIdx]);
	float histLen = currSVGFBuffer.dev_history_len[currIdx];
	float depthC = gbuffer.dev_z[currIdx];
	totalW = 0.0f;
	glm::vec4 totalIllum = glm::vec4(0.0f);
	float luminCoeff = LUMIN_COEFF * sqrt(max(0.0f, abs(totalVar))) + 1e-8f;
	float depthCoeff = DEPTH_COEFF * max(1e-6f, gbuffer.dev_znormalfwidth[currIdx].x) * stepSize;
	float normalCoeff = NORMAL_COEFF;
	for (int dx = -2; dx < 3; dx++) {
		for (int dy = -2; dy < 3; dy++) {
			int nx = x + dx * stepSize, ny = y + dy * stepSize;
			int nIdx = nx + ny * resolution.x;
			if (nx >= 0 && nx < resolution.x && ny >= 0 && ny < resolution.y) {
				float weight = gaussian5x5[dx + 2 + (dy + 2) * 5];
				glm::vec4 illumN = currSVGFBuffer.dev_illum[nIdx];
				float depthN = gbuffer.dev_z[nIdx];
				glm::vec3 normalN = util_oct_to_vec3(gbuffer.dev_normal[nIdx]);
				float w = util_compute_stopping_func(depthC, depthN, depthCoeff * glm::length(glm::vec2(dx, dy)), normalC, normalN, normalCoeff, glm::vec3(illumC), glm::vec3(illumN), luminCoeff, iter);
				weight *= w;
				totalIllum += illumN * glm::vec4(glm::vec3(weight), weight * weight);
				totalW += weight;
			}

		}
	}
	totalIllum /= glm::max(glm::vec4(1e-6f), glm::vec4(glm::vec3(totalW), totalW * totalW));
	//printf("totalIllum: %f %f %f %f", totalIllum.x, totalIllum.y, totalIllum.z, totalIllum.w);
	filteredIllum[currIdx] = totalIllum;
}

void SVGFWavelet(glm::ivec2 resolution, SceneGbufferPtrs gbuffer, SVGFBufferPtrs currSVGFBuffer, int iter, glm::vec4* filteredIllum)
{
	const dim3 blockSize2d(8, 8);
	const dim3 blocksPerGrid2d(
		(resolution.x + blockSize2d.x - 1) / blockSize2d.x,
		(resolution.y + blockSize2d.y - 1) / blockSize2d.y);
	svgfWaveletFilter << <blocksPerGrid2d, blockSize2d >> > (resolution, gbuffer, currSVGFBuffer, iter, filteredIllum);
}

__global__ void svgfCombine(glm::ivec2 resolution, SceneGbufferPtrs gbuffer, glm::vec4* filteredIllum, glm::vec3* image)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x >= resolution.x && y >= resolution.y) return;
	int currIdx = x + y * resolution.x;
	image[currIdx] = glm::vec3(filteredIllum[currIdx]) * gbuffer.dev_albedo[currIdx] + gbuffer.dev_emission[currIdx];
}

void SVGFCombine(glm::ivec2 resolution, SceneGbufferPtrs gbuffer, glm::vec4* filteredIllum, glm::vec3* image)
{
	const dim3 blockSize2d(8, 8);
	const dim3 blocksPerGrid2d(
		(resolution.x + blockSize2d.x - 1) / blockSize2d.x,
		(resolution.y + blockSize2d.y - 1) / blockSize2d.y);
	svgfCombine << <blocksPerGrid2d, blockSize2d >> > (resolution, gbuffer, filteredIllum, image);
}

