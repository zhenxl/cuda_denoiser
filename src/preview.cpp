//#define _CRT_SECURE_NO_DEPRECATE
#include <ctime>
#include <chrono>
#include "main.h"
#include "preview.h"
#include "ImGui/imgui.h"
#include "ImGui/imgui_impl_glfw.h"
#include "ImGui/imgui_impl_opengl3.h"
GLuint positionLocation = 0;
GLuint texcoordsLocation = 1;
GLuint pbo;
GLuint displayImage;

#define CAM_SPEED 5.0f;

GLFWwindow* window;
GuiDataContainer* imguiData = NULL;
ImGuiIO* io = nullptr;
bool mouseOverImGuiWinow = false;
float camAnimTime = 0.0f;

extern RenderState* renderState;
extern float zoom, theta, phi;
extern bool camchanged;

std::string currentTimeString() {
	time_t now;
	time(&now);
	char buf[sizeof "0000-00-00_00-00-00z"];
	strftime(buf, sizeof buf, "%Y-%m-%d_%H-%M-%Sz", gmtime(&now));
	return std::string(buf);
}

//-------------------------------
//----------SETUP STUFF----------
//-------------------------------

void initTextures() {
	glGenTextures(1, &displayImage);
	glBindTexture(GL_TEXTURE_2D, displayImage);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_BGRA, GL_UNSIGNED_BYTE, NULL);
}

void initVAO(void) {
	GLfloat vertices[] = {
		-1.0f, -1.0f,
		1.0f, -1.0f,
		1.0f,  1.0f,
		-1.0f,  1.0f,
	};

	GLfloat texcoords[] = {
		1.0f, 1.0f,
		0.0f, 1.0f,
		0.0f, 0.0f,
		1.0f, 0.0f
	};

	GLushort indices[] = { 0, 1, 3, 3, 1, 2 };

	GLuint vertexBufferObjID[3];
	glGenBuffers(3, vertexBufferObjID);

	glBindBuffer(GL_ARRAY_BUFFER, vertexBufferObjID[0]);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
	glVertexAttribPointer((GLuint)positionLocation, 2, GL_FLOAT, GL_FALSE, 0, 0);
	glEnableVertexAttribArray(positionLocation);

	glBindBuffer(GL_ARRAY_BUFFER, vertexBufferObjID[1]);
	glBufferData(GL_ARRAY_BUFFER, sizeof(texcoords), texcoords, GL_STATIC_DRAW);
	glVertexAttribPointer((GLuint)texcoordsLocation, 2, GL_FLOAT, GL_FALSE, 0, 0);
	glEnableVertexAttribArray(texcoordsLocation);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vertexBufferObjID[2]);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
}

GLuint initShader() {
	const char* attribLocations[] = { "Position", "Texcoords" };
	GLuint program = glslUtility::createDefaultProgram(attribLocations, 2);
	GLint location;

	//glUseProgram(program);
	if ((location = glGetUniformLocation(program, "u_image")) != -1) {
		glUniform1i(location, 0);
	}

	return program;
}

void deletePBO(GLuint* pbo) {
	if (pbo) {
		// unregister this buffer object with CUDA
		cudaGLUnregisterBufferObject(*pbo);

		glBindBuffer(GL_ARRAY_BUFFER, *pbo);
		glDeleteBuffers(1, pbo);

		*pbo = (GLuint)NULL;
	}
}

void deleteTexture(GLuint* tex) {
	glDeleteTextures(1, tex);
	*tex = (GLuint)NULL;
}

void cleanupCuda() {
	if (pbo) {
		deletePBO(&pbo);
	}
	if (displayImage) {
		deleteTexture(&displayImage);
	}
}

void initCuda() {
	cudaGLSetGLDevice(0);
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("CUDA error: %s\n", cudaGetErrorString(err));
		exit(-1);
	}

	// Clean up on program exit
	atexit(cleanupCuda);
}

void initPBO() {
	// set up vertex data parameter
	int num_texels = width * height;
	int num_values = num_texels * 4;
	int size_tex_data = sizeof(GLubyte) * num_values;

	// Generate a buffer ID called a PBO (Pixel Buffer Object)
	glGenBuffers(1, &pbo);

	// Make this the current UNPACK buffer (OpenGL is state-based)
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);

	// Allocate data for the buffer. 4-channel 8-bit image
	glBufferData(GL_PIXEL_UNPACK_BUFFER, size_tex_data, NULL, GL_DYNAMIC_COPY);
	cudaGLRegisterBufferObject(pbo);

}

void errorCallback(int error, const char* description) {
	fprintf(stderr, "%s\n", description);
}

bool init() {
	glfwSetErrorCallback(errorCallback);

	if (!glfwInit()) {
		exit(EXIT_FAILURE);
	}

	window = glfwCreateWindow(width, height, "CIS 565 Path Tracer", NULL, NULL);
	if (!window) {
		glfwTerminate();
		return false;
	}
	glfwMakeContextCurrent(window);
	glfwSetKeyCallback(window, keyCallback);
	glfwSetCursorPosCallback(window, mousePositionCallback);
	glfwSetMouseButtonCallback(window, mouseButtonCallback);

	// Set up GL context
	glewExperimental = GL_TRUE;
	if (glewInit() != GLEW_OK) {
		return false;
	}
	printf("Opengl Version:%s\n", glGetString(GL_VERSION));
	//Set up ImGui

	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	io = &ImGui::GetIO(); (void)io;
	ImGui::StyleColorsLight();
	ImGui_ImplGlfw_InitForOpenGL(window, true);
	ImGui_ImplOpenGL3_Init("#version 120");

	// Initialize other stuff
	initVAO();
	initTextures();
	initCuda();
	initPBO();
	GLuint passthroughProgram = initShader();

	glUseProgram(passthroughProgram);
	glActiveTexture(GL_TEXTURE0);

	return true;
}

void InitImguiData(GuiDataContainer* guiData)
{
	imguiData = guiData;
}


// LOOK: Un-Comment to check ImGui Usage
void RenderImGui()
{
	if (!display_ui) return;
	mouseOverImGuiWinow = io->WantCaptureMouse;

	ImGui_ImplOpenGL3_NewFrame();
	ImGui_ImplGlfw_NewFrame();
	ImGui::NewFrame();

	bool show_demo_window = true;
	bool show_another_window = false;
	ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);
	static float f = 0.0f;
	static int counter = 0;
	/*
	*	Analytics
	*/
	ImGui::Begin("Path Tracer Analytics");                  // Create a window called "Hello, world!" and append into it.
	ImGui::Text("Traced Depth %d", imguiData->TracedDepth);
	ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
	ImGui::Text("Num triangles: %d", scene->triangles.size());
	ImGui::Text("Num bvh nodes: %d", scene->bvhTreeSize);
	ImGui::End();
	/*
	*	Camera Settings
	*/
	ImGui::Begin("Camera Settings");
	if (ImGui::SliderFloat("Theta", &theta, 0.0f, PI))
		camchanged = true;
	if(ImGui::SliderFloat("Phi", &phi, 0.0f, TWO_PI))
		camchanged = true;
	if(ImGui::SliderFloat("Pos x", &renderState->camera.position.x, -10.0f, 10.0f))
		camchanged = true;
	if (ImGui::SliderFloat("Pos y", &renderState->camera.position.y, -10.0f, 10.0f))
		camchanged = true;
	if (ImGui::SliderFloat("Pos z", &renderState->camera.position.z, -10.0f, 10.0f))
		camchanged = true;
	if (ImGui::SliderFloat("Lens Radius", &renderState->camera.lensRadius, 0.0f, 0.15f))
		camchanged = true;
	if (ImGui::SliderFloat("Focal Length", &renderState->camera.focalLength, 0.1f, 20.0f))
		camchanged = true;
	ImGui::End();
	/*
	*	Visualization Settings
	*/
	ImGui::Begin("Visualization Settings");
	static int selected_item = 0;
	const char* items[] = { "Render","GBuffer"};
	if (ImGui::BeginCombo("Type", items[selected_item]))
	{
		for (int n = 0; n < IM_ARRAYSIZE(items); n++)
		{
			bool is_selected = (selected_item == n);
			if (ImGui::Selectable(items[n], is_selected))
			{
				selected_item = n;
				switch (n)
				{
				case 0:
					visType = render;
					break;
				case 1:
					visType = gbuffer;
					break;
				}
				camchanged = true;
			}

			if (is_selected)
			{
				ImGui::SetItemDefaultFocus();
			}
		}
		
		ImGui::EndCombo();
	}
	if (visType == gbuffer)
	{
		static int selected_item = 0;
		const char* items[] = { "Time of flight","Position","Normal","Velocity", "Albedo", "Emission", "Depth"};
		if (ImGui::BeginCombo("GBuffer", items[selected_item]))
		{
			for (int n = 0; n < IM_ARRAYSIZE(items); n++)
			{
				bool is_selected = (selected_item == n);
				if (ImGui::Selectable(items[n], is_selected))
				{
					selected_item = n;
					switch (n)
					{
					case 0:
						gbufVisType = gTime;
						break;
					case 1:
						gbufVisType = gPosition;
						break;
					case 2:
						gbufVisType = gNormal;
						break;
					case 3:
						gbufVisType = gVelocity;
						break;
					case 4:
						gbufVisType = gAlbedo;
						break;
					case 5:
						gbufVisType = gEmission;
						break;
					case 6:
						gbufVisType = gDepth;
						break;
					}
				}

				if (is_selected)
				{
					ImGui::SetItemDefaultFocus();
				}
			}
			ImGui::EndCombo();
		}
	}
	ImGui::End();

	ImGui::Begin("Denoise Settings");
	
	if (ImGui::Checkbox("Denoise", &denoise_enabled))
	{
		camchanged = true;
		//denoiseClear();
	}
	ImGui::Checkbox("Animate Camera", &animate_camera);
	if (denoise_enabled)
	{
		static int selected_item = 0;
		const char* items[] = { "EAW","SVGF" };
		if (ImGui::BeginCombo("GBuffer", items[selected_item]))
		{
			for (int n = 0; n < IM_ARRAYSIZE(items); n++)
			{
				bool is_selected = (selected_item == n);
				if (ImGui::Selectable(items[n], is_selected))
				{
					selected_item = n;
					switch (n)
					{
					case 0:
						denoiserType = EAW;
						break;
					case 1:
						denoiserType = SVGF;
						break;
					}
					camchanged = true;
				}

				if (is_selected)
				{
					ImGui::SetItemDefaultFocus();
				}
			}
			ImGui::EndCombo();
		}
	}
	ImGui::End();

	ImGui::Render();
	ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

}

bool MouseOverImGuiWindow()
{
	return mouseOverImGuiWinow;
}

void mainLoop() {
	while (!glfwWindowShouldClose(window)) {
		
		glfwPollEvents();
		auto currTime = std::chrono::high_resolution_clock::now();
		float deltaT = std::chrono::duration_cast<std::chrono::duration<float>>(currTime - lastTime).count();
		lastTime = currTime;
		
		if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
		{
			camchanged = true;
			renderState->camera.position += renderState->camera.view * deltaT * CAM_SPEED;
		}
		if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
		{
			camchanged = true;
			renderState->camera.position -= renderState->camera.view * deltaT * CAM_SPEED;
		}
		if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
		{
			camchanged = true;
			renderState->camera.position -= renderState->camera.right * deltaT * CAM_SPEED;
		}
		if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
		{
			camchanged = true;
			renderState->camera.position += renderState->camera.right * deltaT * CAM_SPEED;
		}
		glm::vec3 center = renderState->camera.position;
		glm::vec3 offsetCenter = center;
		if (animate_camera)
		{
			camAnimTime += deltaT;
			offsetCenter += cos(camAnimTime * 2.0f) * camMoveRadius * glm::normalize(renderState->camera.right);
			offsetCenter += sin(camAnimTime * 2.0f) * camMoveRadius * glm::normalize(renderState->camera.view);
			renderState->camera.position = offsetCenter;
			camchanged = true;
		}
			
		runCuda();

		if (animate_camera)
		{
			renderState->camera.position = center;
		}

		string title = "CIS565 Path Tracer | " + utilityCore::convertIntToString(iteration) + " Iterations";
		glfwSetWindowTitle(window, title.c_str());
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
		glBindTexture(GL_TEXTURE_2D, displayImage);
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
		glClear(GL_COLOR_BUFFER_BIT);

		// Binding GL_PIXEL_UNPACK_BUFFER back to default
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

		// VAO, shader program, and texture already bound
		glDrawElements(GL_TRIANGLES, 6,  GL_UNSIGNED_SHORT, 0);

		Camera& cam = renderState->camera;
		if (animate_camera)
			renderState->lastCamInfo.position = offsetCenter;
		else
			renderState->lastCamInfo.position = cam.position;
		renderState->lastCamInfo.view = cam.view;
		renderState->lastCamInfo.right = cam.right;
		renderState->lastCamInfo.up = cam.up;
		//renderState->prevViewProj = glm::perspective(cam.fovAngle * (PI / 180), (float)cam.resolution.x / cam.resolution.y, 0.01f, 3000.0f) * glm::lookAt(cam.position, cam.position + cam.view, cam.up);

		// Render ImGui Stuff
		RenderImGui();

		glfwSwapBuffers(window);

		
	}

	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();

	glfwDestroyWindow(window);
	glfwTerminate();
}
