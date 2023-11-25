#ifndef EGLU_HPP
#define EGLU_HPP
#include <EGL/egl.h>
#include <rastmath.hpp>
#ifdef EGLU_IMPLEMENTATION
#define GLAD_GL_IMPLEMENTATION
#define PAR_SHAPES_IMPLEMENTATION
#endif
#include <par_shapes.h>
#include <vector>
#include <glad/gl.h>
#include <iostream>
struct shader {
    GLuint shaderProgram;
    shader(const char *vertexShaderSource, const char *fragmentShaderSource) {
        GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
        glShaderSource(vertexShader, 1, &vertexShaderSource, nullptr);
        glCompileShader(vertexShader);
        // Check for shader compilation errors
        GLint success;
        glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
        if (!success) {
            char infoLog[512];
            glGetShaderInfoLog(vertexShader, sizeof(infoLog), nullptr, infoLog);
            std::cerr << "Vertex shader compilation failed: " << infoLog << std::endl;
        }
        // Create fragment shader
        GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(fragmentShader, 1, &fragmentShaderSource, nullptr);
        glCompileShader(fragmentShader);

        // Check for shader compilation errors
        glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
        if (!success) {
            char infoLog[512] = {0};
            glGetShaderInfoLog(fragmentShader, sizeof(infoLog), nullptr, infoLog);
            std::cerr << "Fragment shader compilation failed: " << infoLog << std::endl;
        }
        
        // Create shader program
        shaderProgram = glCreateProgram();
        glAttachShader(shaderProgram, vertexShader);
        glAttachShader(shaderProgram, fragmentShader);
        glLinkProgram(shaderProgram);

        // Check for linking errors
        glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
        if (!success) {
            char infoLog[512] = {0};
            glGetProgramInfoLog(shaderProgram, sizeof(infoLog), nullptr, infoLog);
            std::cerr << "Shader program linking failed: " << infoLog << std::endl;
        }

        // Delete the shaders as they're linked into the program now and no longer needed
        glDeleteShader(vertexShader);
        glDeleteShader(fragmentShader);
    }
    void setMat4(const char* name, const rm::Matrix<float, 4, 4>& matrix) const {
        float vals [16];
        for(int i = 0;i < 4;i++){
            for(int j = 0;j < 4;j++){
                vals[j + i * 4] = matrix(j, i);
            }
        }
        glUseProgram(shaderProgram);
        GLuint uniformLocation = glGetUniformLocation(shaderProgram, name);
        glUniformMatrix4fv(uniformLocation, 1, GL_FALSE, vals);
    }
};
constexpr EGLint configAttribs[] = {
    EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
    EGL_BLUE_SIZE, 8,
    EGL_GREEN_SIZE, 8,
    EGL_RED_SIZE, 8,
    EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
    EGL_SAMPLE_BUFFERS, 1,        // Enable multisampling
    EGL_SAMPLES, 16,              // Number of samples (16x multisampling)
    EGL_NONE
};
void load_context(EGLint width, EGLint height);

struct Mesh {
    int vertexCount;        // Number of vertices stored in arrays
    int triangleCount;      // Number of triangles stored (indexed or not)

    // Vertex attributes data
    float *vertices;        // Vertex position (XYZ - 3 components per vertex) (shader-location = 0)
    float *texcoords;       // Vertex texture coordinates (UV - 2 components per vertex) (shader-location = 1)
    //float *texcoords2;      // Vertex texture second coordinates (UV - 2 components per vertex) (shader-location = 5)
    //float *normals;         // Vertex normals (XYZ - 3 components per vertex) (shader-location = 2)
    //float *tangents;        // Vertex tangents (XYZW - 4 components per vertex) (shader-location = 4)
    unsigned char *colors;      // Vertex colors (RGBA - 4 components per vertex) (shader-location = 3)
    unsigned short *indices;    // Vertex indices (in case vertex data comes indexed)
};
Mesh GenMeshSphere(float radius, int rings, int slices);
struct vaovbo{
    GLuint vao;
    std::vector<GLuint> vbo;
    void destroy() {
        glDeleteVertexArrays(1, &vao);
        for (auto& vboID : vbo) {
            glDeleteBuffers(1, &vboID);
        }
        vbo.clear();
    }
};
vaovbo to_vao(const Mesh& mesh, float* offsets);
#ifdef EGLU_IMPLEMENTATION
Mesh GenMeshSphere(float radius, int rings, int slices){
    Mesh mesh{0, 0, nullptr, nullptr, nullptr, nullptr};

    if ((rings >= 3) && (slices >= 3)){
        par_shapes_mesh *sphere = par_shapes_create_parametric_sphere(slices, rings);
        par_shapes_scale(sphere, radius, radius, radius);
        // NOTE: Soft normals are computed internally

        mesh.vertices  = (float*)std::malloc(sphere->ntriangles * 3 * 3 * sizeof(float));
        mesh.texcoords = (float*)std::malloc(sphere->ntriangles * 3 * 2 * sizeof(float));
        //mesh.normals =  (float *) std::malloc(sphere->ntriangles*3*3*sizeof(float));

        mesh.vertexCount = sphere->ntriangles * 3;
        mesh.triangleCount = sphere->ntriangles;

        for (int k = 0; k < mesh.vertexCount; k++){
            mesh.vertices[k * 3 + 0] = sphere->points[sphere->triangles[k] * 3 + 0];
            mesh.vertices[k * 3 + 1] = sphere->points[sphere->triangles[k] * 3 + 1];
            mesh.vertices[k * 3 + 2] = sphere->points[sphere->triangles[k] * 3 + 2];

            //mesh.normals[k*3] = sphere->normals[sphere->triangles[k]*3];
            //mesh.normals[k*3 + 1] = sphere->normals[sphere->triangles[k]*3 + 1];
            //mesh.normals[k*3 + 2] = sphere->normals[sphere->triangles[k]*3 + 2];

            mesh.texcoords[k * 2 + 0] = sphere->tcoords[sphere->triangles[k] * 2 + 0];
            mesh.texcoords[k * 2 + 1] = sphere->tcoords[sphere->triangles[k] * 2 + 1];
        }

        par_shapes_free_mesh(sphere);
    }

    return mesh;
}

vaovbo to_vao(const Mesh& mesh, float* offsets, size_t count) {
    // Assuming you have an OpenGL context and necessary bindings

    // Create and bind a Vertex Array Object (VAO)
    unsigned int vao;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    // Create and bind a Vertex Buffer Object (VBO) for vertices
    unsigned int vertexVBO;
    glGenBuffers(1, &vertexVBO);
    glBindBuffer(GL_ARRAY_BUFFER, vertexVBO);
    glBufferData(GL_ARRAY_BUFFER, mesh.vertexCount * 3 * sizeof(float), mesh.vertices, GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // Create and bind a Vertex Buffer Object (VBO) for texture coordinates
    unsigned int texcoordVBO;
    glGenBuffers(1, &texcoordVBO);
    glBindBuffer(GL_ARRAY_BUFFER, texcoordVBO);
    glBufferData(GL_ARRAY_BUFFER, mesh.vertexCount * 2 * sizeof(float), mesh.texcoords, GL_DYNAMIC_DRAW);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(1);

    // Create and bind a Vertex Buffer Object (VBO) for colors
    //unsigned int colorVBO;
    //glGenBuffers(1, &colorVBO);
    //glBindBuffer(GL_ARRAY_BUFFER, colorVBO);
    //glBufferData(GL_ARRAY_BUFFER, mesh.vertexCount * 4 * sizeof(unsigned char), mesh.colors, GL_DYNAMIC_DRAW);
    //glVertexAttribPointer(2, 4, GL_UNSIGNED_BYTE, GL_TRUE, 4 * sizeof(unsigned char), (void*)0);
    //glEnableVertexAttribArray(2);

    // Create and bind a Vertex Buffer Object (VBO) for offsets (instancing)
    unsigned int offsetVBO;
    glGenBuffers(1, &offsetVBO);
    glBindBuffer(GL_ARRAY_BUFFER, offsetVBO);
    glBufferData(GL_ARRAY_BUFFER, count * 3 * sizeof(float), offsets, GL_DYNAMIC_DRAW);
    glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(3);
    glVertexAttribDivisor(3, 1); // Set divisor for instancing

    // Unbind VAO and buffers
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
    return vaovbo{vao, std::vector<GLuint>{vertexVBO, texcoordVBO, offsetVBO}};
}
void load_context(EGLint width, EGLint height){
    const EGLint pbufferAttribs[] = {
        EGL_WIDTH,
        width,
        EGL_HEIGHT,
        height,
        EGL_NONE,
    };
    EGLDisplay eglDpy = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    EGLint major, minor;
    eglInitialize(eglDpy, &major, &minor);
    // 2. Select an appropriate configuration
    EGLint numConfigs;
    EGLConfig eglCfg;
    
    eglChooseConfig(eglDpy, configAttribs, &eglCfg, 1, &numConfigs);
    // 3. Create a surface
    EGLSurface eglSurf = eglCreatePbufferSurface(eglDpy, eglCfg, pbufferAttribs);
    // 4. Bind the API
    eglBindAPI(EGL_OPENGL_API);
    EGLContext eglCtx = eglCreateContext(eglDpy, eglCfg, EGL_NO_CONTEXT, 0);
    eglMakeCurrent(eglDpy, eglSurf, eglSurf, eglCtx);
    gladLoadGL(eglGetProcAddress);
}
#endif

#endif