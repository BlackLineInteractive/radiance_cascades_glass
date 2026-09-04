#include "gl_loader.h"
#include <iostream>

PFNGLCREATESHADERPROC glCreateShader = nullptr;
PFNGLSHADERSOURCEPROC glShaderSource = nullptr;
PFNGLCOMPILESHADERPROC glCompileShader = nullptr;
PFNGLGETSHADERIVPROC glGetShaderiv = nullptr;
PFNGLGETSHADERINFOLOGPROC glGetShaderInfoLog = nullptr;
PFNGLCREATEPROGRAMPROC glCreateProgram = nullptr;
PFNGLATTACHSHADERPROC glAttachShader = nullptr;
PFNGLLINKPROGRAMPROC glLinkProgram = nullptr;
PFNGLGETPROGRAMIVPROC glGetProgramiv = nullptr;
PFNGLGETPROGRAMINFOLOGPROC glGetProgramInfoLog = nullptr;
PFNGLUSEPROGRAMPROC glUseProgram = nullptr;
PFNGLDELETESHADERPROC glDeleteShader = nullptr;
PFNGLDELETEPROGRAMPROC glDeleteProgram = nullptr;

PFNGLGENBUFFERSPROC glGenBuffers = nullptr;
PFNGLBINDBUFFERPROC glBindBuffer = nullptr;
PFNGLBUFFERDATAPROC glBufferData = nullptr;
PFNGLBUFFERSUBDATAPROC glBufferSubData = nullptr;
PFNGLBINDBUFFERBASEPROC glBindBufferBase = nullptr;
PFNGLDELETEBUFFERSPROC glDeleteBuffers = nullptr;

PFNGLGENVERTEXARRAYSPROC glGenVertexArrays = nullptr;
PFNGLBINDVERTEXARRAYPROC glBindVertexArray = nullptr;
PFNGLDELETEVERTEXARRAYSPROC glDeleteVertexArrays = nullptr;
PFNGLDRAWARRAYSPROC glDrawArrays = nullptr;

PFNGLGENTEXTURESPROC glGenTextures = nullptr;
PFNGLBINDTEXTUREPROC glBindTexture = nullptr;
PFNGLTEXSTORAGE2DPROC glTexStorage2D = nullptr;
PFNGLTEXSTORAGE3DPROC glTexStorage3D = nullptr;
PFNGLTEXIMAGE2DPROC glTexImage2D = nullptr;
PFNGLTEXPARAMETERIPROC glTexParameteri = nullptr;
PFNGLACTIVETEXTUREPROC glActiveTexture = nullptr;
PFNGLBINDIMAGETEXTUREPROC glBindImageTexture = nullptr;
PFNGLGETTEXIMAGEPROC glGetTexImage = nullptr;
PFNGLDELETETEXTURESPROC glDeleteTextures = nullptr;

PFNGLDISPATCHCOMPUTEPROC glDispatchCompute = nullptr;
PFNGLMEMORYBARRIERPROC glMemoryBarrier = nullptr;

PFNGLGETSTRINGPROC glGetString = nullptr;
PFNGLVIEWPORTPROC glViewport = nullptr;
PFNGLFINISHPROC glFinish = nullptr;

#define LOAD_PROC(type, name) \
    name = (type)glfwGetProcAddress(#name); \
    if (!name) { \
        std::cerr << "[GL Loader] Warning: Failed to load function: " << #name << "\n"; \
        success = false; \
    }

bool initGLLoader() {
    bool success = true;

    LOAD_PROC(PFNGLCREATESHADERPROC, glCreateShader);
    LOAD_PROC(PFNGLSHADERSOURCEPROC, glShaderSource);
    LOAD_PROC(PFNGLCOMPILESHADERPROC, glCompileShader);
    LOAD_PROC(PFNGLGETSHADERIVPROC, glGetShaderiv);
    LOAD_PROC(PFNGLGETSHADERINFOLOGPROC, glGetShaderInfoLog);
    LOAD_PROC(PFNGLCREATEPROGRAMPROC, glCreateProgram);
    LOAD_PROC(PFNGLATTACHSHADERPROC, glAttachShader);
    LOAD_PROC(PFNGLLINKPROGRAMPROC, glLinkProgram);
    LOAD_PROC(PFNGLGETPROGRAMIVPROC, glGetProgramiv);
    LOAD_PROC(PFNGLGETPROGRAMINFOLOGPROC, glGetProgramInfoLog);
    LOAD_PROC(PFNGLUSEPROGRAMPROC, glUseProgram);
    LOAD_PROC(PFNGLDELETESHADERPROC, glDeleteShader);
    LOAD_PROC(PFNGLDELETEPROGRAMPROC, glDeleteProgram);

    LOAD_PROC(PFNGLGENBUFFERSPROC, glGenBuffers);
    LOAD_PROC(PFNGLBINDBUFFERPROC, glBindBuffer);
    LOAD_PROC(PFNGLBUFFERDATAPROC, glBufferData);
    LOAD_PROC(PFNGLBUFFERSUBDATAPROC, glBufferSubData);
    LOAD_PROC(PFNGLBINDBUFFERBASEPROC, glBindBufferBase);
    LOAD_PROC(PFNGLDELETEBUFFERSPROC, glDeleteBuffers);

    LOAD_PROC(PFNGLGENVERTEXARRAYSPROC, glGenVertexArrays);
    LOAD_PROC(PFNGLBINDVERTEXARRAYPROC, glBindVertexArray);
    LOAD_PROC(PFNGLDELETEVERTEXARRAYSPROC, glDeleteVertexArrays);
    LOAD_PROC(PFNGLDRAWARRAYSPROC, glDrawArrays);

    LOAD_PROC(PFNGLGENTEXTURESPROC, glGenTextures);
    LOAD_PROC(PFNGLBINDTEXTUREPROC, glBindTexture);
    LOAD_PROC(PFNGLTEXSTORAGE2DPROC, glTexStorage2D);
    LOAD_PROC(PFNGLTEXSTORAGE3DPROC, glTexStorage3D);
    LOAD_PROC(PFNGLTEXIMAGE2DPROC, glTexImage2D);
    LOAD_PROC(PFNGLTEXPARAMETERIPROC, glTexParameteri);
    LOAD_PROC(PFNGLACTIVETEXTUREPROC, glActiveTexture);
    LOAD_PROC(PFNGLBINDIMAGETEXTUREPROC, glBindImageTexture);
    LOAD_PROC(PFNGLGETTEXIMAGEPROC, glGetTexImage);
    LOAD_PROC(PFNGLDELETETEXTURESPROC, glDeleteTextures);

    LOAD_PROC(PFNGLDISPATCHCOMPUTEPROC, glDispatchCompute);
    LOAD_PROC(PFNGLMEMORYBARRIERPROC, glMemoryBarrier);

    LOAD_PROC(PFNGLGETSTRINGPROC, glGetString);
    LOAD_PROC(PFNGLVIEWPORTPROC, glViewport);
    LOAD_PROC(PFNGLFINISHPROC, glFinish);

    return success;
}
