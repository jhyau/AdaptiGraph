//
// Created by yunzhu on 6/10/18.
//


#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <cstring>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
namespace py = pybind11;

#include "include/NvFlex.h"
#include "include/NvFlexExt.h"
#include "include/NvFlexDevice.h"

#include "core/maths.h"
#include "core/types.h"
#include "core/platform.h"
#include "core/mesh.h"
#include "core/voxelize.h"
#include "core/sdf.h"
#include "core/pfm.h"
#include "core/tga.h"
#include "core/perlin.h"
#include "core/convex.h"
#include "core/cloth.h"

#include "external/SDL2-2.0.4/include/SDL.h"
#include "external/glad/include/glad/glad.h" //add

#include "bindings/opengl/shader.h" //add
#include "shaders.h"
#include "imgui.h"
#include "shadersDemoContext.h"


SDL_Window* g_window;           // window handle
unsigned int g_windowId;        // window id

#define SDL_CONTROLLER_BUTTON_LEFT_TRIGGER (SDL_CONTROLLER_BUTTON_MAX + 1)
#define SDL_CONTROLLER_BUTTON_RIGHT_TRIGGER (SDL_CONTROLLER_BUTTON_MAX + 2)

int GetKeyFromGameControllerButton(SDL_GameControllerButton button) {
    switch (button) {
        case SDL_CONTROLLER_BUTTON_DPAD_UP:         {    return SDLK_q;        }    // -- camera translate up
        case SDL_CONTROLLER_BUTTON_DPAD_DOWN:       {    return SDLK_z;        }    // -- camera translate down
        case SDL_CONTROLLER_BUTTON_DPAD_LEFT:       {    return SDLK_h;        }    // -- hide GUI
        case SDL_CONTROLLER_BUTTON_DPAD_RIGHT:      {    return -1;            }    // -- unassigned
        case SDL_CONTROLLER_BUTTON_START:           {    return SDLK_RETURN;    }   // -- start selected scene
        case SDL_CONTROLLER_BUTTON_BACK:            {    return SDLK_ESCAPE;    }   // -- quit
        case SDL_CONTROLLER_BUTTON_LEFTSHOULDER:    {    return SDLK_UP;        }   // -- select prev scene
        case SDL_CONTROLLER_BUTTON_RIGHTSHOULDER:   {    return SDLK_DOWN;    }     // -- select next scene
        case SDL_CONTROLLER_BUTTON_A:               {    return SDLK_g;        }    // -- toggle gravity
        case SDL_CONTROLLER_BUTTON_B:               {    return SDLK_p;        }    // -- pause
        case SDL_CONTROLLER_BUTTON_X:               {    return SDLK_r;        }    // -- reset
        case SDL_CONTROLLER_BUTTON_Y:               {    return SDLK_o;        }    // -- step sim
        case SDL_CONTROLLER_BUTTON_RIGHT_TRIGGER:   {    return SDLK_SPACE;    }    // -- emit particles
        default:                                    {    return -1;            }    // -- nop
    };
};

//
// Gamepad thresholds taken from XINPUT API
//
#define XINPUT_GAMEPAD_LEFT_THUMB_DEADZONE  7849
#define XINPUT_GAMEPAD_RIGHT_THUMB_DEADZONE 8689
#define XINPUT_GAMEPAD_TRIGGER_THRESHOLD    30

int deadzones[3] = {
   XINPUT_GAMEPAD_LEFT_THUMB_DEADZONE,
   XINPUT_GAMEPAD_RIGHT_THUMB_DEADZONE,
   XINPUT_GAMEPAD_TRIGGER_THRESHOLD };

inline float joyAxisFilter(int value, int stick) {
    //clamp values in deadzone to zero, and remap rest of range so that it linearly rises in value from edge of deadzone toward max value.
    if (value < -deadzones[stick])
        return (value + deadzones[stick]) / (32768.0f - deadzones[stick]);
    else if (value > deadzones[stick])
        return (value - deadzones[stick]) / (32768.0f - deadzones[stick]);
    else
        return 0.0f;
}

SDL_GameController* g_gamecontroller = nullptr;

using namespace std;

int g_screenWidth = 720;
int g_screenHeight = 720;
int g_msaaSamples = 0;

int g_numSubsteps;

// a setting of -1 means Flex will use the device specified in the NVIDIA control panel
int g_device = -1;
char g_deviceName[256];
bool g_vsync = true;

// these two are migrated from Flex 2.0
bool g_headless = true; //add

bool g_benchmark = false;
bool g_extensions = true;
bool g_teamCity = false;
bool g_interop = true;
bool g_d3d12 = false;
bool g_useAsyncCompute = true;
bool g_increaseGfxLoadForAsyncComputeTesting = false;
int g_graphics = 0; // 0=ogl, 1=DX11, 2=DX12

FluidRenderer* g_fluidRenderer;
FluidRenderBuffers* g_fluidRenderBuffers;
DiffuseRenderBuffers* g_diffuseRenderBuffers;

NvFlexSolver* g_solver;
NvFlexSolverDesc g_solverDesc;
NvFlexLibrary* g_flexLib;
NvFlexParams g_params;
NvFlexTimers g_timers;
int g_numDetailTimers;
NvFlexDetailTimer* g_detailTimers;

int g_maxDiffuseParticles;
int g_maxNeighborsPerParticle;
int g_numExtraParticles;
int g_numExtraMultiplier = 1;
int g_maxContactsPerParticle;

// mesh used for deformable object rendering
Mesh* g_mesh;
vector<int> g_meshSkinIndices;
vector<float> g_meshSkinWeights;
vector<Point3> g_meshRestPositions;
const int g_numSkinWeights = 4;

// mapping of collision mesh to render mesh
std::map<NvFlexConvexMeshId, GpuMesh*> g_convexes;
std::map<NvFlexTriangleMeshId, GpuMesh*> g_meshes;
std::map<NvFlexDistanceFieldId, GpuMesh*> g_fields;

// flag to request collision shapes be updated
bool g_shapesChanged = false;

/* Note that this array of colors is altered by demo code, and is also read from global by graphics API impls */
// add 2000 for g_colors
Colour g_colors[2000] = {
    Colour(0.000f, 0.349f, 0.173f),
    Colour(0.875f, 0.782f, 0.051f),
    Colour(0.000f, 0.170f, 0.453f),
    Colour(0.673f, 0.111f, 0.000f),
    Colour(0.612f, 0.194f, 0.394f),
    Colour(0.0f, 0.5f, 1.0f),
    Colour(0.797f, 0.354f, 0.000f),
    Colour(0.092f, 0.465f, 0.820f)
};

struct SimBuffers {
    NvFlexVector<Vec4> positions;
    NvFlexVector<Vec4> restPositions;
    NvFlexVector<Vec3> velocities;
    NvFlexVector<int> phases;
    NvFlexVector<float> densities;
    NvFlexVector<Vec4> anisotropy1;
    NvFlexVector<Vec4> anisotropy2;
    NvFlexVector<Vec4> anisotropy3;
    NvFlexVector<Vec4> normals;
    NvFlexVector<Vec4> smoothPositions;
    NvFlexVector<Vec4> diffusePositions;
    NvFlexVector<Vec4> diffuseVelocities;
    NvFlexVector<int> diffuseCount;

    NvFlexVector<int> activeIndices;

    NvFlexVector<int> clusters;

    // convexes
    NvFlexVector<NvFlexCollisionGeometry> shapeGeometry;
    NvFlexVector<Vec4> shapePositions;
    NvFlexVector<Quat> shapeRotations;
    NvFlexVector<Vec4> shapePrevPositions;
    NvFlexVector<Quat> shapePrevRotations;
    NvFlexVector<int> shapeFlags;

    // rigids
    NvFlexVector<int> rigidOffsets;
    NvFlexVector<int> rigidIndices;
    NvFlexVector<int> rigidMeshSize;
    NvFlexVector<float> rigidCoefficients;
    NvFlexVector<float> rigidPlasticThresholds;
    NvFlexVector<float> rigidPlasticCreeps;
    NvFlexVector<Quat> rigidRotations;
    NvFlexVector<Vec3> rigidTranslations;
    NvFlexVector<Vec3> rigidLocalPositions;
    NvFlexVector<Vec4> rigidLocalNormals;

    // inflatables
    NvFlexVector<int> inflatableTriOffsets;
    NvFlexVector<int> inflatableTriCounts;
    NvFlexVector<float> inflatableVolumes;
    NvFlexVector<float> inflatableCoefficients;
    NvFlexVector<float> inflatablePressures;

    // springs
    NvFlexVector<int> springIndices;
    NvFlexVector<float> springLengths;
    NvFlexVector<float> springStiffness;

    NvFlexVector<int> triangles;
    NvFlexVector<Vec3> triangleNormals;
    NvFlexVector<Vec3> uvs;

    // Map from particle index to object instance index
    NvFlexVector<int> particle2objInstance;

    SimBuffers(NvFlexLibrary* l) :
        positions(l), restPositions(l), velocities(l), phases(l), densities(l),
        anisotropy1(l), anisotropy2(l), anisotropy3(l), normals(l), smoothPositions(l),
        diffusePositions(l), diffuseVelocities(l), diffuseCount(l), activeIndices(l), clusters(l),
        shapeGeometry(l), shapePositions(l), shapeRotations(l), shapePrevPositions(l),
        shapePrevRotations(l),    shapeFlags(l), rigidOffsets(l), rigidIndices(l), rigidMeshSize(l),
        rigidCoefficients(l), rigidPlasticThresholds(l), rigidPlasticCreeps(l), rigidRotations(l), rigidTranslations(l),
        rigidLocalPositions(l), rigidLocalNormals(l), inflatableTriOffsets(l),
        inflatableTriCounts(l), inflatableVolumes(l), inflatableCoefficients(l),
        inflatablePressures(l), springIndices(l), springLengths(l),
        springStiffness(l), triangles(l), triangleNormals(l), uvs(l), particle2objInstance(l)
    {}
};

SimBuffers* g_buffers;

void MapBuffers(SimBuffers* buffers) {
    buffers->positions.map();
    buffers->restPositions.map();
    buffers->velocities.map();
    buffers->phases.map();
    buffers->densities.map();
    buffers->anisotropy1.map();
    buffers->anisotropy2.map();
    buffers->anisotropy3.map();
    buffers->normals.map();
    buffers->diffusePositions.map();
    buffers->diffuseVelocities.map();
    buffers->diffuseCount.map();
    buffers->smoothPositions.map();
    buffers->activeIndices.map();
    
    buffers->clusters.map();

    // convexes
    buffers->shapeGeometry.map();
    buffers->shapePositions.map();
    buffers->shapeRotations.map();
    buffers->shapePrevPositions.map();
    buffers->shapePrevRotations.map();
    buffers->shapeFlags.map();

    buffers->rigidOffsets.map();
    buffers->rigidIndices.map();
    buffers->rigidMeshSize.map();
    buffers->rigidCoefficients.map();
    buffers->rigidPlasticThresholds.map();
    buffers->rigidPlasticCreeps.map();
    buffers->rigidRotations.map();
    buffers->rigidTranslations.map();
    buffers->rigidLocalPositions.map();
    buffers->rigidLocalNormals.map();

    buffers->springIndices.map();
    buffers->springLengths.map();
    buffers->springStiffness.map();

    // inflatables
    buffers->inflatableTriOffsets.map();
    buffers->inflatableTriCounts.map();
    buffers->inflatableVolumes.map();
    buffers->inflatableCoefficients.map();
    buffers->inflatablePressures.map();

    buffers->triangles.map();
    buffers->triangleNormals.map();
    buffers->uvs.map();

    // Map from particle index to object instance index
    buffers->particle2objInstance.map();
}

void UnmapBuffers(SimBuffers* buffers) {
    // particles
    buffers->positions.unmap();
    buffers->restPositions.unmap();
    buffers->velocities.unmap();
    buffers->phases.unmap();
    buffers->densities.unmap();
    buffers->anisotropy1.unmap();
    buffers->anisotropy2.unmap();
    buffers->anisotropy3.unmap();
    buffers->normals.unmap();
    buffers->diffusePositions.unmap();
    buffers->diffuseVelocities.unmap();
    buffers->diffuseCount.unmap();
    buffers->smoothPositions.unmap();
    buffers->activeIndices.unmap();

    buffers->clusters.unmap();

    // convexes
    buffers->shapeGeometry.unmap();
    buffers->shapePositions.unmap();
    buffers->shapeRotations.unmap();
    buffers->shapePrevPositions.unmap();
    buffers->shapePrevRotations.unmap();
    buffers->shapeFlags.unmap();

    // rigids
    buffers->rigidOffsets.unmap();
    buffers->rigidIndices.unmap();
    buffers->rigidMeshSize.unmap();
    buffers->rigidCoefficients.unmap();
    buffers->rigidPlasticThresholds.unmap();
    buffers->rigidPlasticCreeps.unmap();
    buffers->rigidRotations.unmap();
    buffers->rigidTranslations.unmap();
    buffers->rigidLocalPositions.unmap();
    buffers->rigidLocalNormals.unmap();

    // springs
    buffers->springIndices.unmap();
    buffers->springLengths.unmap();
    buffers->springStiffness.unmap();

    // inflatables
    buffers->inflatableTriOffsets.unmap();
    buffers->inflatableTriCounts.unmap();
    buffers->inflatableVolumes.unmap();
    buffers->inflatableCoefficients.unmap();
    buffers->inflatablePressures.unmap();

    // triangles
    buffers->triangles.unmap();
    buffers->triangleNormals.unmap();
    buffers->uvs.unmap();

    // Map from particle index to object instance index
    buffers->particle2objInstance.unmap();

}

SimBuffers* AllocBuffers(NvFlexLibrary* lib) {
    return new SimBuffers(lib);
}

void DestroyBuffers(SimBuffers* buffers) {
    // particles
    buffers->positions.destroy();
    buffers->restPositions.destroy();
    buffers->velocities.destroy();
    buffers->phases.destroy();
    buffers->densities.destroy();
    buffers->anisotropy1.destroy();
    buffers->anisotropy2.destroy();
    buffers->anisotropy3.destroy();
    buffers->normals.destroy();
    buffers->diffusePositions.destroy();
    buffers->diffuseVelocities.destroy();
    buffers->diffuseCount.destroy();
    buffers->smoothPositions.destroy();
    buffers->activeIndices.destroy();

    buffers->clusters.destroy();

    // convexes
    buffers->shapeGeometry.destroy();
    buffers->shapePositions.destroy();
    buffers->shapeRotations.destroy();
    buffers->shapePrevPositions.destroy();
    buffers->shapePrevRotations.destroy();
    buffers->shapeFlags.destroy();

    // rigids
    buffers->rigidOffsets.destroy();
    buffers->rigidIndices.destroy();
    buffers->rigidMeshSize.destroy();
    buffers->rigidCoefficients.destroy();
    buffers->rigidPlasticThresholds.destroy();
    buffers->rigidPlasticCreeps.destroy();
    buffers->rigidRotations.destroy();
    buffers->rigidTranslations.destroy();
    buffers->rigidLocalPositions.destroy();
    buffers->rigidLocalNormals.destroy();

    // springs
    buffers->springIndices.destroy();
    buffers->springLengths.destroy();
    buffers->springStiffness.destroy();

    // inflatables
    buffers->inflatableTriOffsets.destroy();
    buffers->inflatableTriCounts.destroy();
    buffers->inflatableVolumes.destroy();
    buffers->inflatableCoefficients.destroy();
    buffers->inflatablePressures.destroy();

    // triangles
    buffers->triangles.destroy();
    buffers->triangleNormals.destroy();
    buffers->uvs.destroy();

    // Map from particle index to object instance index
    buffers->particle2objInstance.destroy();

    delete buffers;
}

Vec3 g_camPos(6.0f, 8.0f, 18.0f);
Vec3 g_camAngle(0.0f, -DegToRad(20.0f), 0.0f);
Vec3 g_camVel(0.0f);
Vec3 g_camSmoothVel(0.0f);

float g_camSpeed;
float g_camNear;
float g_camFar;

Vec3 g_lightPos;
Vec3 g_lightDir = Normalize(Vec3(5.0f, 15.0f, 7.5f)); //add
Vec3 g_lightTarget;
float g_lightFov; //add

bool g_pause = false;
bool g_step = false;
bool g_capture = false;
bool g_showHelp = true;
bool g_tweakPanel = false;
bool g_fullscreen = false;
bool g_wireframe = false;
bool g_debug = false;

bool g_emit = false;
bool g_warmup = false;

float g_windTime = 0.0f;
float g_windFrequency = 0.1f;
float g_windStrength = 0.0f;

bool g_wavePool = false;
float g_waveTime = 0.0f;
float g_wavePlane;
float g_waveFrequency = 1.5f;
float g_waveAmplitude = 1.0f;
float g_waveFloorTilt = 0.0f;

Vec3 g_sceneLower;
Vec3 g_sceneUpper;

float g_blur;
float g_ior;
bool g_drawEllipsoids;
bool g_drawPoints;
bool g_drawMesh;
bool g_drawCloth;
float g_expandCloth;    // amount to expand cloth along normal (to account for particle radius)

bool g_drawOpaque;
int g_drawSprings;        // 0: no draw, 1: draw stretch 2: draw tether
bool g_drawBases = false;
bool g_drawContacts = false;
bool g_drawNormals = false;
bool g_drawDiffuse;
bool g_drawShapeGrid = false;
bool g_drawDensity = false;
bool g_drawRopes;
float g_pointScale;
float g_ropeScale;
float g_drawPlaneBias;    // move planes along their normal for rendering
float g_floorScaleSize; //add

float g_diffuseScale;
float g_diffuseMotionScale;
bool g_diffuseShadow;
float g_diffuseInscatter;
float g_diffuseOutscatter;

float g_dt = 1.0f / 60.0f;    // the time delta used for simulation
float g_realdt;               // the real world time delta between updates

float g_waitTime;       // the CPU time spent waiting for the GPU
float g_updateTime;     // the CPU time spent on Flex
float g_renderTime;     // the CPU time spent calling OpenGL to render the scene
                        // the above times don't include waiting for vsync
float g_simLatency;     // the time the GPU spent between the first and last NvFlexUpdateSolver() operation. Because some GPUs context switch, this can include graphics time.

int g_scene = 0;
int g_selectedScene = g_scene;
int g_levelScroll;          // offset for level selection scroll area
bool g_resetScene = false;  //if the user clicks the reset button or presses the reset key this is set to true;

int g_frame = 0;
int g_numSolidParticles = 0;

int g_mouseParticle = -1;
float g_mouseT = 0.0f;
Vec3 g_mousePos;
float g_mouseMass;
bool g_mousePicked = false;

// mouse
int g_lastx;
int g_lasty;
int g_lastb = -1;

bool g_profile = false;
bool g_outputAllFrameTimes = false;
bool g_asyncComputeBenchmark = false;

ShadowMap* g_shadowMap;

Vec4 g_fluidColor;
Vec4 g_diffuseColor;
Vec3 g_meshColor;
Vec3  g_clearColor;
float g_lightDistance;
float g_fogDistance;

FILE* g_ffmpeg;

// mask for whether show a specific shape (add)
int g_hideShapes[1000] = {0};
vector<Vec3> g_shapeColors;
void DrawShapes();

class Scene;
vector<Scene*> g_scenes;

struct Emitter {
    Emitter() : mSpeed(0.0f), mEnabled(false), mLeftOver(0.0f), mWidth(8)   {}

    Vec3 mPos;
    Vec3 mDir;
    Vec3 mRight;
    float mSpeed;
    bool mEnabled;
    float mLeftOver;
    int mWidth;
};

vector<Emitter> g_emitters(1);    // first emitter is the camera 'gun'

struct Rope {
    std::vector<int> mIndices;
};

vector<Rope> g_ropes;

inline float sqr(float x) { return x*x; }

#include "helpers.h"
#include "scenes.h"
#include "benchmark.h"

void InitRenderHeadless(const RenderInitOptions& options, int width, int height); //add

void Init(int scene, py::array_t<float> scene_params, 
        py::array_t<float> vertices, py::array_t<int> stretch_edges,
        py::array_t<int> bend_edges, py::array_t<int> shear_edges, py::array_t<int> faces,
            bool centerCamera = true, int thread_idx = 0) {
    RandInit();

    if (g_solver) {
        if (g_buffers)
            DestroyBuffers(g_buffers);

        DestroyFluidRenderBuffers(g_fluidRenderBuffers);
        DestroyDiffuseRenderBuffers(g_diffuseRenderBuffers);

        for (auto& iter : g_meshes) {
            NvFlexDestroyTriangleMesh(g_flexLib, iter.first);
            DestroyGpuMesh(iter.second);
        }

        for (auto& iter : g_fields) {
            NvFlexDestroyDistanceField(g_flexLib, iter.first);
            DestroyGpuMesh(iter.second);
        }

        for (auto& iter : g_convexes) {
            NvFlexDestroyConvexMesh(g_flexLib, iter.first);
            DestroyGpuMesh(iter.second);
        }

        g_fields.clear();
        g_meshes.clear();
        g_convexes.clear();

        NvFlexDestroySolver(g_solver);
        g_solver = nullptr;
    }

    // alloc buffers
    g_buffers = AllocBuffers(g_flexLib);

    // map during initialization
    MapBuffers(g_buffers);

    g_buffers->positions.resize(0);
    g_buffers->velocities.resize(0);
    g_buffers->phases.resize(0);

    g_buffers->rigidOffsets.resize(0);
    g_buffers->rigidIndices.resize(0);
    g_buffers->rigidMeshSize.resize(0);
    g_buffers->rigidRotations.resize(0);
    g_buffers->rigidTranslations.resize(0);
    g_buffers->rigidCoefficients.resize(0);
    g_buffers->rigidPlasticThresholds.resize(0);
    g_buffers->rigidPlasticCreeps.resize(0);
    g_buffers->rigidLocalPositions.resize(0);
    g_buffers->rigidLocalNormals.resize(0);

    g_buffers->springIndices.resize(0);
    g_buffers->springLengths.resize(0);
    g_buffers->springStiffness.resize(0);
    g_buffers->triangles.resize(0);
    g_buffers->triangleNormals.resize(0);
    g_buffers->uvs.resize(0);
    
    g_buffers->particle2objInstance.resize(0);

    g_meshSkinIndices.resize(0);
    g_meshSkinWeights.resize(0);

    g_emitters.resize(1);
    g_emitters[0].mEnabled = false;
    g_emitters[0].mSpeed = 1.0f;
    g_emitters[0].mLeftOver = 0.0f;
    g_emitters[0].mWidth = 8;

    g_buffers->shapeGeometry.resize(0);
    g_buffers->shapePositions.resize(0);
    g_buffers->shapeRotations.resize(0);
    g_buffers->shapePrevPositions.resize(0);
    g_buffers->shapePrevRotations.resize(0);
    g_buffers->shapeFlags.resize(0);

    g_ropes.resize(0);

    // remove collision shapes
    delete g_mesh; g_mesh = NULL;

    g_frame = 0;
    g_pause = false;

    g_dt = 1.0f / 60.0f;
    g_waveTime = 0.0f;
    g_windTime = 0.0f;
    g_windStrength = 1.0f;

    g_blur = 1.0f;
    g_fluidColor = Vec4(0.1f, 0.4f, 0.8f, 1.0f);
    g_meshColor = Vec3(0.9f, 0.9f, 0.9f);
    g_drawEllipsoids = false;
    g_drawPoints = true;
    g_drawCloth = true;
    g_expandCloth = 0.0f;

    g_drawOpaque = false;
    g_drawSprings = false;
    g_drawDiffuse = false;
    g_drawMesh = true;
    g_drawRopes = true;
    g_drawDensity = false;
    g_ior = 1.0f;
    g_lightDistance = 2.0f;
    g_fogDistance = 0.005f;

    g_camSpeed = 0.075f;
    g_camNear = 0.01f;
    g_camFar = 1000.0f;

    g_pointScale = 1.0f;
    g_ropeScale = 1.0f;
    g_drawPlaneBias = 0.0f;
    g_floorScaleSize = 200.0f; //add

    // sim params
    g_params.gravity[0] = 0.0f;
    g_params.gravity[1] = -9.8f;
    g_params.gravity[2] = 0.0f;

    g_params.wind[0] = 0.0f;
    g_params.wind[1] = 0.0f;
    g_params.wind[2] = 0.0f;

    g_params.radius = 0.15f;
    g_params.viscosity = 0.0f;
    g_params.dynamicFriction = 0.0f;
    g_params.staticFriction = 0.0f;
    g_params.particleFriction = 0.0f; // scale friction between particles by default
    g_params.freeSurfaceDrag = 0.0f;
    g_params.drag = 0.0f;
    g_params.lift = 0.0f;
    g_params.numIterations = 3;
    g_params.fluidRestDistance = 0.0f;
    g_params.solidRestDistance = 0.0f;

    g_params.anisotropyScale = 1.0f;
    g_params.anisotropyMin = 0.1f;
    g_params.anisotropyMax = 2.0f;
    g_params.smoothing = 1.0f;

    g_params.dissipation = 0.0f;
    g_params.damping = 0.0f;
    g_params.particleCollisionMargin = 0.0f;
    g_params.shapeCollisionMargin = 0.0f;
    g_params.collisionDistance = 0.0f;
    g_params.sleepThreshold = 0.0f;
    g_params.shockPropagation = 0.0f;
    g_params.restitution = 0.0f;

    g_params.maxSpeed = FLT_MAX;
    g_params.maxAcceleration = 100.0f;    // approximately 10x gravity

    g_params.relaxationMode = eNvFlexRelaxationLocal;
    g_params.relaxationFactor = 1.0f;
    g_params.solidPressure = 1.0f;
    g_params.adhesion = 0.0f;
    g_params.cohesion = 0.025f;
    g_params.surfaceTension = 0.0f;
    g_params.vorticityConfinement = 0.0f;
    g_params.buoyancy = 1.0f;
    g_params.diffuseThreshold = 100.0f;
    g_params.diffuseBuoyancy = 1.0f;
    g_params.diffuseDrag = 0.8f;
    g_params.diffuseBallistic = 16;
    g_params.diffuseLifetime = 2.0f;

    g_numSubsteps = 2;

    // planes created after particles
    g_params.numPlanes = 1;

    g_diffuseScale = 0.5f;
    g_diffuseColor = 1.0f;
    g_diffuseMotionScale = 1.0f;
    g_diffuseShadow = false;
    g_diffuseInscatter = 0.8f;
    g_diffuseOutscatter = 0.53f;

    // reset phase 0 particle color to blue
    g_colors[0] = Colour(0.0f, 0.5f, 1.0f);

    g_numSolidParticles = 0;

    g_waveFrequency = 1.5f;
    g_waveAmplitude = 1.5f;
    g_waveFloorTilt = 0.0f;
    g_emit = false;
    g_warmup = false;

    g_mouseParticle = -1;

    g_maxDiffuseParticles = 0;    // number of diffuse particles
    g_maxNeighborsPerParticle = 96;
    g_numExtraParticles = 0;    // number of particles allocated but not made active
    g_maxContactsPerParticle = 6;

    g_sceneLower = FLT_MAX;
    g_sceneUpper = -FLT_MAX;

    // initialize solver desc
    NvFlexSetSolverDescDefaults(&g_solverDesc);

    // create scene
    StartGpuWork();
    // g_scenes[g_scene]->Initialize(scene_params, thread_idx);
    g_scenes[g_scene]->Initialize(scene_params, vertices, stretch_edges, bend_edges, shear_edges, faces, thread_idx);
    std::cout << "scene initialized" << std::endl;
    EndGpuWork();

    uint32_t numParticles = g_buffers->positions.size();
    uint32_t maxParticles = numParticles + g_numExtraParticles*g_numExtraMultiplier;
    std::cout << "numParticles: " << numParticles << std::endl;
    // std::cout << "maxParticles: " << maxParticles << std::endl;

    if (g_params.solidRestDistance == 0.0f)
        g_params.solidRestDistance = g_params.radius;

    // if fluid present then we assume solid particles have the same radius
    if (g_params.fluidRestDistance > 0.0f)
        g_params.solidRestDistance = g_params.fluidRestDistance;

    // set collision distance automatically based on rest distance if not already set
    if (g_params.collisionDistance == 0.0f)
        g_params.collisionDistance = Max(g_params.solidRestDistance, g_params.fluidRestDistance)*0.5f;

    // default particle friction to 10% of shape friction
    if (g_params.particleFriction == 0.0f)
        g_params.particleFriction = g_params.dynamicFriction*0.1f;

    // add a margin for detecting contacts between particles and shapes
    if (g_params.shapeCollisionMargin == 0.0f)
        g_params.shapeCollisionMargin = g_params.collisionDistance*0.5f;

    // calculate particle bounds
    Vec3 particleLower, particleUpper;
    GetParticleBounds(particleLower, particleUpper);
    // std::cout << "particleLower: " << particleLower.x << ", " << particleLower.y << ", " << particleLower.z << std::endl;

    // accommodate shapes
    Vec3 shapeLower, shapeUpper;
    GetShapeBounds(shapeLower, shapeUpper);
    // std::cout << "shapeLower: " << shapeLower.x << ", " << shapeLower.y << ", " << shapeLower.z << std::endl;

    // update bounds
    g_sceneLower = Min(Min(g_sceneLower, particleLower), shapeLower);
    g_sceneUpper = Max(Max(g_sceneUpper, particleUpper), shapeUpper);

    g_sceneLower -= g_params.collisionDistance;
    g_sceneUpper += g_params.collisionDistance;

    // update collision planes to match flexs
    Vec3 up = Normalize(Vec3(-g_waveFloorTilt, 1.0f, 0.0f));

    (Vec4&)g_params.planes[0] = Vec4(up.x, up.y, up.z, 0.0f);
    (Vec4&)g_params.planes[1] = Vec4(0.0f, 0.0f, 1.0f, -g_sceneLower.z);
    (Vec4&)g_params.planes[2] = Vec4(1.0f, 0.0f, 0.0f, -g_sceneLower.x);
    (Vec4&)g_params.planes[3] = Vec4(-1.0f, 0.0f, 0.0f, g_sceneUpper.x);
    (Vec4&)g_params.planes[4] = Vec4(0.0f, 0.0f, -1.0f, g_sceneUpper.z);
    (Vec4&)g_params.planes[5] = Vec4(0.0f, -1.0f, 0.0f, g_sceneUpper.y);

    g_wavePlane = g_params.planes[2][3];

    g_buffers->diffusePositions.resize(g_maxDiffuseParticles);
    g_buffers->diffuseVelocities.resize(g_maxDiffuseParticles);
    g_buffers->diffuseCount.resize(1, 0);

    // for fluid rendering these are the Laplacian smoothed positions
    g_buffers->smoothPositions.resize(maxParticles);

    g_buffers->normals.resize(0);
    g_buffers->normals.resize(maxParticles);

    // initialize normals (just for rendering before simulation starts)
    int numTris = g_buffers->triangles.size() / 3;
    for (int i = 0; i < numTris; ++i) {
        Vec3 v0 = Vec3(g_buffers->positions[g_buffers->triangles[i * 3 + 0]]);
        Vec3 v1 = Vec3(g_buffers->positions[g_buffers->triangles[i * 3 + 1]]);
        Vec3 v2 = Vec3(g_buffers->positions[g_buffers->triangles[i * 3 + 2]]);

        Vec3 n = Cross(v1 - v0, v2 - v0);

        g_buffers->normals[g_buffers->triangles[i * 3 + 0]] += Vec4(n, 0.0f);
        g_buffers->normals[g_buffers->triangles[i * 3 + 1]] += Vec4(n, 0.0f);
        g_buffers->normals[g_buffers->triangles[i * 3 + 2]] += Vec4(n, 0.0f);
    }

    for (int i = 0; i < int(maxParticles); ++i)
        g_buffers->normals[i] = Vec4(SafeNormalize(Vec3(g_buffers->normals[i]), Vec3(0.0f, 1.0f, 0.0f)), 0.0f);


    // save mesh positions for skinning
    if (g_mesh) {
        g_meshRestPositions = g_mesh->m_positions;
    }
    else {
        g_meshRestPositions.resize(0);
    }

    g_solverDesc.maxParticles = maxParticles;
    g_solverDesc.maxDiffuseParticles = g_maxDiffuseParticles;
    g_solverDesc.maxNeighborsPerParticle = g_maxNeighborsPerParticle;
    g_solverDesc.maxContactsPerParticle = g_maxContactsPerParticle;

    // main create method for the Flex solver
    g_solver = NvFlexCreateSolver(g_flexLib, &g_solverDesc);

    // give scene a chance to do some post solver initialization
    g_scenes[g_scene]->PostInitialize();

    // center camera on particles
    if (centerCamera) {
        g_camPos = Vec3((g_sceneLower.x + g_sceneUpper.x)*0.5f, min(g_sceneUpper.y*1.25f, 6.0f), g_sceneUpper.z + min(g_sceneUpper.y, 6.0f)*2.0f);
        g_camAngle = Vec3(0.0f, -DegToRad(15.0f), 0.0f);

        // give scene a chance to modify camera position
        g_scenes[g_scene]->CenterCamera();
    }

    // create active indices (just a contiguous block for the demo)
    g_buffers->activeIndices.resize(g_buffers->positions.size());
    for (int i = 0; i < g_buffers->activeIndices.size(); ++i)
        g_buffers->activeIndices[i] = i;

    // resize particle buffers to fit
    g_buffers->positions.resize(maxParticles);
    g_buffers->velocities.resize(maxParticles);
    g_buffers->phases.resize(maxParticles);

    g_buffers->densities.resize(maxParticles);
    g_buffers->anisotropy1.resize(maxParticles);
    g_buffers->anisotropy2.resize(maxParticles);
    g_buffers->anisotropy3.resize(maxParticles);

    g_buffers->particle2objInstance.resize(maxParticles);

    // save rest positions
    g_buffers->restPositions.resize(g_buffers->positions.size());
    for (int i = 0; i < g_buffers->positions.size(); ++i)
        g_buffers->restPositions[i] = g_buffers->positions[i];

    // builds rigids constraints
    if (g_buffers->rigidOffsets.size()) {
        assert(g_buffers->rigidOffsets.size() > 1);

        const int numRigids = g_buffers->rigidOffsets.size() - 1;

        /*
        printf("rigidOffsets\n");
        for (size_t i = 0; i < (size_t) g_buffers->rigidOffsets.size(); i++) {
            printf("%d %d\n", i, g_buffers->rigidOffsets[i]);
        }

        printf("rigidIndices\n");
        for (size_t i = 0; i < (size_t) g_buffers->rigidIndices.size(); i++) {
            printf("%d %d\n", i, g_buffers->rigidIndices[i]);
        }
         */

        // If the centers of mass for the rigids are not yet computed, this is done here
        // (If the CreateParticleShape method is used instead of the NvFlexExt methods, the centers of mass will be calculated here)
        if (g_buffers->rigidTranslations.size() == 0) {
            g_buffers->rigidTranslations.resize(g_buffers->rigidOffsets.size() - 1, Vec3());
            CalculateRigidCentersOfMass(&g_buffers->positions[0], g_buffers->positions.size(), &g_buffers->rigidOffsets[0], &g_buffers->rigidTranslations[0], &g_buffers->rigidIndices[0], numRigids);
        }

        // calculate local rest space positions
        g_buffers->rigidLocalPositions.resize(g_buffers->rigidOffsets.back());
        CalculateRigidLocalPositions(&g_buffers->positions[0], &g_buffers->rigidOffsets[0], &g_buffers->rigidTranslations[0], &g_buffers->rigidIndices[0], numRigids, &g_buffers->rigidLocalPositions[0]);

        // set rigidRotations to correct length, probably NULL up until here
        g_buffers->rigidRotations.resize(g_buffers->rigidOffsets.size() - 1, Quat());
    }

    // unmap so we can start transferring data to GPU
    UnmapBuffers(g_buffers);

    //-----------------------------
    // Send data to Flex

    NvFlexCopyDesc copyDesc;
    copyDesc.dstOffset = 0;
    copyDesc.srcOffset = 0;
    copyDesc.elementCount = numParticles;

    NvFlexSetParams(g_solver, &g_params);
    NvFlexSetParticles(g_solver, g_buffers->positions.buffer, &copyDesc);
    NvFlexSetVelocities(g_solver, g_buffers->velocities.buffer, &copyDesc);
    NvFlexSetNormals(g_solver, g_buffers->normals.buffer, &copyDesc);
    NvFlexSetPhases(g_solver, g_buffers->phases.buffer, &copyDesc);
    NvFlexSetRestParticles(g_solver, g_buffers->restPositions.buffer, &copyDesc);

    NvFlexSetActive(g_solver, g_buffers->activeIndices.buffer, &copyDesc);
    NvFlexSetActiveCount(g_solver, numParticles);

    // springs
    if (g_buffers->springIndices.size()) {
        assert((g_buffers->springIndices.size() & 1) == 0);
        assert((g_buffers->springIndices.size() / 2) == g_buffers->springLengths.size());

        NvFlexSetSprings(g_solver, g_buffers->springIndices.buffer, g_buffers->springLengths.buffer, g_buffers->springStiffness.buffer, g_buffers->springLengths.size());
    }

    // rigids
    if (g_buffers->rigidOffsets.size()) {
        // std::cout << "g_buffers->rigidOffsets.size(): " << g_buffers->rigidOffsets.size() << std::endl;
        NvFlexSetRigids(g_solver, g_buffers->rigidOffsets.buffer, g_buffers->rigidIndices.buffer, g_buffers->rigidLocalPositions.buffer, g_buffers->rigidLocalNormals.buffer, g_buffers->rigidCoefficients.buffer, g_buffers->rigidPlasticThresholds.buffer, g_buffers->rigidPlasticCreeps.buffer, g_buffers->rigidRotations.buffer, g_buffers->rigidTranslations.buffer, g_buffers->rigidOffsets.size() - 1, g_buffers->rigidIndices.size());
    }

    // inflatables
    if (g_buffers->inflatableTriOffsets.size()) {
        NvFlexSetInflatables(g_solver, g_buffers->inflatableTriOffsets.buffer, g_buffers->inflatableTriCounts.buffer, g_buffers->inflatableVolumes.buffer, g_buffers->inflatablePressures.buffer, g_buffers->inflatableCoefficients.buffer, g_buffers->inflatableTriOffsets.size());
    }

    // dynamic triangles
    if (g_buffers->triangles.size()) {
        NvFlexSetDynamicTriangles(g_solver, g_buffers->triangles.buffer, g_buffers->triangleNormals.buffer, g_buffers->triangles.size() / 3);
    }

    // collision shapes
    if (g_buffers->shapeFlags.size()) {
        NvFlexSetShapes(
            g_solver,
            g_buffers->shapeGeometry.buffer,
            g_buffers->shapePositions.buffer,
            g_buffers->shapeRotations.buffer,
            g_buffers->shapePrevPositions.buffer,
            g_buffers->shapePrevRotations.buffer,
            g_buffers->shapeFlags.buffer,
            int(g_buffers->shapeFlags.size()));
    }

    // create render buffers
    g_fluidRenderBuffers = CreateFluidRenderBuffers(maxParticles, g_interop);
    g_diffuseRenderBuffers = CreateDiffuseRenderBuffers(g_maxDiffuseParticles, g_interop);

    // perform initial sim warm up
    if (g_warmup) {
        printf("Warming up sim..\n");

        // warm it up (relax positions to reach rest density without affecting velocity)
        NvFlexParams copy = g_params;
        copy.numIterations = 4;

        NvFlexSetParams(g_solver, &copy);

        const int kWarmupIterations = 100;

        for (int i = 0; i < kWarmupIterations; ++i) {
            NvFlexUpdateSolver(g_solver, 0.0001f, 1, false);
            NvFlexSetVelocities(g_solver, g_buffers->velocities.buffer, NULL);
        }

        // udpate host copy
        NvFlexGetParticles(g_solver, g_buffers->positions.buffer, NULL);
        NvFlexGetSmoothParticles(g_solver, g_buffers->smoothPositions.buffer, NULL);
        NvFlexGetAnisotropy(g_solver, g_buffers->anisotropy1.buffer, g_buffers->anisotropy2.buffer, g_buffers->anisotropy3.buffer, NULL);

        printf("Finished warm up.\n");
    }
}

/*
void Reset() {
    Init(g_scene, false);
}
*/

void Shutdown() {
    // free buffers
    DestroyBuffers(g_buffers);

    for (auto& iter : g_meshes) {
        NvFlexDestroyTriangleMesh(g_flexLib, iter.first);
        DestroyGpuMesh(iter.second);
    }

    for (auto& iter : g_fields) {
        NvFlexDestroyDistanceField(g_flexLib, iter.first);
        DestroyGpuMesh(iter.second);
    }

    for (auto& iter : g_convexes) {
        NvFlexDestroyConvexMesh(g_flexLib, iter.first);
        DestroyGpuMesh(iter.second);
    }

    g_fields.clear();
    g_meshes.clear();

    NvFlexDestroySolver(g_solver);
    NvFlexShutdown(g_flexLib);
}

void UpdateEmitters() {
    float spin = DegToRad(15.0f);

    const Vec3 forward(-sinf(g_camAngle.x + spin)*cosf(g_camAngle.y), sinf(g_camAngle.y), -cosf(g_camAngle.x + spin)*cosf(g_camAngle.y));
    const Vec3 right(Normalize(Cross(forward, Vec3(0.0f, 1.0f, 0.0f))));

    g_emitters[0].mDir = Normalize(forward + Vec3(0.0, 0.4f, 0.0f));
    g_emitters[0].mRight = right;
    g_emitters[0].mPos = g_camPos + forward*1.f + Vec3(0.0f, 0.2f, 0.0f) + right*0.65f;

    // process emitters
    if (g_emit) {
        int activeCount = NvFlexGetActiveCount(g_solver);

        size_t e = 0;

        // skip camera emitter when moving forward or things get messy
        if (g_camSmoothVel.z >= 0.025f)
            e = 1;

        for (; e < g_emitters.size(); ++e) {
            if (!g_emitters[e].mEnabled)
                continue;

            Vec3 emitterDir = g_emitters[e].mDir;
            Vec3 emitterRight = g_emitters[e].mRight;
            Vec3 emitterPos = g_emitters[e].mPos;

            float r = g_params.fluidRestDistance;
            int phase = NvFlexMakePhase(0, eNvFlexPhaseSelfCollide | eNvFlexPhaseFluid);

            float numParticles = (g_emitters[e].mSpeed / r)*g_dt;

            // whole number to emit
            auto n = int(numParticles + g_emitters[e].mLeftOver);

            if (n)
                g_emitters[e].mLeftOver = (numParticles + g_emitters[e].mLeftOver) - n;
            else
                g_emitters[e].mLeftOver += numParticles;

            // create a grid of particles (n particles thick)
            for (int k = 0; k < n; ++k) {
                int emitterWidth = g_emitters[e].mWidth;
                int numParticles = emitterWidth*emitterWidth;
                for (int i = 0; i < numParticles; ++i) {
                    float x = float(i%emitterWidth) - float(emitterWidth/2);
                    float y = float((i / emitterWidth) % emitterWidth) - float(emitterWidth/2);

                    if ((sqr(x) + sqr(y)) <= (emitterWidth / 2)*(emitterWidth / 2)) {
                        Vec3 up = Normalize(Cross(emitterDir, emitterRight));
                        Vec3 offset = r*(emitterRight*x + up*y) + float(k)*emitterDir*r;

                        if (activeCount < g_buffers->positions.size()) {
                            g_buffers->positions[activeCount] = Vec4(emitterPos + offset, 1.0f);
                            g_buffers->velocities[activeCount] = emitterDir*g_emitters[e].mSpeed;
                            g_buffers->phases[activeCount] = phase;

                            g_buffers->activeIndices.push_back(activeCount);

                            activeCount++;
                        }
                    }
                }
            }
        }
    }
}

void UpdateCamera() {
    Vec3 forward(-sinf(g_camAngle.x)*cosf(g_camAngle.y), sinf(g_camAngle.y), -cosf(g_camAngle.x)*cosf(g_camAngle.y));
    Vec3 right(Normalize(Cross(forward, Vec3(0.0f, 1.0f, 0.0f))));

    g_camSmoothVel = Lerp(g_camSmoothVel, g_camVel, 0.1f);
    g_camPos += (forward*g_camSmoothVel.z + right*g_camSmoothVel.x + Cross(right, forward)*g_camSmoothVel.y);
}

void UpdateMouse() {
    // mouse button is up release particle
    if (g_lastb == -1) {
        if (g_mouseParticle != -1) {
            // restore particle mass
            g_buffers->positions[g_mouseParticle].w = g_mouseMass;

            // deselect
            g_mouseParticle = -1;
        }
    }

    // mouse went down, pick new particle
    if (g_mousePicked) {
        assert(g_mouseParticle == -1);

        Vec3 origin, dir;
        GetViewRay(g_lastx, g_screenHeight - g_lasty, origin, dir);

        const int numActive = NvFlexGetActiveCount(g_solver);

        g_mouseParticle = PickParticle(origin, dir, &g_buffers->positions[0], &g_buffers->phases[0], numActive, g_params.radius*0.8f, g_mouseT);

        if (g_mouseParticle != -1) {
            printf("picked: %d, mass: %f v: %f %f %f\n", g_mouseParticle, g_buffers->positions[g_mouseParticle].w, g_buffers->velocities[g_mouseParticle].x, g_buffers->velocities[g_mouseParticle].y, g_buffers->velocities[g_mouseParticle].z);

            g_mousePos = origin + dir*g_mouseT;
            g_mouseMass = g_buffers->positions[g_mouseParticle].w;
            g_buffers->positions[g_mouseParticle].w = 0.0f;        // increase picked particle's mass to force it towards the point
        }

        g_mousePicked = false;
    }

    // update picked particle position
    if (g_mouseParticle != -1) {
        Vec3 p = Lerp(Vec3(g_buffers->positions[g_mouseParticle]), g_mousePos, 0.8f);
        Vec3 delta = p - Vec3(g_buffers->positions[g_mouseParticle]);

        g_buffers->positions[g_mouseParticle].x = p.x;
        g_buffers->positions[g_mouseParticle].y = p.y;
        g_buffers->positions[g_mouseParticle].z = p.z;

        g_buffers->velocities[g_mouseParticle].x = delta.x / g_dt;
        g_buffers->velocities[g_mouseParticle].y = delta.y / g_dt;
        g_buffers->velocities[g_mouseParticle].z = delta.z / g_dt;
    }
}

void UpdateWind() {
    g_windTime += g_dt;

    const Vec3 kWindDir = Vec3(3.0f, 15.0f, 0.0f);
    const float kNoise = Perlin1D(g_windTime*g_windFrequency, 10, 0.25f);
    Vec3 wind = g_windStrength*kWindDir*Vec3(kNoise, fabsf(kNoise), 0.0f);

    g_params.wind[0] = wind.x;
    g_params.wind[1] = wind.y;
    g_params.wind[2] = wind.z;

    if (g_wavePool) {
        g_waveTime += g_dt;
        g_params.planes[2][3] = g_wavePlane + (sinf(float(g_waveTime)*g_waveFrequency - kPi*0.5f)*0.5f + 0.5f)*g_waveAmplitude;
    }
}

void SyncScene() {
    // let the scene send updates to flex directly
    g_scenes[g_scene]->Sync();
}

void UpdateScene(py::array_t<float> update_params) {
    // give scene a chance to make changes to particle buffers
    g_scenes[g_scene]->Update(update_params);
}

// add input for RenderScene 
void RenderScene(int draw_planes, int draw_objects, int draw_shadow) {
    const int numParticles = NvFlexGetActiveCount(g_solver);
    const int numDiffuse = g_buffers->diffuseCount[0];

    // std::cout << "numParticles: " << numParticles << std::endl;
    // std::cout << "numDiffuse: " << numDiffuse << std::endl;

    //---------------------------------------------------
    // use VBO buffer wrappers to allow Flex to write directly to the OpenGL buffers
    // Flex will take care of any CUDA interop mapping/unmapping during the get() operations
    
    if (numParticles) {
        if (g_interop) {
            // copy data directly from solver to the renderer buffers
            UpdateFluidRenderBuffers(g_fluidRenderBuffers, g_solver, g_drawEllipsoids, g_drawDensity);
        }
        else {
            // copy particle data to GPU render device
            if (g_drawEllipsoids) {
                // if fluid surface rendering then update with smooth positions and anisotropy
                UpdateFluidRenderBuffers(g_fluidRenderBuffers,
                    &g_buffers->smoothPositions[0],
                    (g_drawDensity) ? &g_buffers->densities[0] : (float*)&g_buffers->phases[0],
                    &g_buffers->anisotropy1[0],
                    &g_buffers->anisotropy2[0],
                    &g_buffers->anisotropy3[0],
                    g_buffers->positions.size(),
                    &g_buffers->activeIndices[0],
                    numParticles);
            }
            else {
                // otherwise just send regular positions and no anisotropy
                UpdateFluidRenderBuffers(g_fluidRenderBuffers,
                    &g_buffers->positions[0],
                    (float*)&g_buffers->phases[0],
                    nullptr, nullptr, nullptr,
                    g_buffers->positions.size(),
                    &g_buffers->activeIndices[0],
                    numParticles);
            }
        }
    }

    // GPU Render time doesn't include CPU->GPU copy time
    GraphicsTimerBegin();

    if (numDiffuse) {
        if (g_interop) {
            // copy data directly from solver to the renderer buffers
            UpdateDiffuseRenderBuffers(g_diffuseRenderBuffers, g_solver);
        }
        else {
            // copy diffuse particle data from host to GPU render device
            UpdateDiffuseRenderBuffers(g_diffuseRenderBuffers,
                &g_buffers->diffusePositions[0],
                &g_buffers->diffuseVelocities[0],
                numDiffuse);
        }
    }

    //---------------------------------------
    // setup view and state

    float fov = kPi / 4.0f;
    float aspect = float(g_screenWidth) / g_screenHeight;

    Matrix44 proj = ProjectionMatrix(RadToDeg(fov), aspect, g_camNear, g_camFar);
    Matrix44 view = RotationMatrix(-g_camAngle.x, Vec3(0.0f, 1.0f, 0.0f))*RotationMatrix(-g_camAngle.y, Vec3(cosf(-g_camAngle.x), 0.0f, sinf(-g_camAngle.x)))*TranslationMatrix(-Point3(g_camPos));

    //------------------------------------
    // lighting pass

    // expand scene bounds to fit most scenes
    g_sceneLower = Min(g_sceneLower, Vec3(-2.0f, 0.0f, -2.0f));
    g_sceneUpper = Max(g_sceneUpper, Vec3(2.0f, 2.0f, 2.0f));

    Vec3 sceneExtents = g_sceneUpper - g_sceneLower;
    Vec3 sceneCenter = 0.5f*(g_sceneUpper + g_sceneLower);

    g_lightPos = sceneCenter + g_lightDir*Length(sceneExtents)*g_lightDistance;
    g_lightTarget = sceneCenter;

    // calculate tight bounds for shadow frustum
    // float lightFov = 2.0f*atanf(Length(g_sceneUpper - sceneCenter) / Length(g_lightPos - sceneCenter));
    float lightFov = g_lightFov; //add

    // scale and clamp fov for aesthetics
    lightFov = Clamp(lightFov, DegToRad(25.0f), DegToRad(65.0f));

    Matrix44 lightPerspective = ProjectionMatrix(RadToDeg(lightFov), 1.0f, 1.0f, 1000.0f);
    Matrix44 lightView = LookAtMatrix(Point3(g_lightPos), Point3(g_lightTarget));
    Matrix44 lightTransform = lightPerspective*lightView;

    // radius used for drawing
    float radius = Max(g_params.solidRestDistance, g_params.fluidRestDistance)*0.5f*g_pointScale;

    //std::cout << "g_meshSkinIndices.size(): " << g_meshSkinIndices.size() << std::endl;

    //-------------------------------------
    // shadowing pass
    if (g_meshSkinIndices.size())
        SkinMesh();

    // create shadow maps
    ShadowBegin(g_shadowMap);

    SetView(lightView, lightPerspective);
    SetCullMode(false);

    if (draw_shadow) {
        // give scene a chance to do custom drawing
        if (draw_objects)
            g_scenes[g_scene]->Draw(1);

        if (draw_objects && g_drawMesh)
            DrawMesh(g_mesh, g_meshColor);

        if (draw_objects)
            DrawShapes();

        if (draw_objects && g_drawCloth && g_buffers->triangles.size()) {
            DrawCloth(&g_buffers->positions[0], &g_buffers->normals[0], g_buffers->uvs.size() ? &g_buffers->uvs[0].x : NULL, &g_buffers->triangles[0], g_buffers->triangles.size() / 3, g_buffers->positions.size(), 3, g_expandCloth);
        }

        // if (draw_objects && g_drawRopes) {
        //     for (size_t i = 0; i < g_ropes.size(); ++i){
        //         printf("\ndrawRopes\n");
                
        //         cout << g_ropes.size() << endl; //1
        //         cout << g_buffers->positions.size() << endl; //42 -> 41
        //         cout << g_ropes[i].mIndices[0] << endl; //1 -> 0
        //         cout << g_ropes[i].mIndices[1] << endl; //2 -> 1
        //         cout << g_ropes[i].mIndices[2] << endl; //3 -> 2
        //         cout << g_ropes[i].mIndices.size() << endl; //41

        //         printf("\n");

        //         DrawRope(&g_buffers->positions[0], &g_ropes[i].mIndices[0], g_ropes[i].mIndices.size(), radius*g_ropeScale, i);
        //         }
        // }

        int shadowParticles = numParticles;
        int shadowParticlesOffset = 0;
        
        if (!g_drawPoints) {
            shadowParticles = 0;

            if (g_drawEllipsoids) {
                shadowParticles = numParticles - g_numSolidParticles;
                shadowParticlesOffset = g_numSolidParticles;
            }
        }
        else {
            int offset = g_drawMesh ? g_numSolidParticles : 0;

            shadowParticles = numParticles - offset;
            shadowParticlesOffset = offset;
        }

        if (draw_objects && g_buffers->activeIndices.size())
            DrawPoints(g_fluidRenderBuffers, shadowParticles, shadowParticlesOffset, radius, 2048, 1.0f, lightFov, g_lightPos, g_lightTarget, lightTransform, g_shadowMap, g_drawDensity);

    }

    ShadowEnd();

    //----------------
    // lighting pass

    BindSolidShader(g_lightPos, g_lightTarget, lightTransform, g_shadowMap, 0.0f, Vec4(g_clearColor, g_fogDistance));

    SetView(view, proj);
    SetCullMode(true);

    // When the benchmark measures async compute, we need a graphics workload that runs for a whole frame.
    // We do this by rerendering our simple graphics many times.
    int passes = g_increaseGfxLoadForAsyncComputeTesting ? 50 : 1;

    for (int i = 0; i != passes; i++) {
        if (draw_planes)
            DrawPlanes(
                (Vec4*)g_params.planes,
                g_params.numPlanes,
                g_drawPlaneBias,
                g_floorScaleSize);

        if (draw_objects && g_drawMesh)
            DrawMesh(g_mesh, g_meshColor);


        if (draw_objects)
            DrawShapes();

        if (draw_objects && g_drawCloth && g_buffers->triangles.size())
            DrawCloth(&g_buffers->positions[0], &g_buffers->normals[0], g_buffers->uvs.size() ? &g_buffers->uvs[0].x : nullptr, &g_buffers->triangles[0], g_buffers->triangles.size() / 3, g_buffers->positions.size(), 3, g_expandCloth);

        // if (draw_objects && g_drawRopes) {
        //     for (size_t i = 0; i < g_ropes.size(); ++i)
        //         DrawRope(&g_buffers->positions[0], &g_ropes[i].mIndices[0], g_ropes[i].mIndices.size(), g_params.radius*0.5f*g_ropeScale, i);
        // }

        // give scene a chance to do custom drawing
        if (draw_objects)
            g_scenes[g_scene]->Draw(0);
    }
    UnbindSolidShader();

    // first pass of diffuse particles (behind fluid surface)
    if (draw_objects && g_drawDiffuse)
        RenderDiffuse(g_fluidRenderer, g_diffuseRenderBuffers, numDiffuse, radius*g_diffuseScale, float(g_screenWidth), aspect, fov, g_diffuseColor, g_lightPos, g_lightTarget, lightTransform, g_shadowMap, g_diffuseMotionScale, g_diffuseInscatter, g_diffuseOutscatter, g_diffuseShadow, false);

    if (draw_objects) {
        if (g_drawEllipsoids) {
            // draw solid particles separately
            if (g_numSolidParticles && g_drawPoints)
                DrawPoints(g_fluidRenderBuffers, g_numSolidParticles, 0, radius, float(g_screenWidth), aspect, fov, g_lightPos, g_lightTarget, lightTransform, g_shadowMap, g_drawDensity);

            // render fluid surface
            RenderEllipsoids(g_fluidRenderer, g_fluidRenderBuffers, numParticles - g_numSolidParticles, g_numSolidParticles, radius, float(g_screenWidth), aspect, fov, g_lightPos, g_lightTarget, lightTransform, g_shadowMap, g_fluidColor, g_blur, g_ior, g_drawOpaque);

            // second pass of diffuse particles for particles in front of fluid surface
            if (g_drawDiffuse)
                RenderDiffuse(g_fluidRenderer, g_diffuseRenderBuffers, numDiffuse, radius*g_diffuseScale, float(g_screenWidth), aspect, fov, g_diffuseColor, g_lightPos, g_lightTarget, lightTransform, g_shadowMap, g_diffuseMotionScale, g_diffuseInscatter, g_diffuseOutscatter, g_diffuseShadow, true);
        }
        else {
            // draw all particles as spheres
            if (g_drawPoints) {
                int offset = g_drawMesh ? g_numSolidParticles : 0;

                if (g_buffers->activeIndices.size())
                    DrawPoints(g_fluidRenderBuffers, numParticles - offset, offset, radius, float(g_screenWidth), aspect, fov, g_lightPos, g_lightTarget, lightTransform, g_shadowMap, g_drawDensity);
            }
        }
    }

    GraphicsTimerEnd();
}

void RenderDebug() {
    if (g_mouseParticle != -1) {
        // draw mouse spring
        BeginLines();
        DrawLine(g_mousePos, Vec3(g_buffers->positions[g_mouseParticle]), Vec4(1.0f));
        EndLines();
    }

    // springs
    if (g_drawSprings) {
        Vec4 color;

        if (g_drawSprings == 1) {
            // stretch
            color = Vec4(0.0f, 0.0f, 1.0f, 0.8f);
        }
        if (g_drawSprings == 2) {
            // tether
            color = Vec4(0.0f, 1.0f, 0.0f, 0.8f);
        }

        BeginLines();

        int start = 0;

        for (int i = start; i < g_buffers->springLengths.size(); ++i) {
            if (g_drawSprings == 1 && g_buffers->springStiffness[i] < 0.0f)
                continue;
            if (g_drawSprings == 2 && g_buffers->springStiffness[i] > 0.0f)
                continue;

            int a = g_buffers->springIndices[i * 2];
            int b = g_buffers->springIndices[i * 2 + 1];

            DrawLine(Vec3(g_buffers->positions[a]), Vec3(g_buffers->positions[b]), color);
        }

        EndLines();
    }

    // visualize contacts against the environment
    if (g_drawContacts) {
        const int maxContactsPerParticle = 6;

        NvFlexVector<Vec4> contactPlanes(g_flexLib, g_buffers->positions.size()*maxContactsPerParticle);
        NvFlexVector<Vec4> contactVelocities(g_flexLib, g_buffers->positions.size()*maxContactsPerParticle);
        NvFlexVector<int> contactIndices(g_flexLib, g_buffers->positions.size());
        NvFlexVector<unsigned int> contactCounts(g_flexLib, g_buffers->positions.size());

        NvFlexGetContacts(g_solver, contactPlanes.buffer, contactVelocities.buffer, contactIndices.buffer, contactCounts.buffer);

        // ensure transfers have finished
        contactPlanes.map();
        contactVelocities.map();
        contactIndices.map();
        contactCounts.map();

        BeginLines();

        for (int i = 0; i < int(g_buffers->activeIndices.size()); ++i) {
            const int contactIndex = contactIndices[g_buffers->activeIndices[i]];
            const unsigned int count = contactCounts[contactIndex];

            const float scale = 0.1f;

            for (unsigned int c = 0; c < count; ++c) {
                Vec4 plane = contactPlanes[contactIndex*maxContactsPerParticle + c];

                DrawLine(Vec3(g_buffers->positions[g_buffers->activeIndices[i]]),
                         Vec3(g_buffers->positions[g_buffers->activeIndices[i]]) + Vec3(plane)*scale,
                         Vec4(0.0f, 1.0f, 0.0f, 0.0f));
            }
        }

        EndLines();
    }

    if (g_drawBases) {
        for (int i = 0; i < int(g_buffers->rigidRotations.size()); ++i) {
            BeginLines();

            float size = 0.1f;

            for (int b = 0; b < 3; ++b) {
                Vec3 color(0.0f, 0.0f, 0.0f);

                if (b == 0)
                    color.x = 1.0f;
                else if (b == 1)
                    color.y = 1.0f;
                else
                    color.z = 1.0f;

                Matrix33 frame(g_buffers->rigidRotations[i]);

                DrawLine(Vec3(g_buffers->rigidTranslations[i]),
                         Vec3(g_buffers->rigidTranslations[i] + frame.cols[b] * size),
                         Vec4(color, 0.0f));
            }

            EndLines();
        }
    }

    if (g_drawNormals) {
        NvFlexGetNormals(g_solver, g_buffers->normals.buffer, nullptr);

        BeginLines();

        for (int i = 0; i < g_buffers->normals.size(); ++i) {
            DrawLine(Vec3(g_buffers->positions[i]),
                     Vec3(g_buffers->positions[i] - g_buffers->normals[i] * g_buffers->normals[i].w),
                     Vec4(0.0f, 1.0f, 0.0f, 0.0f));
        }

        EndLines();
    }
}

void DrawShapes() {
    for (int i = 0; i < g_buffers->shapeFlags.size(); ++i) {

        if (g_hideShapes[i] == 1) {
            continue;
        }

        const int flags = g_buffers->shapeFlags[i];

        // unpack flags
        auto type = int(flags & eNvFlexShapeFlagTypeMask);
        //bool dynamic = int(flags&eNvFlexShapeFlagDynamic) > 0;

        // Vec3 color = Vec3(0.9f);
        Vec3 color = g_shapeColors[i];

        if (flags & eNvFlexShapeFlagTrigger) {
            color = Vec3(0.6f, 1.0, 0.6f);
            SetFillMode(true);

            printf("hahahah\n");
        }

        // render with prev positions to match particle update order
        // can also think of this as current/next
        const Quat rotation = g_buffers->shapePrevRotations[i];
        const Vec3 position = Vec3(g_buffers->shapePrevPositions[i]);

        NvFlexCollisionGeometry geo = g_buffers->shapeGeometry[i];

        if (type == eNvFlexShapeSphere) {
            Mesh* sphere = CreateSphere(20, 20, geo.sphere.radius);

            Matrix44 xform = TranslationMatrix(Point3(position))*RotationMatrix(Quat(rotation));
            sphere->Transform(xform);

            DrawMesh(sphere, Vec3(color));

            delete sphere;
        }
        else if (type == eNvFlexShapeCapsule) {
            Mesh* capsule = CreateCapsule(10, 20, geo.capsule.radius, geo.capsule.halfHeight);

            // transform to world space
            Matrix44 xform = TranslationMatrix(Point3(position))*RotationMatrix(Quat(rotation))*RotationMatrix(DegToRad(-90.0f), Vec3(0.0f, 0.0f, 1.0f));
            capsule->Transform(xform);

            DrawMesh(capsule, Vec3(color));

            delete capsule;
        }
        else if (type == eNvFlexShapeBox) {
            Mesh* box = CreateCubeMesh();

            Matrix44 xform = TranslationMatrix(Point3(position))*RotationMatrix(Quat(rotation))*ScaleMatrix(Vec3(geo.box.halfExtents)*2.0f);
            box->Transform(xform);

            DrawMesh(box, Vec3(color));
            delete box;
        }
        else if (type == eNvFlexShapeConvexMesh) {
            if (g_convexes.find(geo.convexMesh.mesh) != g_convexes.end()) {
                GpuMesh* m = g_convexes[geo.convexMesh.mesh];

                if (m) {
                    Matrix44 xform = TranslationMatrix(Point3(g_buffers->shapePositions[i]))*RotationMatrix(Quat(g_buffers->shapeRotations[i]))*ScaleMatrix(geo.convexMesh.scale);
                    DrawGpuMesh(m, xform, Vec3(color));
                }
            }
        }
        else if (type == eNvFlexShapeTriangleMesh) {
            if (g_meshes.find(geo.triMesh.mesh) != g_meshes.end()) {
                GpuMesh* m = g_meshes[geo.triMesh.mesh];

                if (m) {
                    Matrix44 xform = TranslationMatrix(Point3(position))*RotationMatrix(Quat(rotation))*ScaleMatrix(geo.triMesh.scale);
                    DrawGpuMesh(m, xform, Vec3(color));
                }
            }
        }
        else if (type == eNvFlexShapeSDF) {
            if (g_fields.find(geo.sdf.field) != g_fields.end()) {
                GpuMesh* m = g_fields[geo.sdf.field];

                if (m) {
                    Matrix44 xform = TranslationMatrix(Point3(position))*RotationMatrix(Quat(rotation))*ScaleMatrix(geo.sdf.scale);
                    DrawGpuMesh(m, xform, Vec3(color));
                }
            }
        }
    }

    SetFillMode(g_wireframe);
}


// returns the new scene if one is selected
int DoUI() {
    // gui may set a new scene
    int newScene = -1;

    if (g_showHelp) {
        const int numParticles = NvFlexGetActiveCount(g_solver);
        const int numDiffuse = g_buffers->diffuseCount[0];

        int x = g_screenWidth - 200;
        int y = g_screenHeight - 23;

        // imgui
        unsigned char button = 0;
        if (g_lastb == SDL_BUTTON_LEFT)
            button = IMGUI_MBUT_LEFT;
        else if (g_lastb == SDL_BUTTON_RIGHT)
            button = IMGUI_MBUT_RIGHT;

        imguiBeginFrame(g_lastx, g_screenHeight - g_lasty, button, 0);

        x += 180;

        int fontHeight = 13;

        if (0) {
            DrawImguiString(x, y, Vec3(1.0f), IMGUI_ALIGN_RIGHT, "Frame: %d", g_frame); y -= fontHeight * 2;

            if (!g_ffmpeg) {
                DrawImguiString(x, y, Vec3(1.0f), IMGUI_ALIGN_RIGHT, "Frame Time: %.2fms", g_realdt*1000.0f); y -= fontHeight * 2;

                // If detailed profiling is enabled, then these timers will contain the overhead of the detail timers, so we won't display them.
                if (!g_profile) {
                    DrawImguiString(x, y, Vec3(1.0f), IMGUI_ALIGN_RIGHT, "Sim Time (CPU): %.2fms", g_updateTime*1000.0f); y -= fontHeight;
                    DrawImguiString(x, y, Vec3(0.97f, 0.59f, 0.27f), IMGUI_ALIGN_RIGHT, "Sim Latency (GPU): %.2fms", g_simLatency); y -= fontHeight * 2;

                    BenchmarkUpdateGraph();
                }
                else {
                    y -= fontHeight * 3;
                }
            }

            DrawImguiString(x, y, Vec3(1.0f), IMGUI_ALIGN_RIGHT, "Particle Count: %d", numParticles); y -= fontHeight;
            DrawImguiString(x, y, Vec3(1.0f), IMGUI_ALIGN_RIGHT, "Diffuse Count: %d", numDiffuse); y -= fontHeight;
            DrawImguiString(x, y, Vec3(1.0f), IMGUI_ALIGN_RIGHT, "Rigid Count: %d", g_buffers->rigidOffsets.size() > 0 ? g_buffers->rigidOffsets.size() - 1 : 0); y -= fontHeight;
            DrawImguiString(x, y, Vec3(1.0f), IMGUI_ALIGN_RIGHT, "Spring Count: %d", g_buffers->springLengths.size()); y -= fontHeight;
            DrawImguiString(x, y, Vec3(1.0f), IMGUI_ALIGN_RIGHT, "Num Substeps: %d", g_numSubsteps); y -= fontHeight;
            DrawImguiString(x, y, Vec3(1.0f), IMGUI_ALIGN_RIGHT, "Num Iterations: %d", g_params.numIterations); y -= fontHeight * 2;

            DrawImguiString(x, y, Vec3(1.0f), IMGUI_ALIGN_RIGHT, "Device: %s", g_deviceName); y -= fontHeight * 2;
        }

        if (g_profile) {
            DrawImguiString(x, y, Vec3(0.97f, 0.59f, 0.27f), IMGUI_ALIGN_RIGHT, "Total GPU Sim Latency: %.2fms", g_timers.total); y -= fontHeight * 2;

            DrawImguiString(x, y, Vec3(0.0f, 1.0f, 0.0f), IMGUI_ALIGN_RIGHT, "GPU Latencies"); y -= fontHeight;

            DrawImguiString(x, y, Vec3(1.0f), IMGUI_ALIGN_RIGHT, "Predict: %.2fms", g_timers.predict); y -= fontHeight;
            DrawImguiString(x, y, Vec3(1.0f), IMGUI_ALIGN_RIGHT, "Create Cell Indices: %.2fms", g_timers.createCellIndices); y -= fontHeight;
            DrawImguiString(x, y, Vec3(1.0f), IMGUI_ALIGN_RIGHT, "Sort Cell Indices: %.2fms", g_timers.sortCellIndices); y -= fontHeight;
            DrawImguiString(x, y, Vec3(1.0f), IMGUI_ALIGN_RIGHT, "Reorder: %.2fms", g_timers.reorder); y -= fontHeight;
            DrawImguiString(x, y, Vec3(1.0f), IMGUI_ALIGN_RIGHT, "CreateGrid: %.2fms", g_timers.createGrid); y -= fontHeight;
            DrawImguiString(x, y, Vec3(1.0f), IMGUI_ALIGN_RIGHT, "Collide Particles: %.2fms", g_timers.collideParticles); y -= fontHeight;
            DrawImguiString(x, y, Vec3(1.0f), IMGUI_ALIGN_RIGHT, "Collide Shapes: %.2fms", g_timers.collideShapes); y -= fontHeight;
            DrawImguiString(x, y, Vec3(1.0f), IMGUI_ALIGN_RIGHT, "Collide Triangles: %.2fms", g_timers.collideTriangles); y -= fontHeight;
            DrawImguiString(x, y, Vec3(1.0f), IMGUI_ALIGN_RIGHT, "Calculate Density: %.2fms", g_timers.calculateDensity); y -= fontHeight;
            DrawImguiString(x, y, Vec3(1.0f), IMGUI_ALIGN_RIGHT, "Solve Densities: %.2fms", g_timers.solveDensities); y -= fontHeight;
            DrawImguiString(x, y, Vec3(1.0f), IMGUI_ALIGN_RIGHT, "Solve Velocities: %.2fms", g_timers.solveVelocities); y -= fontHeight;
            DrawImguiString(x, y, Vec3(1.0f), IMGUI_ALIGN_RIGHT, "Solve Rigids: %.2fms", g_timers.solveShapes); y -= fontHeight;
            DrawImguiString(x, y, Vec3(1.0f), IMGUI_ALIGN_RIGHT, "Solve Springs: %.2fms", g_timers.solveSprings); y -= fontHeight;
            DrawImguiString(x, y, Vec3(1.0f), IMGUI_ALIGN_RIGHT, "Solve Inflatables: %.2fms", g_timers.solveInflatables); y -= fontHeight;
            DrawImguiString(x, y, Vec3(1.0f), IMGUI_ALIGN_RIGHT, "Solve Contacts: %.2fms", g_timers.solveContacts); y -= fontHeight;
            DrawImguiString(x, y, Vec3(1.0f), IMGUI_ALIGN_RIGHT, "Apply Deltas: %.2fms", g_timers.applyDeltas); y -= fontHeight;
            DrawImguiString(x, y, Vec3(1.0f), IMGUI_ALIGN_RIGHT, "Finalize: %.2fms", g_timers.finalize); y -= fontHeight;
            DrawImguiString(x, y, Vec3(1.0f), IMGUI_ALIGN_RIGHT, "Update Triangles: %.2fms", g_timers.updateTriangles); y -= fontHeight;
            DrawImguiString(x, y, Vec3(1.0f), IMGUI_ALIGN_RIGHT, "Update Normals: %.2fms", g_timers.updateNormals); y -= fontHeight;
            DrawImguiString(x, y, Vec3(1.0f), IMGUI_ALIGN_RIGHT, "Update Bounds: %.2fms", g_timers.updateBounds); y -= fontHeight;
            DrawImguiString(x, y, Vec3(1.0f), IMGUI_ALIGN_RIGHT, "Calculate Anisotropy: %.2fms", g_timers.calculateAnisotropy); y -= fontHeight;
            DrawImguiString(x, y, Vec3(1.0f), IMGUI_ALIGN_RIGHT, "Update Diffuse: %.2fms", g_timers.updateDiffuse); y -= fontHeight * 2;
        }

        /*
        x -= 180;

        int uiOffset = 250;
        int uiBorder = 20;
        int uiWidth = 200;
        int uiHeight = g_screenHeight - uiOffset - uiBorder * 3;
        int uiLeft = uiBorder;

        if (g_tweakPanel)
            imguiBeginScrollArea("Scene", uiLeft, g_screenHeight - uiBorder - uiOffset, uiWidth, uiOffset, &g_levelScroll);
        else
            imguiBeginScrollArea("Scene", uiLeft, uiBorder, uiWidth, g_screenHeight - uiBorder - uiBorder, &g_levelScroll);

        for (int i = 0; i < int(g_scenes.size()); ++i) {
            unsigned int color = g_scene == i ? imguiRGBA(255, 151, 61, 255) : imguiRGBA(255, 255, 255, 200);
            if (imguiItem(g_scenes[i]->GetName(), true, color)) {
                newScene = i;
            }
        }

        imguiEndScrollArea();

        if (g_tweakPanel) {
            static int scroll = 0;

            imguiBeginScrollArea("Options", uiLeft, g_screenHeight - uiBorder - uiHeight - uiOffset - uiBorder, uiWidth, uiHeight, &scroll);
            imguiSeparatorLine();

            // global options
            imguiLabel("Global");
            if (imguiCheck("Emit particles", g_emit))
                g_emit = !g_emit;

            if (imguiCheck("Pause", g_pause))
                g_pause = !g_pause;

            imguiSeparatorLine();

            if (imguiCheck("Wireframe", g_wireframe))
                g_wireframe = !g_wireframe;

            if (imguiCheck("Draw Points", g_drawPoints))
                g_drawPoints = !g_drawPoints;

            if (imguiCheck("Draw Fluid", g_drawEllipsoids))
                g_drawEllipsoids = !g_drawEllipsoids;

            if (imguiCheck("Draw Mesh", g_drawMesh)) {
                g_drawMesh = !g_drawMesh;
                g_drawRopes = !g_drawRopes;
            }

            if (imguiCheck("Draw Basis", g_drawBases))
                g_drawBases = !g_drawBases;

            if (imguiCheck("Draw Springs", bool(g_drawSprings != 0)))
                g_drawSprings = (g_drawSprings) ? 0 : 1;

            if (imguiCheck("Draw Contacts", g_drawContacts))
                g_drawContacts = !g_drawContacts;

            imguiSeparatorLine();

            // scene options
            g_scenes[g_scene]->DoGui();

            if (imguiButton("Reset Scene"))
                g_resetScene = true;

            imguiSeparatorLine();

            auto n = float(g_numSubsteps);
            if (imguiSlider("Num Substeps", &n, 1, 10, 1))
                g_numSubsteps = int(n);

            n = float(g_params.numIterations);
            if (imguiSlider("Num Iterations", &n, 1, 20, 1))
                g_params.numIterations = int(n);

            imguiSeparatorLine();
            imguiSlider("Gravity X", &g_params.gravity[0], -50.0f, 50.0f, 1.0f);
            imguiSlider("Gravity Y", &g_params.gravity[1], -50.0f, 50.0f, 1.0f);
            imguiSlider("Gravity Z", &g_params.gravity[2], -50.0f, 50.0f, 1.0f);

            imguiSeparatorLine();
            imguiSlider("Radius", &g_params.radius, 0.01f, 0.5f, 0.01f);
            imguiSlider("Solid Radius", &g_params.solidRestDistance, 0.0f, 0.5f, 0.001f);
            imguiSlider("Fluid Radius", &g_params.fluidRestDistance, 0.0f, 0.5f, 0.001f);

            // common params
            imguiSeparatorLine();
            imguiSlider("Dynamic Friction", &g_params.dynamicFriction, 0.0f, 1.0f, 0.01f);
            imguiSlider("Static Friction", &g_params.staticFriction, 0.0f, 1.0f, 0.01f);
            imguiSlider("Particle Friction", &g_params.particleFriction, 0.0f, 1.0f, 0.01f);
            imguiSlider("Restitution", &g_params.restitution, 0.0f, 1.0f, 0.01f);
            imguiSlider("SleepThreshold", &g_params.sleepThreshold, 0.0f, 1.0f, 0.01f);
            imguiSlider("Shock Propagation", &g_params.shockPropagation, 0.0f, 10.0f, 0.01f);
            imguiSlider("Damping", &g_params.damping, 0.0f, 10.0f, 0.01f);
            imguiSlider("Dissipation", &g_params.dissipation, 0.0f, 0.01f, 0.0001f);
            imguiSlider("SOR", &g_params.relaxationFactor, 0.0f, 5.0f, 0.01f);

            imguiSlider("Collision Distance", &g_params.collisionDistance, 0.0f, 0.5f, 0.001f);
            imguiSlider("Collision Margin", &g_params.shapeCollisionMargin, 0.0f, 5.0f, 0.01f);

            // cloth params
            imguiSeparatorLine();
            imguiSlider("Wind", &g_windStrength, -1.0f, 1.0f, 0.01f);
            imguiSlider("Drag", &g_params.drag, 0.0f, 1.0f, 0.01f);
            imguiSlider("Lift", &g_params.lift, 0.0f, 1.0f, 0.01f);
            imguiSeparatorLine();

            // fluid params
            imguiSlider("Adhesion", &g_params.adhesion, 0.0f, 10.0f, 0.01f);
            imguiSlider("Cohesion", &g_params.cohesion, 0.0f, 0.2f, 0.0001f);
            imguiSlider("Surface Tension", &g_params.surfaceTension, 0.0f, 50.0f, 0.01f);
            imguiSlider("Viscosity", &g_params.viscosity, 0.0f, 120.0f, 0.01f);
            imguiSlider("Vorticicty Confinement", &g_params.vorticityConfinement, 0.0f, 120.0f, 0.1f);
            imguiSlider("Solid Pressure", &g_params.solidPressure, 0.0f, 1.0f, 0.01f);
            imguiSlider("Surface Drag", &g_params.freeSurfaceDrag, 0.0f, 1.0f, 0.01f);
            imguiSlider("Buoyancy", &g_params.buoyancy, -1.0f, 1.0f, 0.01f);

            imguiSeparatorLine();
            imguiSlider("Anisotropy Scale", &g_params.anisotropyScale, 0.0f, 30.0f, 0.01f);
            imguiSlider("Smoothing", &g_params.smoothing, 0.0f, 1.0f, 0.01f);

            // diffuse params
            imguiSeparatorLine();
            imguiSlider("Diffuse Threshold", &g_params.diffuseThreshold, 0.0f, 1000.0f, 1.0f);
            imguiSlider("Diffuse Buoyancy", &g_params.diffuseBuoyancy, 0.0f, 2.0f, 0.01f);
            imguiSlider("Diffuse Drag", &g_params.diffuseDrag, 0.0f, 2.0f, 0.01f);
            imguiSlider("Diffuse Scale", &g_diffuseScale, 0.0f, 1.5f, 0.01f);
            imguiSlider("Diffuse Alpha", &g_diffuseColor.w, 0.0f, 3.0f, 0.01f);
            imguiSlider("Diffuse Inscatter", &g_diffuseInscatter, 0.0f, 2.0f, 0.01f);
            imguiSlider("Diffuse Outscatter", &g_diffuseOutscatter, 0.0f, 2.0f, 0.01f);
            imguiSlider("Diffuse Motion Blur", &g_diffuseMotionScale, 0.0f, 5.0f, 0.1f);

            n = float(g_params.diffuseBallistic);
            if (imguiSlider("Diffuse Ballistic", &n, 1, 40, 1))
                g_params.diffuseBallistic = int(n);

            imguiEndScrollArea();
        }
        */

        imguiEndFrame();

        // kick render commands
        DrawImguiGraph();
    }

    return newScene;
}

void UpdateFrame(py::array_t<float> update_params, int draw_planes, int draw_objects, int draw_shadow) {
    static double lastTime;

    // real elapsed frame time
    double frameBeginTime = GetSeconds();

    g_realdt = float(frameBeginTime - lastTime);
    lastTime = frameBeginTime;

    // do gamepad input polling
    double currentTime = frameBeginTime;
    static double lastJoyTime = currentTime;

    if (g_gamecontroller && currentTime - lastJoyTime > g_dt) {
        lastJoyTime = currentTime;

        int leftStickX = SDL_GameControllerGetAxis(g_gamecontroller, SDL_CONTROLLER_AXIS_LEFTX);
        int leftStickY = SDL_GameControllerGetAxis(g_gamecontroller, SDL_CONTROLLER_AXIS_LEFTY);
        int rightStickX = SDL_GameControllerGetAxis(g_gamecontroller, SDL_CONTROLLER_AXIS_RIGHTX);
        int rightStickY = SDL_GameControllerGetAxis(g_gamecontroller, SDL_CONTROLLER_AXIS_RIGHTY);
        int leftTrigger = SDL_GameControllerGetAxis(g_gamecontroller, SDL_CONTROLLER_AXIS_TRIGGERLEFT);
        int rightTrigger = SDL_GameControllerGetAxis(g_gamecontroller, SDL_CONTROLLER_AXIS_TRIGGERRIGHT);

        Vec2 leftStick(joyAxisFilter(leftStickX, 0), joyAxisFilter(leftStickY, 0));
        Vec2 rightStick(joyAxisFilter(rightStickX, 1), joyAxisFilter(rightStickY, 1));
        Vec2 trigger(leftTrigger / 32768.0f, rightTrigger / 32768.0f);

        if (leftStick.x != 0.0f || leftStick.y != 0.0f ||
            rightStick.x != 0.0f || rightStick.y != 0.0f) {
            // note constant factor to speed up analog control compared to digital because it is more controllable.
            g_camVel.z = -4 * g_camSpeed * leftStick.y;
            g_camVel.x = 4 * g_camSpeed * leftStick.x;

            // cam orientation
            g_camAngle.x -= rightStick.x * 0.05f;
            g_camAngle.y -= rightStick.y * 0.05f;
        }

        // Handle left stick motion
        static bool bLeftStick = false;

        if ((leftStick.x != 0.0f || leftStick.y != 0.0f) && !bLeftStick) {
            bLeftStick = true;
        }
        else if ((leftStick.x == 0.0f && leftStick.y == 0.0f) && bLeftStick) {
            bLeftStick = false;
            g_camVel.z = -4 * g_camSpeed * leftStick.y;
            g_camVel.x = 4 * g_camSpeed * leftStick.x;
        }

        // Handle triggers as controller button events
        void ControllerButtonEvent(SDL_ControllerButtonEvent event);

        static bool bLeftTrigger = false;
        static bool bRightTrigger = false;
        SDL_ControllerButtonEvent e;

        if (!bLeftTrigger && trigger.x > 0.0f) {
            e.type = SDL_CONTROLLERBUTTONDOWN;
            e.button = SDL_CONTROLLER_BUTTON_LEFT_TRIGGER;
            ControllerButtonEvent(e);
            bLeftTrigger = true;
        }
        else if (bLeftTrigger && trigger.x == 0.0f) {
            e.type = SDL_CONTROLLERBUTTONUP;
            e.button = SDL_CONTROLLER_BUTTON_LEFT_TRIGGER;
            ControllerButtonEvent(e);
            bLeftTrigger = false;
        }

        if (!bRightTrigger && trigger.y > 0.0f) {
            e.type = SDL_CONTROLLERBUTTONDOWN;
            e.button = SDL_CONTROLLER_BUTTON_RIGHT_TRIGGER;
            ControllerButtonEvent(e);
            bRightTrigger = true;
        }
        else if (bRightTrigger && trigger.y == 0.0f) {
            e.type = SDL_CONTROLLERBUTTONDOWN;
            e.button = SDL_CONTROLLER_BUTTON_RIGHT_TRIGGER;
            ControllerButtonEvent(e);
            bRightTrigger = false;
        }
    }

    //-------------------------------------------------------------------
    // Scene Update

    double waitBeginTime = GetSeconds();

    MapBuffers(g_buffers);

    double waitEndTime = GetSeconds();

    // Getting timers causes CPU/GPU sync, so we do it after a map
    float newSimLatency = NvFlexGetDeviceLatency(g_solver, &g_GpuTimers.computeBegin, &g_GpuTimers.computeEnd, &g_GpuTimers.computeFreq);
    float newGfxLatency = RendererGetDeviceTimestamps(&g_GpuTimers.renderBegin, &g_GpuTimers.renderEnd, &g_GpuTimers.renderFreq);
    (void)newGfxLatency;

    UpdateCamera();

    if (!g_pause || g_step) {
        UpdateEmitters();
        UpdateMouse();
        UpdateWind();
        UpdateScene(update_params);
    }
   
    //-------------------------------------------------------------------
    // Render

    double renderBeginTime = GetSeconds();

    if (g_profile && (!g_pause || g_step)) {
        if (g_benchmark) {
            g_numDetailTimers = NvFlexGetDetailTimers(g_solver, &g_detailTimers);
        }
        else {
            memset(&g_timers, 0, sizeof(g_timers));
            NvFlexGetTimers(g_solver, &g_timers);
        }
    }

    StartFrame(Vec4(g_clearColor, 1.0f));

    // main scene render
    RenderScene(draw_planes, draw_objects, draw_shadow);
    RenderDebug();

    int newScene = DoUI();

    EndFrame();

    // If user has disabled async compute, ensure that no compute can overlap
    // graphics by placing a sync between them
    if (!g_useAsyncCompute) {
        NvFlexComputeWaitForGraphics(g_flexLib);}

    UnmapBuffers(g_buffers);

    // move mouse particle (must be done here as GetViewRay() uses the GL projection state)
    if (g_mouseParticle != -1) {
        Vec3 origin, dir;
        GetViewRay(g_lastx, g_screenHeight - g_lasty, origin, dir);

        g_mousePos = origin + dir*g_mouseT;
    }

    double renderEndTime = GetSeconds();

    // if user requested a scene reset process it now
    if (g_resetScene) {
        // Reset();
        g_resetScene = false;
    }
    //-------------------------------------------------------------------
    // Flex Update

    double updateBeginTime = GetSeconds();

    // send any particle updates to the solver
    NvFlexSetParticles(g_solver, g_buffers->positions.buffer, nullptr);
    NvFlexSetVelocities(g_solver, g_buffers->velocities.buffer, nullptr);
    NvFlexSetPhases(g_solver, g_buffers->phases.buffer, nullptr);
    NvFlexSetActive(g_solver, g_buffers->activeIndices.buffer, nullptr);

    NvFlexSetActiveCount(g_solver, g_buffers->activeIndices.size());

    // allow scene to update constraints etc
    SyncScene();

    if (g_shapesChanged) {
        NvFlexSetShapes(
            g_solver,
            g_buffers->shapeGeometry.buffer,
            g_buffers->shapePositions.buffer,
            g_buffers->shapeRotations.buffer,
            g_buffers->shapePrevPositions.buffer,
            g_buffers->shapePrevRotations.buffer,
            g_buffers->shapeFlags.buffer,
            int(g_buffers->shapeFlags.size()));

        g_shapesChanged = false;
    }

    if (!g_pause || g_step) {
        // tick solver
        NvFlexSetParams(g_solver, &g_params);
        NvFlexUpdateSolver(g_solver, g_dt, g_numSubsteps, g_profile);

        g_frame++;
        g_step = false;
    }

    // read back base particle data
    // Note that flexGet calls don't wait for the GPU, they just queue a GPU copy
    // to be executed later.
    // When we're ready to read the fetched buffers we'll Map them, and that's when
    // the CPU will wait for the GPU flex update and GPU copy to finish.
    NvFlexGetParticles(g_solver, g_buffers->positions.buffer, nullptr);
    NvFlexGetVelocities(g_solver, g_buffers->velocities.buffer, nullptr);
    NvFlexGetNormals(g_solver, g_buffers->normals.buffer, nullptr);

    // readback triangle normals
    if (g_buffers->triangles.size())
        NvFlexGetDynamicTriangles(g_solver, g_buffers->triangles.buffer, g_buffers->triangleNormals.buffer, g_buffers->triangles.size() / 3);

    // readback rigid transforms
    if (g_buffers->rigidOffsets.size())
        NvFlexGetRigids(g_solver, g_buffers->rigidOffsets.buffer, g_buffers->rigidIndices.buffer, g_buffers->rigidLocalPositions.buffer, nullptr, nullptr, nullptr, nullptr, g_buffers->rigidRotations.buffer, g_buffers->rigidTranslations.buffer);

    if (!g_interop) {
        // if not using interop then we read back fluid data to host
        if (g_drawEllipsoids) {
            NvFlexGetSmoothParticles(g_solver, g_buffers->smoothPositions.buffer, nullptr);
            NvFlexGetAnisotropy(g_solver, g_buffers->anisotropy1.buffer, g_buffers->anisotropy2.buffer, g_buffers->anisotropy3.buffer, NULL);
        }

        // read back diffuse data to host
        if (g_drawDensity)
            NvFlexGetDensities(g_solver, g_buffers->densities.buffer, nullptr);

        if (GetNumDiffuseRenderParticles(g_diffuseRenderBuffers)) {
            NvFlexGetDiffuseParticles(g_solver, g_buffers->diffusePositions.buffer, g_buffers->diffuseVelocities.buffer, g_buffers->diffuseCount.buffer);
        }
    }
    else {
        // read back just the new diffuse particle count, render buffers will be updated during rendering
        NvFlexGetDiffuseParticles(g_solver, nullptr, nullptr, g_buffers->diffuseCount.buffer);
    }

    double updateEndTime = GetSeconds();
    //-------------------------------------------------------
    // Update the on-screen timers

    auto newUpdateTime = float(updateEndTime - updateBeginTime);
    auto newRenderTime = float(renderEndTime - renderBeginTime);
    auto newWaitTime = float(waitBeginTime - waitEndTime);

    // Exponential filter to make the display easier to read
    const float timerSmoothing = 0.05f;

    g_updateTime = (g_updateTime == 0.0f) ? newUpdateTime : Lerp(g_updateTime, newUpdateTime, timerSmoothing);
    g_renderTime = (g_renderTime == 0.0f) ? newRenderTime : Lerp(g_renderTime, newRenderTime, timerSmoothing);
    g_waitTime = (g_waitTime == 0.0f) ? newWaitTime : Lerp(g_waitTime, newWaitTime, timerSmoothing);
    g_simLatency = (g_simLatency == 0.0f) ? newSimLatency : Lerp(g_simLatency, newSimLatency, timerSmoothing);

    if (g_benchmark) newScene = BenchmarkUpdate();

    // flush out the last frame before freeing up resources in the event of a scene change
    // this is necessary for d3d12
    PresentFrame(g_vsync);

    // if gui or benchmark requested a scene change process it now
    if (newScene != -1) {
        g_scene = newScene;
        // Init(g_scene);
    }
}

void ReshapeWindow(int width, int height) {
    if (!g_benchmark)
        printf("Reshaping\n");

    ReshapeRender(g_window);

    if (!g_fluidRenderer || (width != g_screenWidth || height != g_screenHeight)) {
        if (g_fluidRenderer)
            DestroyFluidRenderer(g_fluidRenderer);
        g_fluidRenderer = CreateFluidRenderer(width, height);
    }

    g_screenWidth = width;
    g_screenHeight = height;
}

void InputArrowKeysDown(int key, int x, int y) {
    switch (key) {
        case SDLK_DOWN: {
            if (g_selectedScene < int(g_scenes.size()) - 1)
                g_selectedScene++;

            // update scroll UI to center on selected scene
            g_levelScroll = max((g_selectedScene - 4) * 24, 0);
            break;
        }
        case SDLK_UP: {
            if (g_selectedScene > 0)
                g_selectedScene--;

            // update scroll UI to center on selected scene
            g_levelScroll = max((g_selectedScene - 4) * 24, 0);
            break;
        }
        case SDLK_LEFT: {
            if (g_scene > 0)
                --g_scene;
            // Init(g_scene);

            // update scroll UI to center on selected scene
            g_levelScroll = max((g_scene - 4) * 24, 0);
            break;
        }
        case SDLK_RIGHT: {
            if (g_scene < int(g_scenes.size()) - 1)
                ++g_scene;
            // Init(g_scene);

            // update scroll UI to center on selected scene
            g_levelScroll = max((g_scene - 4) * 24, 0);
            break;
        }
    }
}

void InputArrowKeysUp(int key, int x, int y) {
}

bool InputKeyboardDown(unsigned char key, int x, int y) {
    if (key > '0' && key <= '9') {
        g_scene = key - '0' - 1;
        // Init(g_scene);
        return false;
    }

    float kSpeed = g_camSpeed;

    switch (key) {
        case 'w': {
            g_camVel.z = kSpeed;
            break;
        }
        case 's': {
            g_camVel.z = -kSpeed;
            break;
        }
        case 'a': {
            g_camVel.x = -kSpeed;
            break;
        }
        case 'd': {
            g_camVel.x = kSpeed;
            break;
        }
        case 'q': {
            g_camVel.y = kSpeed;
            break;
        }
        case 'z': {
            //g_drawCloth = !g_drawCloth;
            g_camVel.y = -kSpeed;
            break;
        }
        case 'r': {
            g_resetScene = true;
            break;
        }
        case 'y': {
            g_wavePool = !g_wavePool;
            break;
        }
        case 'p': {
            g_pause = !g_pause;
            break;
        }
        case 'o': {
            g_step = true;
            break;
        }
        case 'h': {
            g_showHelp = !g_showHelp;
            break;
        }
        case 'e': {
            g_drawEllipsoids = !g_drawEllipsoids;
            break;
        }
        case 't': {
            g_drawOpaque = !g_drawOpaque;
            break;
        }
        case 'v': {
            g_drawPoints = !g_drawPoints;
            break;
        }
        case 'f': {
            g_drawSprings = (g_drawSprings + 1) % 3;
            break;
        }
        case 'i': {
            g_drawDiffuse = !g_drawDiffuse;
            break;
        }
        case 'm': {
            g_drawMesh = !g_drawMesh;
            break;
        }
        case 'n': {
            g_drawRopes = !g_drawRopes;
            break;
        }
        case 'j': {
            g_windTime = 0.0f;
            g_windStrength = 1.5f;
            g_windFrequency = 0.2f;
            break;
        }
        case '.': {
            g_profile = !g_profile;
            break;
        }
        case 'g': {
            if (g_params.gravity[1] != 0.0f)
                g_params.gravity[1] = 0.0f;
            else
                g_params.gravity[1] = -9.8f;

            break;
        }
        case '-': {
            if (g_params.numPlanes)
                g_params.numPlanes--;

            break;
        }
        case ' ': {
            g_emit = !g_emit;
            break;
        }
        case ';': {
            g_debug = !g_debug;
            break;
        }
        case 13: {
            g_scene = g_selectedScene;
            // Init(g_scene);
            break;
        }
        case 27: {
            // return quit = true
            return true;
        }
    };

    g_scenes[g_scene]->KeyDown(key);

    return false;
}

void InputKeyboardUp(unsigned char key, int x, int y) {
    switch (key) {
        case 'w':
        case 's': {
            g_camVel.z = 0.0f;
            break;
        }
        case 'a':
        case 'd': {
            g_camVel.x = 0.0f;
            break;
        }
        case 'q':
        case 'z': {
            g_camVel.y = 0.0f;
            break;
        }
    };
}

void MouseFunc(int b, int state, int x, int y) {
    switch (state) {
        case SDL_RELEASED: {
            g_lastx = x;
            g_lasty = y;
            g_lastb = -1;

            break;
        }
        case SDL_PRESSED: {
            g_lastx = x;
            g_lasty = y;
            g_lastb = b;
            if ((SDL_GetModState() & KMOD_LSHIFT) && g_lastb == SDL_BUTTON_LEFT) {
                // record that we need to update the picked particle
                g_mousePicked = true;
            }
            break;
        }
    };
}

void MousePassiveMotionFunc(int x, int y) {
    g_lastx = x;
    g_lasty = y;
}

void MouseMotionFunc(unsigned state, int x, int y) {
    auto dx = float(x - g_lastx);
    auto dy = float(y - g_lasty);

    g_lastx = x;
    g_lasty = y;

    if (state & SDL_BUTTON_RMASK) {
        const float kSensitivity = DegToRad(0.1f);
        const float kMaxDelta = FLT_MAX;

        g_camAngle.x -= Clamp(dx*kSensitivity, -kMaxDelta, kMaxDelta);
        g_camAngle.y -= Clamp(dy*kSensitivity, -kMaxDelta, kMaxDelta);
    }
}

bool g_Error = false;

void ErrorCallback(NvFlexErrorSeverity severity, const char* msg, const char* file, int line) {
    printf("Flex: %s - %s:%d\n", msg, file, line);
    g_Error = (severity == eNvFlexLogError);
    //assert(0); asserts are bad for TeamCity
}

void ControllerButtonEvent(SDL_ControllerButtonEvent event) {
    // map controller buttons to keyboard keys
    if (event.type == SDL_CONTROLLERBUTTONDOWN) {
        InputKeyboardDown(GetKeyFromGameControllerButton(SDL_GameControllerButton(event.button)), 0, 0);
        InputArrowKeysDown(GetKeyFromGameControllerButton(SDL_GameControllerButton(event.button)), 0, 0);

        if (event.button == SDL_CONTROLLER_BUTTON_LEFT_TRIGGER) {
            // Handle picking events using the game controller
            g_lastx = g_screenWidth / 2;
            g_lasty = g_screenHeight / 2;
            g_lastb = 1;

            // record that we need to update the picked particle
            g_mousePicked = true;
        }
    }
    else {
        InputKeyboardUp(GetKeyFromGameControllerButton(SDL_GameControllerButton(event.button)), 0, 0);
        InputArrowKeysUp(GetKeyFromGameControllerButton(SDL_GameControllerButton(event.button)), 0, 0);

        if (event.button == SDL_CONTROLLER_BUTTON_LEFT_TRIGGER) {
            // Handle picking events using the game controller
            g_lastx = g_screenWidth / 2;
            g_lasty = g_screenHeight / 2;
            g_lastb = -1;
        }
    }
}

void ControllerDeviceUpdate() {
    if (SDL_NumJoysticks() > 0) {
        SDL_JoystickEventState(SDL_ENABLE);
        if (SDL_IsGameController(0)) {
            g_gamecontroller = SDL_GameControllerOpen(0);
        }
    }
}

void SDLInit(const char* title) {
    if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_GAMECONTROLLER) < 0) // Initialize SDL's Video subsystem and game controllers
        printf("Unable to initialize SDL");

    unsigned int flags = SDL_WINDOW_RESIZABLE;

    if (g_graphics == 0) {
        SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
        flags = SDL_WINDOW_RESIZABLE | SDL_WINDOW_OPENGL;
    }

    g_window = SDL_CreateWindow(title, SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
        g_screenWidth, g_screenHeight, flags);

    g_windowId = SDL_GetWindowID(g_window);
}

char octopus_path[100];
char rope_path[100];
char bowl_high_path[100];
char box_ultra_high_path[100];
char box_very_high_path[100];
char teapot_path[100];
char armadillo_path[100];
char bunny_path[100];
char box_high_path[100];
char sphere_path[100];

char* make_path(char* full_path, std::string path) {
    strcpy(full_path, getenv("PYFLEXROOT"));
    strcat(full_path, path.c_str());
    return full_path;
}

void pyflex_init(bool headless=false) {
    //add
    g_headless = headless;
    if (g_headless) {
        g_interop = false;
        g_pause = false;
    }

    // Customized scenes
    // scene_idx
    g_scenes.push_back(new yz_BunnyBath("Bunny Bath", true));
    g_scenes.push_back(new yz_BoxBath("Box Bath", true));
    g_scenes.push_back(new yz_DamBreak("Dam Break", true));
    g_scenes.push_back(new yz_RigidFall("Rigid Fall"));
    g_scenes.push_back(new yz_RiceFall("Rice Fall")); // Fluid Fall 4

    auto *plasticStackScene = new yz_SoftBody("Plastic Stack"); 
    g_scenes.push_back(plasticStackScene); //5

    g_scenes.push_back(new yz_FluidShake("Fluid Shake"));
    g_scenes.push_back(new yz_BoxBathExt("Box Bath Extension", true));
    g_scenes.push_back(new yz_FluidIceShake("Fluid Ice Shake"));

    g_scenes.push_back(new yz_MassRope("Mass Rope"));
    g_scenes.push_back(new yz_FlagCloth("Flag Cloth"));

    g_scenes.push_back(new yz_SoftRod("Soft Rod"));

    g_scenes.push_back(new yz_ClothRigid("Cloth Rigid"));

    g_scenes.push_back(new yz_Granular("Granular"));

    g_scenes.push_back(new yz_BunnyGrip("BunnyGrip"));

    g_scenes.push_back(new yz_ClothManip("ClothManip"));

    g_scenes.push_back(new yz_SoftFall("SoftFall"));

    g_scenes.push_back(new yz_FluidPour("FluidPour"));

    g_scenes.push_back(new yz_GranularManip("GranularManip"));

    g_scenes.push_back(new yz_FluidAndBox("FluidAndBox"));

    g_scenes.push_back(new yx_Coffee("Coffee")); //20

    g_scenes.push_back(new yx_Capsule("Capsule")); //21

    g_scenes.push_back(new yx_Carrots("Carrots")); //22

    g_scenes.push_back(new yx_Coffee_Capsule("Coffee_Capsule")); //23

    g_scenes.push_back(new by_Apple("Apple")); //24
    g_scenes.push_back(new by_SingleYCB("Single YCB")); // 25
    g_scenes.push_back(new by_SoftRope("Soft Rope")); //26
    g_scenes.push_back(new by_Cloth("Cloth")); //27
    g_scenes.push_back(new by_MultiYCB("Multi YCB")); //28
    g_scenes.push_back(new SoftgymCloth("Softgym Cloth"));  //29
    g_scenes.push_back(new SoftgymCloth2("Softgym Cloth")); //30

    g_scenes.push_back(new by_RigidRope("Rigid Rope")); //31
    g_scenes.push_back(new by_RigidGranular("Rigid Granular")); //32
    g_scenes.push_back(new by_RigidCloth("Rigid Cloth")); //33
    g_scenes.push_back(new by_RopeCloth("Rope Cloth")); //34
    g_scenes.push_back(new by_Granular("Bowl Granular")); //35
    g_scenes.push_back(new by_BowlFluid("Bowl Fluid")); //36

    // copied from softgym 37
    g_scenes.push_back(new by_SoftBody("Soft Body")); //37
    g_scenes.push_back(new by_RopeRigid("Rope Granular")); //38
    
    // Hetergeneous scenes 05/29/2024
    g_scenes.push_back(new by_RopeGranular("Rope Granular")); //39

    // Multi objects scene
    g_scenes.push_back(new by_MultiObjects("Multi Objects")); //40

    // Rigid scene
    g_scenes.push_back(new RigidDebris("Rigid Debris")); //41

    // init graphics
    RenderInitOptions options;

    switch (g_graphics) {
        case 0:
            break;
        case 1:
            break;
        case 2:
            // workaround for a driver issue with D3D12 with msaa, force it to off
            // options.numMsaaSamples = 1;
            // Currently interop doesn't work on d3d12
            g_interop = false;
            break;
        default:
            assert(0);
    }

    // Create the demo context
    CreateDemoContext(g_graphics);

    std::string str;
    str = "Flex Demo (Compute: CUDA) ";

    switch (g_graphics) {
        case 0:
            str += "(Graphics: OpenGL)";
            break;
        case 1:
            str += "(Graphics: DX11)";
            break;
        case 2:
            str += "(Graphics: DX12)";
            break;
    }
    const char *title = str.c_str();

    if (!g_headless) {
        SDLInit(title);

        // init graphics
        options.window = g_window;
        options.numMsaaSamples = g_msaaSamples;
        options.asyncComputeBenchmark = g_asyncComputeBenchmark;
        options.defaultFontHeight = -1;
        options.fullscreen = g_fullscreen;

        InitRender(options);

        if (g_fullscreen)
            SDL_SetWindowFullscreen(g_window, SDL_WINDOW_FULLSCREEN_DESKTOP);

        ReshapeWindow(g_screenWidth, g_screenHeight);
    } else {
		options.numMsaaSamples = g_msaaSamples;

		InitRenderHeadless(options, g_screenWidth, g_screenHeight);
        g_fluidRenderer = CreateFluidRenderer(g_screenWidth, g_screenHeight);
    }

    NvFlexInitDesc desc;
    desc.deviceIndex = g_device;
    desc.enableExtensions = g_extensions;
    desc.renderDevice = 0;
    desc.renderContext = 0;
    desc.computeContext = 0;
    desc.computeType = eNvFlexCUDA;

    // Init Flex library, note that no CUDA methods should be called before this
    // point to ensure we get the device context we want
    g_flexLib = NvFlexInit(NV_FLEX_VERSION, ErrorCallback, &desc);

    if (g_Error || g_flexLib == nullptr) {
        printf("Could not initialize Flex, exiting.\n");
        exit(-1);
    }

    // store device name
    strcpy(g_deviceName, NvFlexGetDeviceName(g_flexLib));
    printf("Compute Device: %s\n\n", g_deviceName);

    if (g_benchmark)
        g_scene = BenchmarkInit();

    // create shadow maps
    g_shadowMap = ShadowCreate();

    // init default scene
    // StartGpuWork();
    // Init(g_scene);
    // EndGpuWork();
    printf("Pyflex init done\n");
}

void pyflex_clean() {

    if (g_fluidRenderer)
        DestroyFluidRenderer(g_fluidRenderer);

    DestroyFluidRenderBuffers(g_fluidRenderBuffers);
    DestroyDiffuseRenderBuffers(g_diffuseRenderBuffers);

    ShadowDestroy(g_shadowMap);

    Shutdown();
    if (!g_headless) {
        DestroyRender();

        SDL_DestroyWindow(g_window);
        SDL_Quit();
    }
}

int main() {
    pyflex_init();
    pyflex_clean();

    return 0;
}

void SDL_EventFunc() {
    SDL_Event e;
    while (SDL_PollEvent(&e)) {
        switch (e.type) {
            case SDL_QUIT:
                break;

            case SDL_KEYDOWN:
                InputArrowKeysDown(e.key.keysym.sym, 0, 0);
                break;

            case SDL_KEYUP:
                if (e.key.keysym.sym < 256 && (e.key.keysym.mod == 0 || (e.key.keysym.mod & KMOD_NUM)))
                    InputKeyboardUp(e.key.keysym.sym, 0, 0);
                InputArrowKeysUp(e.key.keysym.sym, 0, 0);
                break;

            case SDL_MOUSEMOTION:
                if (e.motion.state)
                    MouseMotionFunc(e.motion.state, e.motion.x, e.motion.y);
                else
                    MousePassiveMotionFunc(e.motion.x, e.motion.y);
                break;

            case SDL_MOUSEBUTTONDOWN:
            case SDL_MOUSEBUTTONUP:
                MouseFunc(e.button.button, e.button.state, e.motion.x, e.motion.y);
                break;

            case SDL_WINDOWEVENT:
                if (e.window.windowID == g_windowId) {
                    if (e.window.event == SDL_WINDOWEVENT_SIZE_CHANGED)
                        ReshapeWindow(e.window.data1, e.window.data2);
                }
                break;

            case SDL_WINDOWEVENT_LEAVE:
                g_camVel = Vec3(0.0f, 0.0f, 0.0f);
                break;

            case SDL_CONTROLLERBUTTONUP:
            case SDL_CONTROLLERBUTTONDOWN:
                ControllerButtonEvent(e.cbutton);
                break;

            case SDL_JOYDEVICEADDED:
            case SDL_JOYDEVICEREMOVED:
                ControllerDeviceUpdate();
                break;
        }
    }
}

void pyflex_step(
        py::array_t<float> update_params,
        int draw_planes, int draw_objects, int draw_shadow, bool render_depth) {
    UpdateFrame(update_params, draw_planes, draw_objects, draw_shadow);
    SDL_EventFunc();
}

float rand_float(float LO, float HI) {
    return LO + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/(HI-LO)));
}

// void pyflex_set_scene(int scene_idx, py::array_t<float> scene_params, int thread_idx = 0) {
//     g_scene = scene_idx;
//     g_selectedScene = g_scene;
//     Init(g_selectedScene, scene_params, true, thread_idx);
// }

void pyflex_set_scene(
    int scene_idx,
    py::array_t<float> scene_params,
    py::array_t<float> vertices,
    py::array_t<int> stretch_edges,
    py::array_t<int> bend_edges,
    py::array_t<int> shear_edges,
    py::array_t<int> faces,
    int thread_idx = 0)
{
    g_scene = scene_idx;
    g_selectedScene = g_scene;
    Init(g_selectedScene, scene_params, vertices,
         stretch_edges, bend_edges, shear_edges, faces,
         true, thread_idx);
}

void pyflex_MapShapeBuffers(SimBuffers* buffers) {
    buffers->shapeGeometry.map();
    buffers->shapePositions.map();
    buffers->shapeRotations.map();
    buffers->shapePrevPositions.map();
    buffers->shapePrevRotations.map();
    buffers->shapeFlags.map();
}

void pyflex_UnmapShapeBuffers(SimBuffers* buffers) {
    buffers->shapeGeometry.unmap();
    buffers->shapePositions.unmap();
    buffers->shapeRotations.unmap();
    buffers->shapePrevPositions.unmap();
    buffers->shapePrevRotations.unmap();
    buffers->shapeFlags.unmap();
}

void pyflex_add_box(
        py::array_t<float> halfEdge_, py::array_t<float> center_, py::array_t<float> quat_,
        float hideShape, py::array_t<float> color) {

    pyflex_MapShapeBuffers(g_buffers);

    auto ptr_halfEdge = (float *) halfEdge_.request().ptr;
    Vec3 halfEdge = Vec3(ptr_halfEdge[0], ptr_halfEdge[1], ptr_halfEdge[2]);

    auto ptr_center = (float *) center_.request().ptr;
    Vec3 center = Vec3(ptr_center[0], ptr_center[1], ptr_center[2]);

    auto ptr_quat = (float *) quat_.request().ptr;
    Quat quat = Quat(ptr_quat[0], ptr_quat[1], ptr_quat[2], ptr_quat[3]);

    AddBox(halfEdge, center, quat);

    pyflex_UnmapShapeBuffers(g_buffers);

    // record hideShape
    g_hideShapes[g_buffers->shapeFlags.size() - 1] = (int)hideShape;

    // record shape color
    auto ptr = (float *) color.request().ptr;
    g_shapeColors.push_back(Vec3(ptr[0], ptr[1], ptr[2]));
}

void pyflex_add_capsule(py::array_t<float> params, py::array_t<float> lower_pos, py::array_t<float> quat_) {
    pyflex_MapShapeBuffers(g_buffers);

    auto ptr_params = (float *) params.request().ptr;
    float capsule_radius = ptr_params[0];
    float halfheight = ptr_params[1];

    auto ptr_lower_pos = (float *) lower_pos.request().ptr;
    Vec3 lower_position = Vec3(ptr_lower_pos[0], ptr_lower_pos[1], ptr_lower_pos[2]);

    auto ptr_quat = (float *) quat_.request().ptr;
    Quat quat = Quat(ptr_quat[0], ptr_quat[1], ptr_quat[2], ptr_quat[3]);

    AddCapsule(capsule_radius, halfheight, lower_position, quat);

    pyflex_UnmapShapeBuffers(g_buffers);
}

// Modified by Baoyu 11/28/2023
void pyflex_add_mesh(const char *s, float scaling, int hideShape, py::array_t<float> color, 
                    py::array_t<float> translation, py::array_t<float> rotation, bool texture=false) 
{
    Mesh* m = ImportMesh(s, texture);
    m->Transform(ScaleMatrix(Vec3(scaling)));

    NvFlexTriangleMeshId mesh = CreateTriangleMesh(m);

    // record translation
    auto ptr_translation = (float *) translation.request().ptr;
    Vec3 translation_ = Vec3(ptr_translation[0], ptr_translation[1], ptr_translation[2]);

    //record rotation
    auto ptr_rotation = (float *) rotation.request().ptr;
    Quat rotation_ = Quat(ptr_rotation[0], ptr_rotation[1], ptr_rotation[2], ptr_rotation[3]);

    // add triangle mesh
    pyflex_MapShapeBuffers(g_buffers);
    AddTriangleMesh(mesh, translation_, rotation_, 1.0f);
    pyflex_UnmapShapeBuffers(g_buffers);

    // record hideShape
    g_hideShapes[g_buffers->shapeFlags.size() - 1] = (int)hideShape;

    // record shape color
    auto ptr = (float *) color.request().ptr;
    g_shapeColors.push_back(Vec3(ptr[0], ptr[1], ptr[2]));
}

// Copied from softgym
void pyflex_add_sphere(float radius, py::array_t<float> position_, py::array_t<float> quat_) {
    pyflex_MapShapeBuffers(g_buffers);

    auto ptr_center = (float *) position_.request().ptr;
    Vec3 center = Vec3(ptr_center[0], ptr_center[1], ptr_center[2]);

    auto ptr_quat = (float *) quat_.request().ptr;
    Quat quat = Quat(ptr_quat[0], ptr_quat[1], ptr_quat[2], ptr_quat[3]);

    AddSphere(radius, center, quat);

    pyflex_UnmapShapeBuffers(g_buffers);
}

int pyflex_get_n_particles() {
    g_buffers->positions.map();
    int n_particles = g_buffers->positions.size();
    g_buffers->positions.unmap();
    return n_particles;
}

int pyflex_get_n_shapes() {
    g_buffers->shapePositions.map();
    int n_shapes = g_buffers->shapePositions.size();
    g_buffers->shapePositions.unmap();
    return n_shapes;
}

py::array_t<float> pyflex_get_positions() {
    g_buffers->positions.map();

    auto positions = py::array_t<float>((size_t) g_buffers->positions.size() * 4);
    auto ptr = (float *) positions.request().ptr;

    for (size_t i = 0; i < (size_t) g_buffers->positions.size(); i++) {
        ptr[i * 4] = g_buffers->positions[i].x;
        ptr[i * 4 + 1] = g_buffers->positions[i].y;
        ptr[i * 4 + 2] = g_buffers->positions[i].z;
        ptr[i * 4 + 3] = g_buffers->positions[i].w;
    }

    g_buffers->positions.unmap();

    return positions;
}

py::array_t<int> pyflex_get_particle_2_obj_instance() {
    g_buffers->particle2objInstance.map();
    auto indices = py::array_t<int>((size_t)g_buffers->particle2objInstance.size());
    auto ptr = (int *)indices.request().ptr;

    for (size_t i = 0; i < (size_t)g_buffers->particle2objInstance.size(); i++)
    {
        ptr[i] = g_buffers->particle2objInstance[i];
    }

    g_buffers->particle2objInstance.unmap();

    return indices;
}

py::array_t<int> pyflex_get_edges()
{
    g_buffers->springIndices.map();
    auto edges = py::array_t<int>((size_t)g_buffers->springIndices.size());
    auto ptr = (int *)edges.request().ptr;

    for (size_t i = 0; i < (size_t)g_buffers->springIndices.size(); i++)
    {
        ptr[i] = g_buffers->springIndices[i];
    }

    g_buffers->springIndices.unmap();

    return edges;
}

py::array_t<int> pyflex_get_faces()
{
    g_buffers->triangles.map();
    auto triangles = py::array_t<int>((size_t)g_buffers->triangles.size());
    auto ptr = (int *)triangles.request().ptr;

    for (size_t i = 0; i < (size_t)g_buffers->triangles.size(); i++)
    {
        ptr[i] = g_buffers->triangles[i];
    }

    g_buffers->triangles.unmap();
    return triangles;
}

void pyflex_set_positions(py::array_t<float> positions) {
    g_buffers->positions.map();

    auto buf = positions.request();
    auto ptr = (float *) buf.ptr;

    for (size_t i = 0; i < (size_t)g_buffers->positions.size(); i++) {
        g_buffers->positions[i].x = ptr[i * 4];
        g_buffers->positions[i].y = ptr[i * 4 + 1];
        g_buffers->positions[i].z = ptr[i * 4 + 2];
        g_buffers->positions[i].w = ptr[i * 4 + 3];
    }

    g_buffers->positions.unmap();

    NvFlexSetParticles(g_solver, g_buffers->positions.buffer, nullptr);
}

py::array_t<float> pyflex_get_restPositions() {
    g_buffers->restPositions.map();

    auto restPositions = py::array_t<float>((size_t) g_buffers->restPositions.size() * 4);
    auto ptr = (float *) restPositions.request().ptr;

    for (size_t i = 0; i < (size_t) g_buffers->restPositions.size(); i++) {
        ptr[i * 4] = g_buffers->restPositions[i].x;
        ptr[i * 4 + 1] = g_buffers->restPositions[i].y;
        ptr[i * 4 + 2] = g_buffers->restPositions[i].z;
        ptr[i * 4 + 3] = g_buffers->restPositions[i].w;
    }

    g_buffers->restPositions.unmap();

    return restPositions;
}

py::array_t<int> pyflex_get_rigidOffsets() {
    g_buffers->rigidOffsets.map();

    auto rigidOffsets = py::array_t<int>((size_t) g_buffers->rigidOffsets.size());
    auto ptr = (int *) rigidOffsets.request().ptr;

    for (size_t i = 0; i < (size_t) g_buffers->rigidOffsets.size(); i++) {
        ptr[i] = g_buffers->rigidOffsets[i];
    }

    g_buffers->rigidOffsets.unmap();

    return rigidOffsets;
}

py::array_t<int> pyflex_get_rigidIndices() {
    g_buffers->rigidIndices.map();

    auto rigidIndices = py::array_t<int>((size_t) g_buffers->rigidIndices.size());
    auto ptr = (int *) rigidIndices.request().ptr;

    for (size_t i = 0; i < (size_t) g_buffers->rigidIndices.size(); i++) {
        ptr[i] = g_buffers->rigidIndices[i];
    }

    g_buffers->rigidIndices.unmap();

    return rigidIndices;
}

int pyflex_get_n_rigidPositions() {
    g_buffers->rigidLocalPositions.map();
    int n_rigidPositions = g_buffers->rigidLocalPositions.size();
    g_buffers->rigidLocalPositions.unmap();
    return n_rigidPositions;
}

py::array_t<float> pyflex_get_rigidLocalPositions() {
    g_buffers->rigidLocalPositions.map();

    auto rigidLocalPositions = py::array_t<float>((size_t) g_buffers->rigidLocalPositions.size() * 3);
    auto ptr = (float *) rigidLocalPositions.request().ptr;

    for (size_t i = 0; i < (size_t) g_buffers->rigidLocalPositions.size(); i++) {
        ptr[i * 3] = g_buffers->rigidLocalPositions[i].x;
        ptr[i * 3 + 1] = g_buffers->rigidLocalPositions[i].y;
        ptr[i * 3 + 2] = g_buffers->rigidLocalPositions[i].z;
    }

    g_buffers->rigidLocalPositions.unmap();

    return rigidLocalPositions;
}

py::array_t<float> pyflex_get_rigidGlobalPositions() {
    g_buffers->rigidOffsets.map();
    g_buffers->rigidIndices.map();
    g_buffers->rigidLocalPositions.map();
    g_buffers->rigidTranslations.map();
    g_buffers->rigidRotations.map();

    auto rigidGlobalPositions = py::array_t<float>((size_t) g_buffers->positions.size() * 3);
    auto ptr = (float *) rigidGlobalPositions.request().ptr;

    int count = 0;
    int numRigids = g_buffers->rigidOffsets.size() - 1;
    float n_clusters[g_buffers->positions.size()] = {0};

    for (int i = 0; i < numRigids; i++) {
        const int st = g_buffers->rigidOffsets[i];
        const int ed = g_buffers->rigidOffsets[i + 1];

        assert(ed - st);

        for (int j = st; j < ed; j++) {
            const int r = g_buffers->rigidIndices[j];
            Vec3 p = Rotate(g_buffers->rigidRotations[i], g_buffers->rigidLocalPositions[count++]) + g_buffers->rigidTranslations[i];

            if (n_clusters[r] == 0) {
                ptr[r * 3] = p.x;
                ptr[r * 3 + 1] = p.y;
                ptr[r * 3 + 2] = p.z;
            }
            else {
                ptr[r * 3] += p.x;
                ptr[r * 3 + 1] += p.y;
                ptr[r * 3 + 2] += p.z;
            }
            n_clusters[r] += 1;
        }
    }

    for (int i = 0; i < g_buffers->positions.size(); i++) {
        if (n_clusters[i] > 0) {
            ptr[i * 3] /= n_clusters[i];
            ptr[i * 3 + 1] /= n_clusters[i];
            ptr[i * 3 + 2] /= n_clusters[i];
        }
    }

    g_buffers->rigidOffsets.unmap();
    g_buffers->rigidIndices.unmap();
    g_buffers->rigidLocalPositions.unmap();
    g_buffers->rigidTranslations.unmap();
    g_buffers->rigidRotations.unmap();

    return rigidGlobalPositions;
}

int pyflex_get_n_rigids() {
    g_buffers->rigidRotations.map();
    int n_rigids = g_buffers->rigidRotations.size();
    g_buffers->rigidRotations.unmap();
    return n_rigids;
}

py::array_t<float> pyflex_get_rigidRotations() {
    g_buffers->rigidRotations.map();

    auto rigidRotations = py::array_t<float>((size_t) g_buffers->rigidRotations.size() * 4);
    auto ptr = (float *) rigidRotations.request().ptr;

    for (size_t i = 0; i < (size_t) g_buffers->rigidRotations.size(); i++) {
        ptr[i * 4] = g_buffers->rigidRotations[i].x;
        ptr[i * 4 + 1] = g_buffers->rigidRotations[i].y;
        ptr[i * 4 + 2] = g_buffers->rigidRotations[i].z;
        ptr[i * 4 + 3] = g_buffers->rigidRotations[i].w;
    }

    g_buffers->rigidRotations.unmap();

    return rigidRotations;
}

py::array_t<float> pyflex_get_rigidTranslations() {
    g_buffers->rigidTranslations.map();

    auto rigidTranslations = py::array_t<float>((size_t) g_buffers->rigidTranslations.size() * 3);
    auto ptr = (float *) rigidTranslations.request().ptr;

    for (size_t i = 0; i < (size_t) g_buffers->rigidTranslations.size(); i++) {
        ptr[i * 3] = g_buffers->rigidTranslations[i].x;
        ptr[i * 3 + 1] = g_buffers->rigidTranslations[i].y;
        ptr[i * 3 + 2] = g_buffers->rigidTranslations[i].z;
    }

    g_buffers->rigidTranslations.unmap();

    return rigidTranslations;
}

py::array_t<float> pyflex_get_velocities() {
    g_buffers->velocities.map();

    auto velocities = py::array_t<float>((size_t) g_buffers->velocities.size() * 3);
    auto ptr = (float *) velocities.request().ptr;

    for (size_t i = 0; i < (size_t) g_buffers->velocities.size(); i++) {
        ptr[i * 3] = g_buffers->velocities[i].x;
        ptr[i * 3 + 1] = g_buffers->velocities[i].y;
        ptr[i * 3 + 2] = g_buffers->velocities[i].z;
    }

    g_buffers->velocities.unmap();

    return velocities;
}

void pyflex_set_velocities(py::array_t<float> velocities) {
    g_buffers->velocities.map();

    auto buf = velocities.request();
    auto ptr = (float *) buf.ptr;

    for (size_t i = 0; i < (size_t)g_buffers->velocities.size(); i++) {
        g_buffers->velocities[i].x = ptr[i * 3];
        g_buffers->velocities[i].y = ptr[i * 3 + 1];
        g_buffers->velocities[i].z = ptr[i * 3 + 2];
    }

    g_buffers->velocities.unmap();
}

py::array_t<float> pyflex_get_shape_states() {
    pyflex_MapShapeBuffers(g_buffers);

    // position + velocity + rotation + prev_rotation
    auto states = py::array_t<float>((size_t) g_buffers->shapePositions.size() * (3 + 3 + 4 + 4));
    auto buf = states.request();
    auto ptr = (float *) buf.ptr;

    for (size_t i = 0; i < (size_t) g_buffers->shapePositions.size(); i++) {
        ptr[i * 14] = g_buffers->shapePositions[i].x;
        ptr[i * 14 + 1] = g_buffers->shapePositions[i].y;
        ptr[i * 14 + 2] = g_buffers->shapePositions[i].z;

        ptr[i * 14 + 3] = g_buffers->shapePrevPositions[i].x;
        ptr[i * 14 + 4] = g_buffers->shapePrevPositions[i].y;
        ptr[i * 14 + 5] = g_buffers->shapePrevPositions[i].z;

        ptr[i * 14 + 6] = g_buffers->shapeRotations[i].x;
        ptr[i * 14 + 7] = g_buffers->shapeRotations[i].y;
        ptr[i * 14 + 8] = g_buffers->shapeRotations[i].z;
        ptr[i * 14 + 9] = g_buffers->shapeRotations[i].w;

        ptr[i * 14 + 10] = g_buffers->shapePrevRotations[i].x;
        ptr[i * 14 + 11] = g_buffers->shapePrevRotations[i].y;
        ptr[i * 14 + 12] = g_buffers->shapePrevRotations[i].z;
        ptr[i * 14 + 13] = g_buffers->shapePrevRotations[i].w;
    }

    pyflex_UnmapShapeBuffers(g_buffers);

    return states;
}

void pyflex_set_shape_states(py::array_t<float> states) {
    pyflex_MapShapeBuffers(g_buffers);

    auto buf = states.request();
    auto ptr = (float *) buf.ptr;

    for (size_t i = 0; i < (size_t) g_buffers->shapePositions.size(); i++) {
        g_buffers->shapePositions[i].x = ptr[i * 14];
        g_buffers->shapePositions[i].y = ptr[i * 14 + 1];
        g_buffers->shapePositions[i].z = ptr[i * 14 + 2];

        g_buffers->shapePrevPositions[i].x = ptr[i * 14 + 3];
        g_buffers->shapePrevPositions[i].y = ptr[i * 14 + 4];
        g_buffers->shapePrevPositions[i].z = ptr[i * 14 + 5];

        g_buffers->shapeRotations[i].x = ptr[i * 14 + 6];
        g_buffers->shapeRotations[i].y = ptr[i * 14 + 7];
        g_buffers->shapeRotations[i].z = ptr[i * 14 + 8];
        g_buffers->shapeRotations[i].w = ptr[i * 14 + 9];

        g_buffers->shapePrevRotations[i].x = ptr[i * 14 + 10];
        g_buffers->shapePrevRotations[i].y = ptr[i * 14 + 11];
        g_buffers->shapePrevRotations[i].z = ptr[i * 14 + 12];
        g_buffers->shapePrevRotations[i].w = ptr[i * 14 + 13];
    }

    UpdateShapes();

    pyflex_UnmapShapeBuffers(g_buffers);
}

py::array_t<int> pyflex_get_phases() {
    g_buffers->phases.map();

    auto phases = py::array_t<int>((size_t) g_buffers->phases.size());
    auto ptr = (int *) phases.request().ptr;

    for (size_t i = 0; i < (size_t) g_buffers->phases.size(); i++) {
        ptr[i] = g_buffers->phases[i];
    }

    g_buffers->phases.unmap();

    return phases;
}

void pyflex_set_phases(py::array_t<int> phases) {
    g_buffers->phases.map();

    auto buf = phases.request();
    auto ptr = (int *) buf.ptr;

    for (size_t i = 0; i < (size_t) g_buffers->phases.size(); i++) {
        g_buffers->phases[i] = ptr[i];
    }

    g_buffers->phases.unmap();
    NvFlexSetPhases(g_solver, g_buffers->phases.buffer, nullptr);
}

void pyflex_print_g_colors() {
    for (int i = 0; i < 63*3; i+=64) {
        std::cout << "g_colors " << i << ": " << g_colors[i][0] << " " << g_colors[i][1] << " " << g_colors[i][2] << std::endl;
    }
}

void pyflex_set_color(py::array_t<float> color) {
    auto buf = color.request();
    auto ptr = (float *) buf.ptr;
    g_colors[(int)ptr[0]] = Colour(ptr[1], ptr[2], ptr[3]);
}

void pyflex_set_colors(int n_particles, py::array_t<float> colors) {
    auto buf = colors.request();
    auto ptr = (float *) buf.ptr;

    for (int i = 0; i < n_particles; i++) {
        g_colors[i] = Colour((float)ptr[i*3], (float)ptr[i*3+1], (float)ptr[i*3+2]);
    }
}

void pyflex_set_fluid_color(py::array_t<float> color) {
    auto buf = color.request();
    auto ptr = (float *) buf.ptr;
    g_fluidColor = Vec4(ptr[0], ptr[1], ptr[2], ptr[3]);
}

void pyflex_reset_colors() {
    Colour colors[] = {
        Colour(0.000f, 0.349f, 0.173f),
        Colour(0.875f, 0.782f, 0.051f),
        Colour(0.000f, 0.170f, 0.453f),
        Colour(0.673f, 0.111f, 0.000f),
        Colour(0.612f, 0.194f, 0.394f),
        Colour(0.0f, 0.5f, 1.0f),
        Colour(0.797f, 0.354f, 0.000f),
        Colour(0.092f, 0.465f, 0.820f)
    };
    for (int i = 0; i < 8; i++) {
        g_colors[i] = colors[i];
    }
}

py::array_t<float> pyflex_get_sceneParams() {
    if (g_scene == 5) {
        auto params = py::array_t<float>(3);
        auto ptr = (float *) params.request().ptr;

        ptr[0] = ((yz_SoftBody *) g_scenes[g_scene])->mInstances[0].mClusterStiffness;
        ptr[1] = ((yz_SoftBody *) g_scenes[g_scene])->mInstances[0].mClusterPlasticThreshold * 1e4f;
        ptr[2] = ((yz_SoftBody *) g_scenes[g_scene])->mInstances[0].mClusterPlasticCreep;

        return params;
    }
    else {
        printf("Unsupprted scene_idx %d\n", g_scene);

        auto params = py::array_t<float>(1);
        auto ptr = (float *) params.request().ptr;
        ptr[0] = 0.0f;

        return params;
    }
}

py::array_t<float> pyflex_get_sceneUpper() {
    auto scene_upper = py::array_t<float>(3);
    auto buf = scene_upper.request();
    auto ptr = (float *) buf.ptr;

    ptr[0] = g_sceneUpper.x;
    ptr[1] = g_sceneUpper.y;
    ptr[2] = g_sceneUpper.z;

    return scene_upper;
}

void pyflex_set_sceneUpper(py::array_t<float> sceneUpper) {
    auto buf = sceneUpper.request();
    auto ptr = (float *) buf.ptr;

    g_sceneUpper = Vec3(ptr[0], ptr[1], ptr[2]);
}

py::array_t<float> pyflex_get_sceneLower() {
    auto scene_lower = py::array_t<float>(3);
    auto buf = scene_lower.request();
    auto ptr = (float *) buf.ptr;

    ptr[0] = g_sceneLower.x;
    ptr[1] = g_sceneLower.y;
    ptr[2] = g_sceneLower.z;

    return scene_lower;
}

void pyflex_set_sceneLower(py::array_t<float> sceneLower) {
    auto buf = sceneLower.request();
    auto ptr = (float *) buf.ptr;

    g_sceneLower = Vec3(ptr[0], ptr[1], ptr[2]);
}

py::array_t<float> pyflex_get_camPos() {
    auto camPos = py::array_t<float>(3);
    auto buf = camPos.request();
    auto ptr = (float *) buf.ptr;

    ptr[0] = g_camPos.x;
    ptr[1] = g_camPos.y;
    ptr[2] = g_camPos.z;

    return camPos;
}

void pyflex_set_camPos(py::array_t<float> camPos) {
    auto buf = camPos.request();
    auto ptr = (float *) buf.ptr;

    g_camPos = Vec3(ptr[0], ptr[1], ptr[2]);
}

py::array_t<float> pyflex_get_camAngle() {
    auto camAngle = py::array_t<float>(3);
    auto buf = camAngle.request();
    auto ptr = (float *) buf.ptr;

    ptr[0] = g_camAngle.x;
    ptr[1] = g_camAngle.y;
    ptr[2] = g_camAngle.z;

    return camAngle;
}

void pyflex_set_camAngle(py::array_t<float> camAngle) {
    auto buf = camAngle.request();
    auto ptr = (float *) buf.ptr;

    g_camAngle = Vec3(ptr[0], ptr[1], ptr[2]);
}

py::array_t<float> pyflex_get_viewMatrix() {
    auto viewMatrix = py::array_t<float>(16);
    auto buf = viewMatrix.request();
    auto ptr = (float *) buf.ptr;

    Matrix44 view = RotationMatrix(-g_camAngle.x, Vec3(0.0f, 1.0f, 0.0f))*RotationMatrix(-g_camAngle.y, Vec3(cosf(-g_camAngle.x), 0.0f, sinf(-g_camAngle.x)))*TranslationMatrix(-Point3(g_camPos));

    int cnt = 0;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            ptr[cnt++] = view.columns[i][j];
        }
    }

    return viewMatrix;
}

py::array_t<float> pyflex_get_projMatrix() {
    auto projMatrix = py::array_t<float>(16);
    auto buf = projMatrix.request();
    auto ptr = (float *) buf.ptr;

    float fov = kPi / 4.0f;
    float aspect = float(g_screenWidth) / g_screenHeight;
    Matrix44 proj = ProjectionMatrix(RadToDeg(fov), aspect, g_camNear, g_camFar);

    int cnt = 0;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            ptr[cnt++] = proj.columns[i][j];
        }
    }

    return projMatrix;
}

void pyflex_set_floorScaleSize(float kSize) {
    g_floorScaleSize = kSize;
}

float pyflex_get_floorScaleSize() {
    return g_floorScaleSize;
}

void pyflex_set_screenWidth(int w) {
    g_screenWidth = w;
}

void pyflex_set_screenHeight(int h) {
    g_screenHeight = h;
}

int pyflex_get_screenWidth() {
    return g_screenWidth;
}

int pyflex_get_screenHeight() {
    return g_screenHeight;
}

void pyflex_set_hideShapes(py::array_t<float> hideShapes) {
    auto buf = hideShapes.request();
    auto ptr = (float *) buf.ptr;

    for (int i = 0; i < g_buffers->shapeFlags.size(); ++i) {
        g_hideShapes[i] = (int)ptr[i];
    }
}

void pyflex_set_lightDir(py::array_t<float> lightDir) {
    auto buf = lightDir.request();
    auto ptr = (float *) buf.ptr;

    g_lightDir = Vec3(ptr[0], ptr[1], ptr[2]);
}

py::array_t<float> pyflex_get_lightDir() {
    auto lightDir = py::array_t<float>(3);
    auto buf = lightDir.request();
    auto ptr = (float *) buf.ptr;

    ptr[0] = g_lightDir.x;
    ptr[1] = g_lightDir.y;
    ptr[2] = g_lightDir.z;

    return lightDir;
}

void pyflex_set_lightFov(float lightFov) {
    g_lightFov = lightFov;
}

float pyflex_get_lightFov() {
    return g_lightFov;
}

// Added by Baoyu 06/02/2024
py::array_t<int> pyflex_get_clusters() {
    g_buffers->clusters.map();

    auto clusters = py::array_t<int>((size_t) g_buffers->clusters.size());
    auto ptr = (int *) clusters.request().ptr;

    for (size_t i = 0; i < (size_t) g_buffers->clusters.size(); i++) {
        ptr[i] = g_buffers->clusters[i];
    }

    g_buffers->clusters.unmap();

    return clusters;
}

py::array_t<float> pyflex_render(
        int draw_planes, int draw_objects, int draw_shadow, bool render_depth) {

    static double lastTime;

    // real elapsed frame time
    double frameBeginTime = GetSeconds();

    g_realdt = float(frameBeginTime - lastTime);
    lastTime = frameBeginTime;

    //-------------------------------------------------------------------
    // Scene Update

    double waitBeginTime = GetSeconds();

    MapBuffers(g_buffers);

    double waitEndTime = GetSeconds();

    // Getting timers causes CPU/GPU sync, so we do it after a map
    float newSimLatency = NvFlexGetDeviceLatency(g_solver, &g_GpuTimers.computeBegin, &g_GpuTimers.computeEnd, &g_GpuTimers.computeFreq);
    float newGfxLatency = RendererGetDeviceTimestamps(&g_GpuTimers.renderBegin, &g_GpuTimers.renderEnd, &g_GpuTimers.renderFreq);
    (void)newGfxLatency;

    UpdateCamera();

    if (!g_pause || g_step) {
        UpdateEmitters();
        UpdateMouse();
        UpdateWind();
        // UpdateScene();
    }

    //-------------------------------------------------------------------
    // Render

    double renderBeginTime = GetSeconds();

    if (g_profile && (!g_pause || g_step)) {
        if (g_benchmark) {
            g_numDetailTimers = NvFlexGetDetailTimers(g_solver, &g_detailTimers);
        }
        else {
            memset(&g_timers, 0, sizeof(g_timers));
            NvFlexGetTimers(g_solver, &g_timers);
        }
    }

    StartFrame(Vec4(g_clearColor, 1.0f));

    // main scene render
    RenderScene(draw_planes, draw_objects, draw_shadow);
    RenderDebug();

    int newScene = DoUI();

    EndFrame();

    // If user has disabled async compute, ensure that no compute can overlap
    // graphics by placing a sync between them
    if (!g_useAsyncCompute)
        NvFlexComputeWaitForGraphics(g_flexLib);

    UnmapBuffers(g_buffers);

    // move mouse particle (must be done here as GetViewRay() uses the GL projection state)
    if (g_mouseParticle != -1) {
        Vec3 origin, dir;
        GetViewRay(g_lastx, g_screenHeight - g_lasty, origin, dir);

        g_mousePos = origin + dir*g_mouseT;
    }

    if (!g_interop) {
        // if not using interop then we read back fluid data to host
        if (g_drawEllipsoids) {
            NvFlexGetSmoothParticles(g_solver, g_buffers->smoothPositions.buffer, nullptr);
            NvFlexGetAnisotropy(g_solver, g_buffers->anisotropy1.buffer, g_buffers->anisotropy2.buffer,
                                g_buffers->anisotropy3.buffer, NULL);
        }

        // read back diffuse data to host
        if (g_drawDensity)
            NvFlexGetDensities(g_solver, g_buffers->densities.buffer, nullptr);

        if (GetNumDiffuseRenderParticles(g_diffuseRenderBuffers)) {
            NvFlexGetDiffuseParticles(g_solver, g_buffers->diffusePositions.buffer, g_buffers->diffuseVelocities.buffer,
                                      g_buffers->diffuseCount.buffer);
        }
    } else {
        // read back just the new diffuse particle count, render buffers will be updated during rendering
        NvFlexGetDiffuseParticles(g_solver, nullptr, nullptr, g_buffers->diffuseCount.buffer);
    }



    double renderEndTime = GetSeconds();

    // if user requested a scene reset process it now
    if (g_resetScene) {
        // Reset();
        g_resetScene = false;
    }

    //-------------------------------------------------------------------
    // Flex Update

    double updateBeginTime = GetSeconds();

    // send any particle updates to the solver
    NvFlexSetParticles(g_solver, g_buffers->positions.buffer, nullptr);
    NvFlexSetVelocities(g_solver, g_buffers->velocities.buffer, nullptr);
    NvFlexSetPhases(g_solver, g_buffers->phases.buffer, nullptr);
    NvFlexSetActive(g_solver, g_buffers->activeIndices.buffer, nullptr);

    NvFlexSetActiveCount(g_solver, g_buffers->activeIndices.size());

    if (!g_pause || g_step) {
        // tick solver
        // NvFlexSetParams(g_solver, &g_params);
        // NvFlexUpdateSolver(g_solver, g_dt, g_numSubsteps, g_profile);

        g_frame++;
        g_step = false;
    }

    // read back base particle data
    // Note that flexGet calls don't wait for the GPU, they just queue a GPU copy
    // to be executed later.
    // When we're ready to read the fetched buffers we'll Map them, and that's when
    // the CPU will wait for the GPU flex update and GPU copy to finish.
    NvFlexGetParticles(g_solver, g_buffers->positions.buffer, nullptr);
    NvFlexGetVelocities(g_solver, g_buffers->velocities.buffer, nullptr);
    NvFlexGetNormals(g_solver, g_buffers->normals.buffer, nullptr);

    // readback rigid transforms
    // if (g_buffers->rigidOffsets.size())
    //    NvFlexGetRigids(g_solver, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, g_buffers->rigidRotations.buffer, g_buffers->rigidTranslations.buffer);

    double updateEndTime = GetSeconds();

    //-------------------------------------------------------
    // Update the on-screen timers

    auto newUpdateTime = float(updateEndTime - updateBeginTime);
    auto newRenderTime = float(renderEndTime - renderBeginTime);
    auto newWaitTime = float(waitBeginTime - waitEndTime);

    // Exponential filter to make the display easier to read
    const float timerSmoothing = 0.05f;

    g_updateTime = (g_updateTime == 0.0f) ? newUpdateTime : Lerp(g_updateTime, newUpdateTime, timerSmoothing);
    g_renderTime = (g_renderTime == 0.0f) ? newRenderTime : Lerp(g_renderTime, newRenderTime, timerSmoothing);
    g_waitTime = (g_waitTime == 0.0f) ? newWaitTime : Lerp(g_waitTime, newWaitTime, timerSmoothing);
    g_simLatency = (g_simLatency == 0.0f) ? newSimLatency : Lerp(g_simLatency, newSimLatency, timerSmoothing);

    if (g_benchmark) newScene = BenchmarkUpdate();

    // flush out the last frame before freeing up resources in the event of a scene change
    // this is necessary for d3d12
    PresentFrame(g_vsync);

    // if gui or benchmark requested a scene change process it now
    if (newScene != -1) {
        g_scene = newScene;
        // Init(g_scene);
    }

    SDL_EventFunc();


    // capture rgb
    int num_ch;
    if (render_depth) {
        num_ch = 5;
    } else {
        num_ch = 4;
    }
    auto rgb_data = new unsigned char[g_screenWidth * g_screenHeight * 4];
    auto depth_data = new float[g_screenWidth * g_screenHeight];
    float z_near = g_camNear;
    float z_far = g_camFar;

    ReadFrame((int *)rgb_data, g_screenWidth, g_screenHeight);
    ReadDepth((float *)depth_data, g_screenWidth, g_screenHeight);

    auto img = py::array_t<float>((size_t) g_screenWidth * g_screenHeight * num_ch);
    auto ptr = (float *) img.request().ptr;

    for (int i = 0; i < g_screenHeight; ++i) {
        for (int j = 0; j < g_screenWidth; ++j) {
            for (int k = 0; k < 3; ++k) {
                int idx_src = i * g_screenWidth * 4 + j * 4 + k;
                int idx_des = (g_screenHeight - i - 1) * g_screenWidth * num_ch + j * num_ch + k;
                ptr[idx_des] = rgb_data[idx_src];
            }
            if (render_depth) {
                int idx_des = (g_screenHeight - i - 1) * g_screenWidth * num_ch + j * num_ch + num_ch - 1;
                float z = depth_data[i * g_screenWidth + j];
                z = (2.0*z)-1.0;
                ptr[idx_des] = ( 2.0 * z_near * z_far) / ( z_far + z_near - z * ( z_far - z_near ) );
                // ptr[idx_des] = depth_data[i * g_screenWidth + j];
            }
        }
    }

    delete[] rgb_data;
    delete[] depth_data;

    return img;
}


PYBIND11_MODULE(pyflex, m) {
    m.def("main", &main);

    m.def("init", &pyflex_init);
    // m.def("set_scene", &pyflex_set_scene);
    m.def("set_scene", &pyflex_set_scene,
          py::arg("scene_idx") = 0,
          py::arg("scene_params") = py::array_t<float>(),
          py::arg("vertices") = py::array_t<float>(),
          py::arg("stretch_edges") = py::array_t<int>(),
          py::arg("bend_edges") = py::array_t<int>(),
          py::arg("shear_edges") = py::array_t<int>(),
          py::arg("faces") = py::array_t<int>(),
          py::arg("thread_idx") = 0);
    m.def("clean", &pyflex_clean);
    
    m.def("step", &pyflex_step,
          py::arg("update_params") = nullptr,
          py::arg("draw_planes") = 1,
          py::arg("draw_objects") = 1,
          py::arg("draw_shadow") = 1,
          py::arg("render_depth") = 0);
    
    m.def("render", &pyflex_render,
          py::arg("draw_planes") = 1,
          py::arg("draw_objects") = 1,
          py::arg("draw_shadow") = 1,
          py::arg("render_depth") = 0);

    m.def("add_box", &pyflex_add_box, "Add box to the scene");
    m.def("add_mesh", &pyflex_add_mesh, "Add mesh to the scene");
    m.def("add_capsule", &pyflex_add_capsule, "Add capsule to the scene");
    m.def("add_sphere", &pyflex_add_sphere, "Add sphere to the scene");

    m.def("get_n_particles", &pyflex_get_n_particles, "Get the number of particles");
    m.def("get_n_shapes", &pyflex_get_n_shapes, "Get the number of shapes");
    m.def("get_n_rigids", &pyflex_get_n_rigids, "Get the number of rigids");
    m.def("get_n_rigidPositions", &pyflex_get_n_rigidPositions, "Get the number of rigid positions");

    m.def("get_positions", &pyflex_get_positions, "Get particle positions");
    m.def("set_positions", &pyflex_set_positions, "Set particle positions");

    m.def("get_edges", &pyflex_get_edges, "Get mesh edges");
    m.def("get_faces", &pyflex_get_faces, "Get mesh faces");

    m.def("get_particle_2_obj_instance", &pyflex_get_particle_2_obj_instance, "Get object instance index for corresponding particle");

    m.def("get_restPositions", &pyflex_get_restPositions, "Get particle restPositions");
    m.def("get_rigidOffsets", &pyflex_get_rigidOffsets, "Get rigid offsets");
    m.def("get_rigidIndices", &pyflex_get_rigidIndices, "Get rigid indices");
    m.def("get_rigidLocalPositions", &pyflex_get_rigidLocalPositions, "Get rigid local positions");
    m.def("get_rigidGlobalPositions", &pyflex_get_rigidGlobalPositions, "Get rigid global positions");
    m.def("get_rigidRotations", &pyflex_get_rigidRotations, "Get rigid rotations");
    m.def("get_rigidTranslations", &pyflex_get_rigidTranslations, "Get rigid translations");

    m.def("get_sceneParams", &pyflex_get_sceneParams, "Get scene parameters");

    m.def("get_velocities", &pyflex_get_velocities, "Get particle velocities");
    m.def("set_velocities", &pyflex_set_velocities, "Set particle velocities");
    m.def("get_phases", &pyflex_get_phases, "Get particle phases");
    m.def("set_phases", &pyflex_set_phases, "Set particle phases");

    m.def("print_g_colors", &pyflex_print_g_colors, "Print current g_colors");
    m.def("set_color", &pyflex_set_color, "Set color");
    m.def("set_colors", &pyflex_set_colors, "Set colors");
    m.def("reset_colors", &pyflex_reset_colors, "Reset g_color to original");

    m.def("set_fluid_color", &pyflex_set_fluid_color, "Set fluid color");

    m.def("get_shape_states", &pyflex_get_shape_states, "Get shape states");
    m.def("set_shape_states", &pyflex_set_shape_states, "Set shape states");
    m.def("clear_shapes", &ClearShapes, "Clear shapes");
    m.def("set_hideShapes", &pyflex_set_hideShapes, "Set hideShapes");

    m.def("get_scene_upper", &pyflex_get_sceneUpper);
    m.def("set_scene_upper", &pyflex_set_sceneUpper);
    m.def("get_scene_lower", &pyflex_get_sceneLower);
    m.def("set_scene_lower", &pyflex_set_sceneLower);

    m.def("get_light_dir", &pyflex_get_lightDir);
    m.def("set_light_dir", &pyflex_set_lightDir);
    m.def("get_light_fov", &pyflex_get_lightFov);
    m.def("set_light_fov", &pyflex_set_lightFov);

    m.def("get_camPos", &pyflex_get_camPos);
    m.def("set_camPos", &pyflex_set_camPos);
    m.def("get_camAngle", &pyflex_get_camAngle);
    m.def("set_camAngle", &pyflex_set_camAngle);

    m.def("get_projMatrix", &pyflex_get_projMatrix);
    m.def("get_viewMatrix", &pyflex_get_viewMatrix);

    m.def("get_screenWidth", &pyflex_get_screenWidth);
    m.def("get_screenHeight", &pyflex_get_screenHeight);

    m.def("set_screenWidth", &pyflex_set_screenWidth);
    m.def("set_screenHeight", &pyflex_set_screenHeight);

    m.def("set_floorScaleSize", &pyflex_set_floorScaleSize);
    m.def("get_floorScaleSize", &pyflex_get_floorScaleSize);

    m.def("get_clusters", &pyflex_get_clusters);
}
