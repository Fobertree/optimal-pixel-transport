#include <iostream>
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
// GLFW wasm32-emscripten triplet
#include <GLFW/glfw3.h>

#if defined(__EMSCRIPTEN__)

#include <emscripten/emscripten.h>
#include <emscripten/html5.h>

#endif

#include <dawn/webgpu_cpp_print.h>
#include <webgpu/webgpu_cpp.h>
#include <webgpu/webgpu_glfw.h>

#include "declare.h"
#include "readfile.h"
#include "buffermanager.h"
#include "particle.h"
#include "particle_buffer.h"
#include "consts.h"

#define WATCH(x) (std::cout << #x << " = " << (x) << std::endl)

// tmp spaghetti var bc gcc doesn't detect changes/re-compile on wgsl-only changes
bool bob = true;

template<typename T>
using vec2 = std::array<T, 2>;

wgpu::Instance instance;
wgpu::Adapter adapter;
wgpu::Device device;
wgpu::Surface surface; // similar to html canvas
wgpu::TextureFormat format;
wgpu::RenderPipeline pipeline;

wgpu::Queue queue;

wgpu::Buffer indexBuffer;
// global
wgpu::Buffer particleBuffer;
wgpu::Buffer paramsBuffer;
// solver
wgpu::Buffer assignmentsBuffer;
wgpu::Buffer pricesBuffer;
wgpu::Buffer bidValueBuffer;
wgpu::Buffer bidFromRowBuffer;
wgpu::Buffer sortIndicesBuffer;
wgpu::Buffer sortIndicesAltBuffer;
wgpu::Buffer ownerBuffer;
// pbf
wgpu::Buffer lambdasBuffer;
wgpu::Buffer deltaPosBuffer;
wgpu::Buffer posStarBuffer;
wgpu::Buffer binStartBuffer;
wgpu::Buffer binEndBuffer;
wgpu::Buffer omegaBuffer;
wgpu::Buffer targetParticleBuffer;
// radix
wgpu::Buffer localPrefixSumBuffer;
wgpu::Buffer prefixBlockSumBuffer;
wgpu::Buffer outputHashBuffer;

wgpu::ComputePipeline solverBiddingPipeline, solverUpdatePipeline, solverInitPipeline, solverSyncPipeline,
        solverRepairPipeline;
wgpu::ComputePipeline solverBiddingConvergePipeline, solverUpdateConvergePipeline, solverRepairConvergePipeline;
bool g_matchingConverged = false;
int32_t g_convergeFramesLeft = 0;
std::vector<wgpu::ComputePipeline> radixSortPipelines, radixReorderPipelines;
wgpu::ComputePipeline radixScanPipeline;
wgpu::ComputePipeline physicsExternalForcesPipeline, physicsClearBinsPipeline, physicsBuildBinsPipeline;
wgpu::ComputePipeline physicsSolverOnePipeline, physicsSolverTwoPipeline, physicsSolverThreePipeline;
wgpu::RenderPipeline renderPipeline;

wgpu::BindGroup globalComputeBG, globalRenderBG, solverBG, physicsBG, radixBG, radixBGAlt;

BufferManager bufferManager;

wgpu::CommandEncoder encoder;

std::vector<uint16_t> indexData = {
        0, 1, 3,
        1, 2, 3
};

struct Params {
    int32_t size;
    // C++ standard doesn't guarantee 32-bit float
    // TODO: actually init the values
    float rho0;
    float H;
    float dt;
    uint32_t solverIterations;
    float cellSize;
    uint32_t numBins;
};

int32_t DIM, NUM_PARTICLES;
uint32_t NUM_BINS;
uint32_t RADIX_WORKGROUP_COUNT;
int32_t RADIX_PASS_COUNT;
uint32_t COMPUTE_WORKGROUPS;
int32_t AUCTION_ITERS_PER_FRAME;
int32_t AUCTION_EPSILON;
int32_t AUCTION_CANDIDATE_RADIUS;

EM_BOOL frame_callback(double time, void *userData) {
    static double last_time = 0.0;
    static int frame_count = 0;

    double dt = (time - last_time) / 1000.0;
    frame_count++;

    if (dt > 1.0) {
        double fps = frame_count / dt;
        printf("FPS: %.2f\n", fps);

        frame_count = 0;
        last_time = time;
    }

    return EM_TRUE;
}

[[nodiscard]] int32_t auctionIterationsPerFrame(int32_t gridDim, int32_t candidateRadius) {
    (void) gridDim;
    (void) candidateRadius;
    return 8;
}

[[nodiscard]] int32_t auctionEpsilon(int32_t numParticles) {
    // Integer costs with min gap 1 (tiebreak). Bertsekas: ε ∈ (0, min_cost_gap] for convergence;
    // ε ≤ 1/n for ε-optimal matching. Use 1 — safe since rgb gaps >> n for typical grids.
    (void) numParticles;
    return 1;
}

[[nodiscard]] int32_t auctionCandidateRadius(int32_t gridDim) {
    // R >= GRID_DIM enables full scan (required for RGB matching across the whole image).
    return gridDim * 2;
}

[[nodiscard]] int32_t auctionInitIterations(int32_t numParticles) {
    return std::clamp(numParticles * 4, 256, 4096);
}

[[nodiscard]] int32_t auctionRepairRounds(int32_t numParticles) {
    return std::clamp(numParticles / 4, 16, 64);
}

[[nodiscard]] int32_t auctionRepairRoundsPerFrame() {
    return 4;
}

[[nodiscard]] int32_t auctionConvergenceFrames(int32_t numParticles) {
    return std::clamp(numParticles / 2, 30, 180);
}

[[nodiscard]] uint32_t radixWorkgroupsForParticles(int32_t numParticles) {
    constexpr uint32_t kThreadsPerRadixWg = 128u;
    return std::max(1u, (static_cast<uint32_t>(numParticles) + kThreadsPerRadixWg - 1u) / kThreadsPerRadixWg);
}

[[nodiscard]] uint32_t spatialBinsForParticles(int32_t numParticles) {
    return std::clamp(static_cast<uint32_t>((numParticles + 15) / 16), 16u, 128u);
}

void updateDispatchConstants() {
    COMPUTE_WORKGROUPS = (static_cast<uint32_t>(NUM_PARTICLES) + kTileSize - 1u) / kTileSize;
    AUCTION_CANDIDATE_RADIUS = auctionCandidateRadius(DIM);
    AUCTION_ITERS_PER_FRAME = auctionIterationsPerFrame(DIM, AUCTION_CANDIDATE_RADIUS);
    AUCTION_EPSILON = auctionEpsilon(NUM_PARTICLES);
}

[[nodiscard]] int32_t radixPassesForBins(uint32_t numBins) {
    if (numBins <= 1u) {
        return 1;
    }
    int32_t bits = 0;
    auto v = numBins - 1u;
    while (v > 0u) {
        ++bits;
        v >>= 1u;
    }
    int32_t passes = (bits + 1) / 2;
    if (passes % 2 != 0) {
        ++passes;
    }
    return passes;
}

auto solverComputePass = [](const wgpu::ComputePipeline &pipeline) {
    auto pass = encoder.BeginComputePass();
    pass.SetPipeline(pipeline);
    pass.SetBindGroup(0, globalComputeBG);
    pass.SetBindGroup(1, solverBG);
    pass.DispatchWorkgroups(COMPUTE_WORKGROUPS);
    pass.End();
};

auto solverConvergePass = [](int32_t repairRounds, const wgpu::ComputePipeline &repairPipeline) {
    for (int32_t i = 0; i < repairRounds; ++i) {
        solverComputePass(solverSyncPipeline);
        solverComputePass(repairPipeline);
    }
};

auto runAuctionRounds = [](int32_t rounds, const wgpu::ComputePipeline &bidding,
                           const wgpu::ComputePipeline &update) {
    for (int32_t i = 0; i < rounds; ++i) {
        solverComputePass(bidding);
        solverComputePass(update);
    }
};

auto physicsComputePass = [](const wgpu::ComputePipeline &pipeline) {
    auto pass = encoder.BeginComputePass();
    pass.SetPipeline(pipeline);
    pass.SetBindGroup(0, globalComputeBG);
    pass.SetBindGroup(1, physicsBG);
    pass.DispatchWorkgroups(COMPUTE_WORKGROUPS);
    pass.End();
};

void setDim(uint16_t dim) {
    DIM = dim;
    NUM_PARTICLES = DIM * DIM;
    updateDispatchConstants();

    WATCH(DIM);
    WATCH(NUM_PARTICLES);
    WATCH(COMPUTE_WORKGROUPS);
    WATCH(AUCTION_ITERS_PER_FRAME);
    WATCH(AUCTION_EPSILON);
    WATCH(AUCTION_CANDIDATE_RADIUS);
}

void Start() {
//    list_all_paths();
    if (!glfwInit()) {
        return;
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    GLFWwindow *window = glfwCreateWindow(kWidth, kHeight, "Optimal Pixel Transport", nullptr, nullptr);

    surface = wgpu::glfw::CreateSurfaceForWindow(instance, window);
    InitGraphics();

#if defined (__EMSCRIPTEN__)
    emscripten_set_main_loop(Render, 0, false);
    emscripten_request_animation_frame_loop(frame_callback, nullptr);
#else
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        Render();
        surface.Present();
        instance.ProcessEvents();
    }
#endif
}

void Init() {
    // TimedWaitAny flag - enable timeout on waiting for GPU tasks
    static const auto kTimedWaitAny = wgpu::InstanceFeatureName::TimedWaitAny;
    wgpu::InstanceDescriptor instanceDesc{.requiredFeatureCount = 1,
            .requiredFeatures = &kTimedWaitAny};
    instance = wgpu::CreateInstance(&instanceDesc);

    // request adapter and device
    wgpu::Future f1 = instance.RequestAdapter(
            nullptr, wgpu::CallbackMode::WaitAnyOnly,
            [](wgpu::RequestAdapterStatus status, wgpu::Adapter a,
               wgpu::StringView message) {
                if (status != wgpu::RequestAdapterStatus::Success) {
                    std::cout << "RequestAdapter: " << message << "\n";
                    exit(0);
                }
                adapter = std::move(a);
            });
    instance.WaitAny(f1, UINT64_MAX);

    static const long long WEBGPU_MAX_LIMIT = 4294967292LL;

    wgpu::DeviceDescriptor desc{};
    wgpu::Limits limits;
    limits.maxStorageBuffersPerShaderStage = 10;
    limits.maxBufferSize = WEBGPU_MAX_LIMIT;
    limits.maxStorageBufferBindingSize = WEBGPU_MAX_LIMIT;
    desc.requiredLimits = &limits;

    desc.SetUncapturedErrorCallback([](const wgpu::Device &,
                                       wgpu::ErrorType errorType,
                                       wgpu::StringView message) {
        std::cout << "Error: " << errorType << " - message: " << message << "\n";
    });

    wgpu::Future f2 = adapter.RequestDevice(
            &desc, wgpu::CallbackMode::WaitAnyOnly,
            [](wgpu::RequestDeviceStatus status, wgpu::Device d, wgpu::StringView message) {
                if (status != wgpu::RequestDeviceStatus::Success) {
                    std::cout << "RequestDevice: " << message << "\n";
                    exit(0);
                }
                device = std::move(d);
            });
    instance.WaitAny(f2, UINT64_MAX);

#ifdef ERROR_SCOPE
    // for debugging
    puts("error validation");
    device.PushErrorScope(wgpu::ErrorFilter::Validation);

    auto f3 = device.PopErrorScope(
            wgpu::CallbackMode::WaitAnyOnly,
            [](wgpu::PopErrorScopeStatus status,
               wgpu::ErrorType type,
               wgpu::StringView message) {

                if (type != wgpu::ErrorType::NoError) {
                    std::cout << "Validation error: "
                              << message << "\n";
                }
            });

    instance.WaitAny(f3, UINT64_MAX);
#endif
    // create commandQueue
    queue = device.GetQueue();

    bufferManager = BufferManager(device, queue);
}

void ConfigureSurface() {
    wgpu::SurfaceCapabilities capabilities;
    surface.GetCapabilities(adapter, &capabilities);
    format = capabilities.formats[0];

    wgpu::SurfaceConfiguration config{
            .device = device,
            .format = format,
            .width = kWidth,
            .height = kHeight
    };

    surface.Configure(&config);
}

void CreateRenderPipeline() {
    auto particleCPUData = loadSourceParticlesFromImage("img_1.png", DIM, DIM);
    auto targetParticleCPUData = loadTargetParticlesFromImage("img_6.png", DIM, DIM);

    RADIX_WORKGROUP_COUNT = radixWorkgroupsForParticles(NUM_PARTICLES);
    NUM_BINS = spatialBinsForParticles(NUM_PARTICLES);
    RADIX_PASS_COUNT = radixPassesForBins(NUM_BINS);

    constexpr float H = 0.1f;
    Params params{.size = NUM_PARTICLES,
            .rho0 = 0.5f,
            .H = H,
            .dt = 1.f / 60.f,
            .solverIterations = 0,
            .cellSize = H,
            .numBins = NUM_BINS,
    };

    /* Load Shader Modules */
    // lambda capture only involves automatic storage objects, global variables do not need to be captured for access
    auto getShaderModule = [](const std::string &shaderPath) {
        std::string shaderCode = read_wgsl_file(shaderPath);
        wgpu::ShaderSourceWGSL shaderSourceWgsl{{.code = shaderCode.c_str()}};
        wgpu::ShaderModuleDescriptor shaderModuleDescriptor{.nextInChain = &shaderSourceWgsl};
        return device.CreateShaderModule(&shaderModuleDescriptor);
    };

    wgpu::ShaderModule solverShaderModule = getShaderModule("Solver/auction.wgsl");
    wgpu::ShaderModule radixSortShaderModule = getShaderModule("Sort/radix.wgsl");
    wgpu::ShaderModule radixReorderShaderModule = getShaderModule("Sort/radix_reorder.wgsl");
    wgpu::ShaderModule radixScanShaderModule = getShaderModule("Sort/radix_scan.wgsl");
    wgpu::ShaderModule physicsShaderModule = getShaderModule("Physics/pbf.wgsl");
    // TODO: update this
    wgpu::ShaderModule renderShaderModule = getShaderModule("Render/particle_shader.wgsl");

    using std::cout, std::endl;

    /* Create Buffers */
    // index buffer
    indexBuffer = bufferManager.createWGPUBuffer<uint32_t>(
            wgpu::BufferUsage::Index | wgpu::BufferUsage::CopyDst, 6);

    /* Globals */
    // particle buffer
    particleBuffer = bufferManager.createWGPUBuffer<ParticleCPU>(wgpu::BufferUsage::Storage |
                                                                 wgpu::BufferUsage::CopyDst, NUM_PARTICLES,
                                                                 "particle buffer");
    // params
    wgpu::BufferDescriptor paramsDesc{};
    paramsDesc.size = sizeof(Params);
    paramsDesc.usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst;
    paramsDesc.label = "params";
    paramsBuffer = device.CreateBuffer(&paramsDesc);

    cout << "GLOBALS" << endl;

    /* Solver */

    // assignments buffer (one entry per source particle / row)
    assignmentsBuffer = bufferManager.createWGPUBuffer<int32_t>(
            wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst, NUM_PARTICLES, "assignments");
    // prices buffer
    pricesBuffer = bufferManager.createWGPUBuffer<int32_t>(
            wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst, NUM_PARTICLES, "prices");
    // bid value buffer
    bidValueBuffer = bufferManager.createWGPUBuffer<int32_t>(
            wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst, NUM_PARTICLES, "bid value");

    bidFromRowBuffer = bufferManager.createWGPUBuffer<int32_t>(
            wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst, NUM_PARTICLES, "bid from row");

    ownerBuffer = bufferManager.createWGPUBuffer<int32_t>(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst,
                                                          NUM_PARTICLES, "owner");

    /* Radix */
    localPrefixSumBuffer = bufferManager.createWGPUBuffer<uint32_t>(
            wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst,
            NUM_PARTICLES, "local prefix");
    prefixBlockSumBuffer = bufferManager.createWGPUBuffer<uint32_t>(
            wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst,
            4 * RADIX_WORKGROUP_COUNT, "prefix block sum");

    outputHashBuffer = bufferManager.createWGPUBuffer<uint32_t>(
            wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst,
            NUM_PARTICLES, "hash");

    sortIndicesBuffer = bufferManager.createWGPUBuffer<uint32_t>(
            wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst,
            NUM_PARTICLES, "sortIndices");
    sortIndicesAltBuffer = bufferManager.createWGPUBuffer<uint32_t>(
            wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst,
            NUM_PARTICLES, "sortIndicesAlt");
    // PBF/Physics
    lambdasBuffer = bufferManager.createWGPUBuffer<float>(
            wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst,
            NUM_PARTICLES, "lambda");
    deltaPosBuffer = bufferManager.createWGPUBuffer<vec2<float>>(
            wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst,
            NUM_PARTICLES, "deltaPos");
    posStarBuffer = bufferManager.createWGPUBuffer<vec2<float>>(
            wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst,
            NUM_PARTICLES, "posStar");
    binStartBuffer = bufferManager.createWGPUBuffer<uint32_t>(
            wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst,
            NUM_BINS, "binStart");
    binEndBuffer = bufferManager.createWGPUBuffer<uint32_t>(
            wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst,
            NUM_BINS, "binEnd");
    omegaBuffer = bufferManager.createWGPUBuffer<float>(
            wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst,
            NUM_PARTICLES, "omega");

    targetParticleBuffer = bufferManager.createWGPUBuffer<TargetParticleCPU>(
            wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst,
            NUM_PARTICLES, "targetParticle");

    /* Bind Group Layouts */
    // might be uglier than direct struct brace initialization
    // Group 0: Global Bind Group
    std::array<wgpu::BindGroupLayoutEntry, 2> globalEntries{};
    // TODO: this code is so redundant - clean it up
    // TODO: move the buffer variadic args into arrays so we have common set of params

    // webgpu spec prohibits storage from being visible to vertex (must be read_only)
    // Must create two BGLs
    // particles
    globalEntries[0].binding = 0;
    globalEntries[0].visibility = wgpu::ShaderStage::Compute;
    globalEntries[0].buffer.type = wgpu::BufferBindingType::Storage;
    globalEntries[0].buffer.minBindingSize = NUM_PARTICLES * sizeof(ParticleCPU);

    // params
    globalEntries[1].binding = 1;
    globalEntries[1].visibility = wgpu::ShaderStage::Compute;
    globalEntries[1].buffer.type = wgpu::BufferBindingType::Uniform;
    globalEntries[1].buffer.minBindingSize = sizeof(Params);

    auto globalComputeBGL = bufferManager.createBGL(globalEntries, "computeRenderBGL");
    // render shader version
    // not sure if copy necessary but cheap anyway, ugly code
    auto globalRenderEntries = globalEntries;

    globalRenderEntries[0].visibility = wgpu::ShaderStage::Vertex | wgpu::ShaderStage::Fragment;
    globalRenderEntries[0].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;

    globalRenderEntries[0].visibility = wgpu::ShaderStage::Vertex | wgpu::ShaderStage::Fragment;

    auto globalRenderBGL = bufferManager.createBGL(globalRenderEntries, "globalRenderBGL");

    // Group: Solver BindGroup (assignments, cost, prices, bid value, bid from row)
    std::array<wgpu::BindGroupLayoutEntry, 6> solverEntries{};

    // assignments
    solverEntries[0].binding = 0;
    solverEntries[0].visibility = wgpu::ShaderStage::Compute;
    solverEntries[0].buffer.type = wgpu::BufferBindingType::Storage;
    solverEntries[0].buffer.minBindingSize = NUM_PARTICLES * sizeof(int32_t);

    // target particles (live cost lookup)
    solverEntries[1].binding = 1;
    solverEntries[1].visibility = wgpu::ShaderStage::Compute;
    solverEntries[1].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
    solverEntries[1].buffer.minBindingSize = NUM_PARTICLES * sizeof(TargetParticleCPU);

    // prices
    solverEntries[2].binding = 2;
    solverEntries[2].visibility = wgpu::ShaderStage::Compute;
    solverEntries[2].buffer.type = wgpu::BufferBindingType::Storage;
    solverEntries[2].buffer.minBindingSize = NUM_PARTICLES * sizeof(int32_t);

    // bid value
    solverEntries[3].binding = 3;
    solverEntries[3].visibility = wgpu::ShaderStage::Compute;
    solverEntries[3].buffer.type = wgpu::BufferBindingType::Storage;
    solverEntries[3].buffer.minBindingSize = NUM_PARTICLES * sizeof(int32_t);

    // bid from row
    solverEntries[4].binding = 4;
    solverEntries[4].visibility = wgpu::ShaderStage::Compute;
    solverEntries[4].buffer.type = wgpu::BufferBindingType::Storage;
    solverEntries[4].buffer.minBindingSize = NUM_PARTICLES * sizeof(int32_t);

    // owner buffer
    solverEntries[5].binding = 5;
    solverEntries[5].visibility = wgpu::ShaderStage::Compute;
    solverEntries[5].buffer.type = wgpu::BufferBindingType::Storage;
    solverEntries[5].buffer.minBindingSize = NUM_PARTICLES * sizeof(int32_t);

    auto solverBGL = bufferManager.createBGL(solverEntries, "solverBGL");

    // Group: Physics (Unique PBF buffers)
    // TODO: migrate to getBGLayoutEntries - binding and visibility are repetitive
    std::array<wgpu::BindGroupLayoutEntry, 9> physicsEntries;
    // lambda
    physicsEntries[0].binding = 0;
    physicsEntries[0].visibility = wgpu::ShaderStage::Compute;
    physicsEntries[0].buffer.type = wgpu::BufferBindingType::Storage;

    // deltaPos
    physicsEntries[1].binding = 1;
    physicsEntries[1].visibility = wgpu::ShaderStage::Compute;
    physicsEntries[1].buffer.type = wgpu::BufferBindingType::Storage;

    // posStar
    physicsEntries[2].binding = 2;
    physicsEntries[2].visibility = wgpu::ShaderStage::Compute;
    physicsEntries[2].buffer.type = wgpu::BufferBindingType::Storage;

    // binStart
    physicsEntries[3].binding = 3;
    physicsEntries[3].visibility = wgpu::ShaderStage::Compute;
    physicsEntries[3].buffer.type = wgpu::BufferBindingType::Storage;

    // binEnd
    physicsEntries[4].binding = 4;
    physicsEntries[4].visibility = wgpu::ShaderStage::Compute;
    physicsEntries[4].buffer.type = wgpu::BufferBindingType::Storage;

    // omega
    physicsEntries[5].binding = 5;
    physicsEntries[5].visibility = wgpu::ShaderStage::Compute;
    physicsEntries[5].buffer.type = wgpu::BufferBindingType::Storage;

    // assignments
    physicsEntries[6].binding = 6;
    physicsEntries[6].visibility = wgpu::ShaderStage::Compute;
    physicsEntries[6].buffer.type = wgpu::BufferBindingType::Storage;

    // target particles
    physicsEntries[7].binding = 7;
    physicsEntries[7].visibility = wgpu::ShaderStage::Compute;
    physicsEntries[7].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;

    // sort indices
    physicsEntries[8].binding = 8;
    physicsEntries[8].visibility = wgpu::ShaderStage::Compute;
    physicsEntries[8].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;

    auto physicsBGL = bufferManager.createBGL(physicsEntries, "physicsBGL");

    // Group: Radix Sort
    std::array<wgpu::BindGroupLayoutEntry, 7> radixEntries{};

    for (int i = 0; i < radixEntries.size(); i++) {
        radixEntries[i].binding = i;
        radixEntries[i].visibility = wgpu::ShaderStage::Compute;
        radixEntries[i].buffer.type = wgpu::BufferBindingType::Storage;
    }
    radixEntries[5].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;

    auto radixBGL = bufferManager.createBGL(radixEntries, "radixBGL");

    /* Pipeline Layouts */
    // Solver pipeline
    std::array<wgpu::BindGroupLayout, 2> solverLayouts = {
            globalComputeBGL,   // group 0
            solverBGL,          // group 1
    };

    wgpu::PipelineLayoutDescriptor solverPLDesc{};
    solverPLDesc.bindGroupLayoutCount = solverLayouts.size();
    solverPLDesc.bindGroupLayouts = solverLayouts.data();

    auto solverPipelineLayout = device.CreatePipelineLayout(&solverPLDesc);

    // Radix
    std::array<wgpu::BindGroupLayout, 2> radixLayouts = {
            globalComputeBGL,   // group 0
            radixBGL,           // group 1
    };
    wgpu::PipelineLayoutDescriptor radixPLDesc{};
    radixPLDesc.bindGroupLayoutCount = radixLayouts.size();
    radixPLDesc.bindGroupLayouts = radixLayouts.data();

    auto radixPipelineLayout = device.CreatePipelineLayout(&radixPLDesc);

    // Physics (PBF) pipeline
    std::array<wgpu::BindGroupLayout, 2> physicsLayouts = {
            globalComputeBGL,   // group 0
            physicsBGL,         // group 1
    };

    wgpu::PipelineLayoutDescriptor physicsPLDesc{};
    physicsPLDesc.bindGroupLayoutCount = physicsLayouts.size();
    physicsPLDesc.bindGroupLayouts = physicsLayouts.data();

    auto physicsPipelineLayout = device.CreatePipelineLayout(&physicsPLDesc);

    // Render pipeline (vertex + fragment shader)
    wgpu::PipelineLayoutDescriptor renderPLDesc{};
    renderPLDesc.bindGroupLayoutCount = 1;
    renderPLDesc.bindGroupLayouts = &globalRenderBGL;

    auto renderPipelineLayout = device.CreatePipelineLayout(&renderPLDesc);

    // TODO: abstract verbose binding setup into functions for readability
    /* Bind Groups */
    // group 0 - params
    std::array<wgpu::BindGroupEntry, 2> globalBGEntries;
    globalBGEntries[0] = {.binding = 0, .buffer = particleBuffer, .offset = 0};
    globalBGEntries[1] = {.binding = 1, .buffer = paramsBuffer, .offset = 0, .size = sizeof(params)};

    wgpu::BindGroupDescriptor globalComputeBGDesc{};
    globalComputeBGDesc.layout = globalComputeBGL;
    globalComputeBGDesc.entryCount = globalBGEntries.size();
    globalComputeBGDesc.entries = globalBGEntries.data();

    globalComputeBG = device.CreateBindGroup(&globalComputeBGDesc);

    wgpu::BindGroupDescriptor globalRenderBGDesc{};
    globalRenderBGDesc.layout = globalRenderBGL;
    globalRenderBGDesc.entryCount = globalBGEntries.size();
    globalRenderBGDesc.entries = globalBGEntries.data();

    globalRenderBG = device.CreateBindGroup(&globalRenderBGDesc);

    puts("params group done!");

    // group 1 - solver
    assert(assignmentsBuffer);
    assert(targetParticleBuffer);
    assert(pricesBuffer);
    assert(bidValueBuffer);
    assert(bidFromRowBuffer);
    assert(solverBGL);

    std::vector<wgpu::BindGroupEntry> solverBGEntries = BufferManager::getBGEntries(
            {
                    assignmentsBuffer,
                    targetParticleBuffer,
                    pricesBuffer,
                    bidValueBuffer,
                    bidFromRowBuffer,
                    ownerBuffer,
            });

    assert(solverBGEntries.size() == solverEntries.size());

    wgpu::BindGroupDescriptor solverBGDesc{};
    solverBGDesc.layout = solverBGL;
    solverBGDesc.entryCount = solverBGEntries.size();
    solverBGDesc.entries = solverBGEntries.data();

    cout << solverBGEntries.size() << endl;

    puts("Attempt bind group creation");

    solverBG = device.CreateBindGroup(&solverBGDesc);

    puts("solver group done!");

    assert(localPrefixSumBuffer);
    assert(prefixBlockSumBuffer);
    assert(particleBuffer);
    assert(binStartBuffer);
    assert(binEndBuffer);
    assert(outputHashBuffer);
    assert(sortIndicesBuffer);
    assert(sortIndicesAltBuffer);

    auto makeRadixBGEntries = [](const wgpu::Buffer &sortIn, const wgpu::Buffer &sortOut) {
        return BufferManager::getBGEntries(
                {localPrefixSumBuffer,
                 prefixBlockSumBuffer,
                 binStartBuffer,
                 binEndBuffer,
                 outputHashBuffer,
                 sortIn,
                 sortOut});
    };

    std::vector<wgpu::BindGroupEntry> radixBGEntries = makeRadixBGEntries(sortIndicesBuffer, sortIndicesAltBuffer);
    std::vector<wgpu::BindGroupEntry> radixBGAltEntries = makeRadixBGEntries(sortIndicesAltBuffer, sortIndicesBuffer);

    assert(radixBGEntries.size() == radixEntries.size());

    wgpu::BindGroupDescriptor radixBGDesc{};
    radixBGDesc.layout = radixBGL;
    radixBGDesc.entryCount = radixBGEntries.size();
    radixBGDesc.entries = radixBGEntries.data();
    radixBG = device.CreateBindGroup(&radixBGDesc);

    wgpu::BindGroupDescriptor radixBGAltDesc{};
    radixBGAltDesc.layout = radixBGL;
    radixBGAltDesc.entryCount = radixBGAltEntries.size();
    radixBGAltDesc.entries = radixBGAltEntries.data();
    radixBGAlt = device.CreateBindGroup(&radixBGAltDesc);

    puts("radix Group done!");

    // group 3 - physics (PBF)
    assert(lambdasBuffer);
    assert(deltaPosBuffer);
    assert(posStarBuffer);
    assert(binStartBuffer);
    assert(binEndBuffer);
    assert(omegaBuffer);
    assert(assignmentsBuffer);
    assert(targetParticleBuffer);
    assert(sortIndicesBuffer);
    std::vector<wgpu::BindGroupEntry> physicsBGEntries = BufferManager::getBGEntries(
            {
                    lambdasBuffer,
                    deltaPosBuffer,
                    posStarBuffer,
                    binStartBuffer,
                    binEndBuffer,
                    omegaBuffer,
                    assignmentsBuffer,
                    targetParticleBuffer,
                    sortIndicesBuffer
            }
    );

    assert(physicsBGEntries.size() == physicsEntries.size());

    wgpu::BindGroupDescriptor physicsBGDesc{};
    physicsBGDesc.layout = physicsBGL;
    physicsBGDesc.entryCount = physicsBGEntries.size();
    physicsBGDesc.entries = physicsBGEntries.data();

    physicsBG = device.CreateBindGroup(&physicsBGDesc);

    puts("pbf group done!");

    /* Pipelines */
    const auto buildComputePipeline = [&](
            const wgpu::StringView &entrypoint,
            const wgpu::ShaderModule &shaderModule,
            const wgpu::PipelineLayout &pipelineLayout,
            std::span<wgpu::ConstantEntry> constants = {}) -> wgpu::ComputePipeline {
        wgpu::ComputePipelineDescriptor pipelineDesc{
                .layout = pipelineLayout,
                .compute = {
                        .module = shaderModule,
                        .entryPoint = entrypoint
                }
        };

        if (!constants.empty()) {
            pipelineDesc.compute.constants = constants.data();
            pipelineDesc.compute.constantCount = std::size(constants);
        }

        return device.CreateComputePipeline(&pipelineDesc);
    };

    auto buildSolverComputePipeline = [&](const wgpu::StringView &entrypoint, double eps, int32_t useDistance) {
        wgpu::ConstantEntry solverConstants[] = {
                {.key = "EPSILON", .value = eps},
                {.key = "GRID_DIM", .value = static_cast<double>(DIM)},
                {.key = "CANDIDATE_RADIUS", .value = static_cast<double>(AUCTION_CANDIDATE_RADIUS)},
                {.key = "USE_DISTANCE", .value = static_cast<double>(useDistance)},
        };
        return buildComputePipeline(
                entrypoint,
                solverShaderModule,
                solverPipelineLayout,
                solverConstants
        );
    };

    puts("SOLVER PIPELINE");

    auto buildPhysicsComputePipeline = [&](const wgpu::StringView &entrypoint) {
        return buildComputePipeline(
                entrypoint,
                physicsShaderModule,
                physicsPipelineLayout
        );
    };

    puts("PHYSICS COMPUTE PIPELINE");

    // radix override constants
    auto buildRadixComputePipeline = [&](const wgpu::StringView &entrypoint, wgpu::ShaderModule shaderModule) {
        std::vector<wgpu::ComputePipeline> out;

        for (int i = 0; i < RADIX_PASS_COUNT; i++) {
            wgpu::ConstantEntry radixConstants[] = {
                    {.key = "WORKGROUP_COUNT", .value = static_cast<double>(RADIX_WORKGROUP_COUNT)},
                    {.key = "THREADS_PER_WORKGROUP", .value = 128},
                    {.key = "WORKGROUP_SIZE_X", .value = 16},
                    {.key = "WORKGROUP_SIZE_Y", .value = 8},
                    {.key = "CURRENT_BIT", .value = static_cast<double>(i * 2)},
            };

            out.emplace_back(buildComputePipeline(
                    entrypoint,
                    shaderModule,
                    radixPipelineLayout,
                    radixConstants)
            );
        }

        return out;
    };

    wgpu::ConstantEntry radixScanConstants[] = {
            {.key = "WORKGROUP_COUNT", .value = static_cast<double>(RADIX_WORKGROUP_COUNT)},
    };

    puts("RADIX COMPUTE PIPELINE");

    // solver — converge: RGB+distance fill-only; live: RGB-only per frame after convergence
    solverBiddingConvergePipeline = buildSolverComputePipeline("auctionBiddingPhase", AUCTION_EPSILON, 1);
    solverUpdateConvergePipeline = buildSolverComputePipeline("auctionUpdatePhase", AUCTION_EPSILON, 1);
    solverRepairConvergePipeline = buildSolverComputePipeline("auctionRepairPhase", AUCTION_EPSILON, 1);

    solverBiddingPipeline = buildSolverComputePipeline("auctionBiddingPhase", AUCTION_EPSILON, 0);
    solverUpdatePipeline = buildSolverComputePipeline("auctionUpdatePhase", AUCTION_EPSILON, 0);
    solverInitPipeline = buildSolverComputePipeline("auctionInit", AUCTION_EPSILON, 0);
    solverSyncPipeline = buildSolverComputePipeline("auctionSyncPhase", AUCTION_EPSILON, 0);
    solverRepairPipeline = buildSolverComputePipeline("auctionRepairPhase", AUCTION_EPSILON, 0);

    g_matchingConverged = false;
    g_convergeFramesLeft = auctionConvergenceFrames(NUM_PARTICLES);

    // radix
    radixSortPipelines = buildRadixComputePipeline("radix_sort", radixSortShaderModule);
    radixReorderPipelines = buildRadixComputePipeline("radix_sort_reorder", radixReorderShaderModule);
    radixScanPipeline = buildComputePipeline(
            "radixScanBlockSums",
            radixScanShaderModule,
            radixPipelineLayout,
            radixScanConstants
    );

    // pbf
    physicsExternalForcesPipeline = buildPhysicsComputePipeline("pbfExternalForces");
    physicsClearBinsPipeline = buildPhysicsComputePipeline("clearSpatialBins");
    physicsBuildBinsPipeline = buildPhysicsComputePipeline("buildSpatialBins");
    physicsSolverOnePipeline = buildPhysicsComputePipeline("pbfSolverPass");
    physicsSolverTwoPipeline = buildPhysicsComputePipeline("pbfSolverPassTwo");
    physicsSolverThreePipeline = buildPhysicsComputePipeline("pbfSolverPassThree");

    puts("COMPUTE");

    // render
    wgpu::RenderPipelineDescriptor renderPipelineDesc{};
    wgpu::ColorTargetState colorTargetState{.format=format};

    wgpu::FragmentState fragmentState{
            .module = renderShaderModule,
            .entryPoint= "fragmentMain",
            .targetCount=1,
            .targets=&colorTargetState
    };

    renderPipelineDesc.layout = renderPipelineLayout;
    renderPipelineDesc.vertex = {.module = renderShaderModule, .entryPoint = "vertexMain"};
    renderPipelineDesc.primitive = {.topology = wgpu::PrimitiveTopology::TriangleList};
    renderPipelineDesc.fragment = &fragmentState;
    renderPipeline = device.CreateRenderPipeline(&renderPipelineDesc);

    // Upload data
    queue.WriteBuffer(
            indexBuffer,
            0,
            indexData.data(),
            indexData.size() * sizeof(uint16_t)
    );

    // solver
    queue.WriteBuffer(
            particleBuffer,
            0,
            particleCPUData.data(),
            particleCPUData.size() * sizeof(ParticleCPU)
    );

    queue.WriteBuffer(paramsBuffer, 0, &params, sizeof(params));

    bufferManager.fillVal(assignmentsBuffer, NUM_PARTICLES, -1);

    queue.WriteBuffer(
            targetParticleBuffer,
            0,
            targetParticleCPUData.data(),
            targetParticleCPUData.size() * sizeof(TargetParticleCPU)
    );

    bufferManager.fillZero(pricesBuffer, NUM_PARTICLES);
    bufferManager.fillVal(bidValueBuffer, NUM_PARTICLES, std::numeric_limits<int32_t>::min());
    bufferManager.fillVal(bidFromRowBuffer, NUM_PARTICLES, -1);

    std::vector<uint32_t> indices(NUM_PARTICLES);
    std::iota(indices.begin(), indices.end(), 0u);
    queue.WriteBuffer(
            sortIndicesBuffer,
            0,
            indices.data(),
            indices.size() * sizeof(uint32_t)
    );
    queue.WriteBuffer(
            sortIndicesAltBuffer,
            0,
            indices.data(),
            indices.size() * sizeof(uint32_t)
    );

    bufferManager.fillVal(ownerBuffer, NUM_PARTICLES, -1);

    // not mapping for now since I want to preserve order of binding args to make sure I don't miss anything
    // radix
    bufferManager.fillZero(localPrefixSumBuffer, NUM_PARTICLES);
    bufferManager.fillZero(prefixBlockSumBuffer, 4 * RADIX_WORKGROUP_COUNT);
    bufferManager.fillVal(binStartBuffer, NUM_BINS, NUM_PARTICLES);
    bufferManager.fillZero(binEndBuffer, NUM_BINS);
    bufferManager.fillZero(outputHashBuffer, NUM_PARTICLES);

    // physics
    bufferManager.fillZero(lambdasBuffer, NUM_PARTICLES);
    bufferManager.fillZeroBytes(deltaPosBuffer, NUM_PARTICLES * sizeof(vec2<float>));
    bufferManager.fillZeroBytes(posStarBuffer, NUM_PARTICLES * sizeof(vec2<float>));
    bufferManager.fillZero(omegaBuffer, NUM_PARTICLES);

    assert(NUM_PARTICLES == static_cast<int32_t>(particleCPUData.size()));

    encoder = device.CreateCommandEncoder();
    solverComputePass(solverInitPipeline);
    const int32_t initAuctionIters = auctionInitIterations(NUM_PARTICLES);
    runAuctionRounds(initAuctionIters, solverBiddingConvergePipeline, solverUpdateConvergePipeline);
    solverConvergePass(auctionRepairRounds(NUM_PARTICLES), solverRepairConvergePipeline);

    wgpu::CommandBuffer initCommands = encoder.Finish();
    queue.Submit(1, &initCommands);
}

void InitGraphics() {
    ConfigureSurface();
    CreateRenderPipeline();
}

// main loop
void Render() {
    wgpu::SurfaceTexture surfaceTexture;
    surface.GetCurrentTexture(&surfaceTexture);

    wgpu::RenderPassColorAttachment attachment{
            .view = surfaceTexture.texture.CreateView(),
            .loadOp = wgpu::LoadOp::Clear,
            .storeOp = wgpu::StoreOp::Store
    };

    wgpu::RenderPassDescriptor renderpass{.colorAttachmentCount = 1,
            .colorAttachments = &attachment};

    // Compute
    encoder = device.CreateCommandEncoder();
    // Step 1: Auction — distance+fill until converged, then RGB-only live rounds
    if (AUCTION_ITERS_PER_FRAME > 0) {
        const bool convergePhase = !g_matchingConverged;
        if (convergePhase) {
            runAuctionRounds(AUCTION_ITERS_PER_FRAME, solverBiddingConvergePipeline,
                             solverUpdateConvergePipeline);
            solverConvergePass(auctionRepairRoundsPerFrame(), solverRepairConvergePipeline);
            if (--g_convergeFramesLeft <= 0) {
                g_matchingConverged = true;
            }
        } else {
            runAuctionRounds(AUCTION_ITERS_PER_FRAME, solverBiddingPipeline, solverUpdatePipeline);
        }
    }

    // Step 2: PBF External Forces
    physicsComputePass(physicsExternalForcesPipeline);

    // Step 3: radix sort (interleaved sort -> scan -> reorder per bit pass)
    bool pingPong = false;
    for (int32_t i = 0; i < RADIX_PASS_COUNT; i++) {
        wgpu::BindGroup radixBindGroup = pingPong ? radixBGAlt : radixBG;

        auto sortPass = encoder.BeginComputePass();
        sortPass.SetPipeline(radixSortPipelines[i]);
        sortPass.SetBindGroup(0, globalComputeBG);
        sortPass.SetBindGroup(1, radixBindGroup);
        sortPass.DispatchWorkgroups(RADIX_WORKGROUP_COUNT);
        sortPass.End();

        auto scanPass = encoder.BeginComputePass();
        scanPass.SetPipeline(radixScanPipeline);
        scanPass.SetBindGroup(0, globalComputeBG);
        scanPass.SetBindGroup(1, radixBindGroup);
        scanPass.DispatchWorkgroups(1);
        scanPass.End();

        auto reorderPass = encoder.BeginComputePass();
        reorderPass.SetPipeline(radixReorderPipelines[i]);
        reorderPass.SetBindGroup(0, globalComputeBG);
        reorderPass.SetBindGroup(1, radixBindGroup);
        reorderPass.DispatchWorkgroups(RADIX_WORKGROUP_COUNT);
        reorderPass.End();

        pingPong = !pingPong;
    }

    physicsComputePass(physicsClearBinsPipeline);
    physicsComputePass(physicsBuildBinsPipeline);

    // Step 4: Continue rest of PBF
    physicsComputePass(physicsSolverOnePipeline);
    physicsComputePass(physicsSolverTwoPipeline);
    physicsComputePass(physicsSolverThreePipeline);

    // Step 5: Render
    // Vertex + Fragment shader must use render pass
    wgpu::RenderPassEncoder pass = encoder.BeginRenderPass(&renderpass);
    pass.SetPipeline(renderPipeline);
    // apply index buffer & bind group
    pass.SetIndexBuffer(indexBuffer, wgpu::IndexFormat::Uint16);
    pass.SetBindGroup(0, globalRenderBG);

    pass.DrawIndexed(
            6,
            static_cast<uint32_t>(NUM_PARTICLES),
            0,
            0,
            0
    );
    pass.End();
    wgpu::CommandBuffer commands = encoder.Finish();

    // submit to command queue
    queue.Submit(1, &commands);
}

void Simulate(uint16_t dim = 32) {
    // main - so can expose args to emscripten
    setDim(dim);
    Init();
    Start();
}

int main() {
    Simulate();

    // Destroy WebGPU instance
    // TODO: release queue

    return 0;
}