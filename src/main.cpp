#include <iostream>
// GLFW wasm32-emscripten triplet
#include <GLFW/glfw3.h>

#if defined(__EMSCRIPTEN__)

#include <emscripten/emscripten.h>

#endif

#include <dawn/webgpu_cpp_print.h>
#include <webgpu/webgpu_cpp.h>
#include <webgpu/webgpu_glfw.h>

#include "declare.h"
#include "solver.h"
#include "readfile.h"
#include "buffermanager.h"

#define WATCH(x) (std::cout << #x << " = " << (x) << std::endl)

// tmp spaghetti var bc gcc doesn't detect changes/re-compile on wgsl-only changes
bool bob = true;

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
wgpu::Buffer costBuffer;
wgpu::Buffer pricesBuffer;
wgpu::Buffer bidValueBuffer;
wgpu::Buffer bidFromRowBuffer;
wgpu::Buffer sortIndicesBuffer; // start with std::iota, shared between solver and radix
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

wgpu::ComputePipeline solverBiddingPipeline, solverUpdatePipeline, solverInitPipeline;
std::vector<wgpu::ComputePipeline> radixSortPipelines, radixReorderPipelines;
wgpu::ComputePipeline physicsExternalForcesPipeline, physicsSolverOnePipeline, physicsSolverTwoPipeline, physicsSolverThreePipeline;
wgpu::RenderPipeline renderPipeline;

wgpu::BindGroup globalComputeBG, globalRenderBG, solverBG, physicsBG, radixBG;

std::vector<ParticleCPU> particleCPUData;
std::vector<TargetParticleCPU> targetParticleCPUData;

// ptr for runtime polymorphism
SolverBase *solver;

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

int32_t DIM, NUM_PARTICLES, NUM_PARTICLES_SQ;

auto solverComputePass = [](const wgpu::ComputePipeline &pipeline) {
    auto pass = encoder.BeginComputePass();
    pass.SetPipeline(pipeline);
    pass.SetBindGroup(0, globalComputeBG);
    pass.SetBindGroup(1, solverBG);
    pass.End();
};

auto radixComputePass = [](const std::vector<wgpu::ComputePipeline> &pipelines) {
    for (const auto &pipeline: pipelines) {
        auto pass = encoder.BeginComputePass();
        pass.SetPipeline(pipeline);
        pass.SetBindGroup(0, globalComputeBG);
        pass.SetBindGroup(1, radixBG);
        pass.End();
    }
};

auto physicsComputePass = [](const wgpu::ComputePipeline &pipeline) {
    auto pass = encoder.BeginComputePass();
    pass.SetPipeline(pipeline);
    pass.SetBindGroup(0, globalComputeBG);
    pass.SetBindGroup(1, physicsBG);
    pass.End();
};

void setDim(uint16_t dim) {
    // lowkey spaghetti
    DIM = dim;
    NUM_PARTICLES = DIM * DIM;
    NUM_PARTICLES_SQ = NUM_PARTICLES * NUM_PARTICLES;

    WATCH(NUM_PARTICLES);
    WATCH(NUM_PARTICLES_SQ);
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
    using COST_ITEM_T = int32_t;

    // cost buffer
    auto srcBuf = ParticleBuffer("img_1.png", DIM, DIM);
    auto tarBuf = ParticleBuffer("img_6.png", DIM, DIM);

    particleCPUData = srcBuf.getParticleCPUBuffer();
    targetParticleCPUData = tarBuf.getTargetParticleCPUBuffer();

    auto cost_buffer = get_cost_buffer<COST_TYPE::RGB_DIST_INT_HYBRID, COST_ITEM_T>(srcBuf, tarBuf, NUM_PARTICLES);

    // params struct
    // TODO: init values - for now just passing some garbage temp values
    int num_bins = 4;

    Params params{.size = DIM,
            .rho0 = 0.5,
            .H = 0.1,
            .dt = 0.5,
            .solverIterations = 0, // TODO: remove
            .cellSize = 2.f / static_cast<float>(num_bins),
            .numBins = static_cast<uint32_t>(num_bins),
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

    // cost buffer
    costBuffer = bufferManager.createWGPUBuffer<int32_t>(
            wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst, NUM_PARTICLES_SQ, "cost");
    // assignments buffer
    assignmentsBuffer = bufferManager.createWGPUBuffer<int32_t>(
            wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst, NUM_PARTICLES_SQ, "assignments");
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
    localPrefixSumBuffer = bufferManager.createWGPUBuffer<int32_t>(
            wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst,
            NUM_PARTICLES, "local prefix");
    // block sum buffer
    prefixBlockSumBuffer = bufferManager.createWGPUBuffer<int32_t>(
            wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst,
            NUM_PARTICLES, "prefix block sum");

    outputHashBuffer = bufferManager.createWGPUBuffer<int32_t>(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst,
                                                               NUM_PARTICLES, "hash");

    sortIndicesBuffer = bufferManager.createWGPUBuffer<int32_t>(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst,
                                                                NUM_PARTICLES, "sortIndices");
    // PBF/Physics
    // lambda
    lambdasBuffer = bufferManager.createWGPUBuffer<int32_t>(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst,
                                                            NUM_PARTICLES, "lambda");
    // deltaPos
    deltaPosBuffer = bufferManager.createWGPUBuffer<int32_t>(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst,
                                                             NUM_PARTICLES, "deltaPos");
    // posStar
    posStarBuffer = bufferManager.createWGPUBuffer<int32_t>(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst,
                                                            NUM_PARTICLES, "posStar");
    // binStart
    binStartBuffer = bufferManager.createWGPUBuffer<int32_t>(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst,
                                                             NUM_PARTICLES, "binStart");
    // binEnd
    binEndBuffer = bufferManager.createWGPUBuffer<int32_t>(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst,
                                                           NUM_PARTICLES, "binEnd");
    // omegaBuffer
    omegaBuffer = bufferManager.createWGPUBuffer<int32_t>(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst,
                                                          NUM_PARTICLES, "omega");

    // targetParticleBuffer
    targetParticleBuffer = bufferManager.createWGPUBuffer<int32_t>(
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
    solverEntries[0].buffer.minBindingSize = NUM_PARTICLES_SQ * sizeof(int32_t);

    // cost
    solverEntries[1].binding = 1;
    solverEntries[1].visibility = wgpu::ShaderStage::Compute;
    solverEntries[1].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
    solverEntries[1].buffer.minBindingSize = NUM_PARTICLES_SQ * sizeof(int32_t);

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
    physicsEntries[3].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;

    // binEnd
    physicsEntries[4].binding = 4;
    physicsEntries[4].visibility = wgpu::ShaderStage::Compute;
    physicsEntries[4].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;

    // omega
    physicsEntries[5].binding = 5;
    physicsEntries[5].visibility = wgpu::ShaderStage::Compute;
    physicsEntries[5].buffer.type = wgpu::BufferBindingType::Storage;

    // assignments
    physicsEntries[6].binding = 6;
    physicsEntries[6].visibility = wgpu::ShaderStage::Compute;
    physicsEntries[6].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;

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
    std::array<wgpu::BindGroupLayoutEntry, 6> radixEntries;

    for (int i = 0; i < radixEntries.size(); i++) {
        radixEntries[i].binding = i;
        radixEntries[i].visibility = wgpu::ShaderStage::Compute;
        radixEntries[i].buffer.type = wgpu::BufferBindingType::Storage;
    }

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
    assert(costBuffer);
    assert(pricesBuffer);
    assert(bidValueBuffer);
    assert(bidFromRowBuffer);
    assert(solverBGL);

    std::vector<wgpu::BindGroupEntry> solverBGEntries = bufferManager.getBGEntries(
            {
                    {assignmentsBuffer,
                            NUM_PARTICLES_SQ},
                    {costBuffer,
                            NUM_PARTICLES_SQ},
                    {pricesBuffer,
                            NUM_PARTICLES},
                    {bidValueBuffer,
                            NUM_PARTICLES},
                    {bidFromRowBuffer,
                            NUM_PARTICLES},
                    {ownerBuffer, NUM_PARTICLES}
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

    // group 2 - radix group
    std::vector<wgpu::BindGroupEntry> radixBGEntries = bufferManager.getBGEntries(
            {{localPrefixSumBuffer, NUM_PARTICLES},
             {prefixBlockSumBuffer, NUM_PARTICLES},
             {binStartBuffer,       NUM_PARTICLES},
             {binEndBuffer,         NUM_PARTICLES},
             {outputHashBuffer,     NUM_PARTICLES},
             {sortIndicesBuffer,    NUM_PARTICLES}});

    assert(radixBGEntries.size() == radixEntries.size());

    wgpu::BindGroupDescriptor radixBGDesc{};
    radixBGDesc.layout = radixBGL;
    radixBGDesc.entryCount = radixBGEntries.size();
    radixBGDesc.entries = radixBGEntries.data();

    radixBG = device.CreateBindGroup(&radixBGDesc);

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
    std::vector<wgpu::BindGroupEntry> physicsBGEntries = bufferManager.getBGEntries({{lambdasBuffer,     NUM_PARTICLES},
                                                                                     {deltaPosBuffer,    NUM_PARTICLES},
                                                                                     {posStarBuffer,     NUM_PARTICLES},
                                                                                     {binStartBuffer,    NUM_PARTICLES},
                                                                                     {binEndBuffer,      NUM_PARTICLES},
                                                                                     {omegaBuffer,       NUM_PARTICLES},
                                                                                     {assignmentsBuffer, NUM_PARTICLES},
                                                                                     {targetParticleBuffer,
                                                                                                         NUM_PARTICLES},
                                                                                     {sortIndicesBuffer,
                                                                                                         NUM_PARTICLES}});

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

    auto buildSolverComputePipeline = [&](const wgpu::StringView &entrypoint, double eps = 1) {
        // TODO: build multiple pipelines to scale epsilon
        wgpu::ConstantEntry solverConstants[] = {
                {.key = "EPSILON", .value = eps},
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
    wgpu::ConstantEntry radixConstants[] = {
            {.key = "WORKGROUP_COUNT", .value = 256},
            {.key = "THREADS_PER_WORKGROUP", .value = 128},
            {.key = "WORKGROUP_SIZE_X", .value = 16},
            {.key = "WORKGROUP_SIZE_Y", .value = 8},
            {.key = "CURRENT_BIT", .value = 4},
    };

    auto buildRadixComputePipeline = [&](const wgpu::StringView &entrypoint, wgpu::ShaderModule shaderModule) {
        std::vector<wgpu::ComputePipeline> out;

        for (int i = 0; i < 32; i++) {
            wgpu::ConstantEntry radixConstants[] = {
                    {.key = "WORKGROUP_COUNT", .value = 256},
                    {.key = "THREADS_PER_WORKGROUP", .value = 128},
                    {.key = "WORKGROUP_SIZE_X", .value = 16},
                    {.key = "WORKGROUP_SIZE_Y", .value = 8},
                    {.key = "CURRENT_BIT", .value = static_cast<double>(i)},
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

    puts("RADIX COMPUTE PIPELINE");

    // solver
    solverBiddingPipeline = buildSolverComputePipeline("auctionBiddingPhase");
    solverUpdatePipeline = buildSolverComputePipeline("auctionUpdatePhase");
    solverInitPipeline = buildSolverComputePipeline("auctionInit");

    // radix
    radixSortPipelines = buildRadixComputePipeline("radix_sort", radixSortShaderModule);
    radixReorderPipelines = buildRadixComputePipeline("radix_sort_reorder", radixReorderShaderModule);

    // pbf
    physicsExternalForcesPipeline = buildPhysicsComputePipeline("pbfExternalForces");
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

    cout << cost_buffer.size() << " " << endl;
    WATCH(NUM_PARTICLES_SQ);
    assert(cost_buffer.size() == NUM_PARTICLES_SQ);

    queue.WriteBuffer(
            costBuffer,
            0,
            cost_buffer.data(),
            cost_buffer.size() * sizeof(COST_ITEM_T)
    );

    bufferManager.fillZero(pricesBuffer, NUM_PARTICLES);
    bufferManager.fillZero(bidValueBuffer, NUM_PARTICLES);
    bufferManager.fillVal(bidFromRowBuffer, NUM_PARTICLES, -1);

    std::vector<int32_t> indices(NUM_PARTICLES);
    std::iota(indices.begin(), indices.end(), 0);
    queue.WriteBuffer(
            sortIndicesBuffer,
            0,
            indices.data(),
            indices.size() * sizeof(int32_t)
    );

    bufferManager.fillZero(ownerBuffer, NUM_PARTICLES);

    // not mapping for now since I want to preserve order of binding args to make sure I don't miss anything
    // radix
    bufferManager.fillZero(localPrefixSumBuffer, NUM_PARTICLES);
    bufferManager.fillZero(prefixBlockSumBuffer, NUM_PARTICLES);
    bufferManager.fillZero(binStartBuffer, NUM_PARTICLES);
    bufferManager.fillZero(binEndBuffer, NUM_PARTICLES);
    bufferManager.fillZero(outputHashBuffer, NUM_PARTICLES);

    // physics
    bufferManager.fillZero(lambdasBuffer, NUM_PARTICLES);
    bufferManager.fillZero(deltaPosBuffer, NUM_PARTICLES);
    bufferManager.fillZero(posStarBuffer, NUM_PARTICLES);
    bufferManager.fillZero(binStartBuffer, NUM_PARTICLES);
    bufferManager.fillZero(binEndBuffer, NUM_PARTICLES);
    bufferManager.fillZero(omegaBuffer, NUM_PARTICLES);
    queue.WriteBuffer(
            targetParticleBuffer,
            0,
            targetParticleCPUData.data(),
            targetParticleCPUData.size()
    );

    queue.WriteBuffer(
            sortIndicesBuffer,
            0,
            indices.data(),
            NUM_PARTICLES * sizeof(int32_t)
    );

    assert(NUM_PARTICLES == particleCPUData.size());

    encoder = device.CreateCommandEncoder();
    // solver init
    solverComputePass(solverInitPipeline);

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
    // Step 1: Single iteration of auction phase
    solverComputePass(solverBiddingPipeline);
    solverComputePass(solverUpdatePipeline);

    // Step 2: PBF External Forces
    physicsComputePass(physicsExternalForcesPipeline);
    // Step 3: radix sort
    radixComputePass(radixSortPipelines);
    radixComputePass(radixReorderPipelines);
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
            particleCPUData.size(),
            0,
            0,
            0
    );
    pass.End();
    wgpu::CommandBuffer commands = encoder.Finish();

    // submit to command queue
    queue.Submit(1, &commands);
}

void Simulate(uint16_t dim = 100) {
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