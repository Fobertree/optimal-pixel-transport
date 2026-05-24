#include <iostream>
// GLFW wasm32-emscripten triplet
#include <GLFW/glfw3.h>

#if defined(__EMSCRIPTEN__)
#define WEBGPU_BACKEND_EMSCRIPTEN

#include <emscripten/emscripten.h>

#endif

#include <dawn/webgpu_cpp_print.h>
#include <webgpu/webgpu_cpp.h>
#include <webgpu/webgpu_glfw.h>

#include "declare.h"
#include "solver.h"
#include "readfile.h"


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
// pbf
wgpu::Buffer lambdasBuffer;
wgpu::Buffer deltaPosBuffer;
wgpu::Buffer posStarBuffer;
wgpu::Buffer binStartBuffer;
wgpu::Buffer binEndBuffer;
wgpu::Buffer omegaBuffer;
// radix
wgpu::Buffer localPrefixSumBuffer;
wgpu::Buffer prefixBlockSumBuffer;
wgpu::Buffer auxParticleSortBuffer; // naive auxiliary buffer since radix sort is not in-place, so copy back
wgpu::Buffer outputHashBuffer;

wgpu::ComputePipeline solverPipeline, physicsPipeline, radixPipeline;
wgpu::RenderPipeline renderPipeline;

wgpu::BindGroup globalBG, solverBG, physicsBG, radixBG;

std::vector<ParticleCPU> particleCPUData;

// ptr for runtime polymorphism
SolverBase *solver;

// TODO: populate
std::string renderShaderCode;
std::string physicsShaderCode;
std::string solverShaderCode;

std::string shaderCode = read_wgsl_file("particle_shader.wgsl");

std::vector<uint16_t> indexData = {
        0, 1, 3,
        1, 2, 3
};

struct Params {
    uint32_t size;
    // C++ standard doesn't guarantee 32-bit float
    float rho0;
    float H;
    float dt;
    uint32_t solverIterations;
    float cellSize;
    uint32_t numBins;
};

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

void InitParticles() {
//    constexpr static COST_TYPE costType = COST_TYPE::RGB_DIST_HYBRID;
    using DefaultSinkhorn = Sinkhorn<COST_TYPE::RGB_DIST_HYBRID>;
    using DefaultLAPJV = LAPJV<int64_t>;
    using DefaultHungarian = Hungarian<float>;
    // Using integral types should be much better
    using IntegralHungarian = Hungarian<int64_t, COST_TYPE::RGB_DIST_INT_HYBRID>;
    solver = new IntegralHungarian("img_1.png", "img_6.png", 100, 100);
    particleCPUData = solver->getParticleCPUBuffer();
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

    wgpu::DeviceDescriptor desc{};
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

    // create commandQueue
    queue = device.GetQueue();

#if ERROR_SCOPE
    // for debugging
    device.PushErrorScope(wgpu::ErrorFilter::Validation);
#endif
}

void ConfigureSurface() {
    wgpu::SurfaceCapabilities capabilities;
    surface.GetCapabilities(adapter, &capabilities);
    format = capabilities.formats[0];

    wgpu::SurfaceConfiguration config{.device = device,
            .format = format,
            .width = kWidth,
            .height = kHeight};

    surface.Configure(&config);
}

void CreateRenderPipeline() {
    // TODO: abstract this or reduce LOC bc lowkey unreadable
    using COST_ITEM_T = int;

    // cost buffer
    constexpr uint32_t DIM = 100;
    auto src_buf = ParticleBuffer("img_1.png", DIM, DIM);
    auto tar_buf = ParticleBuffer("img_6.png", DIM, DIM);
    auto cost_buffer = get_cost_buffer<COST_TYPE::RGB_DIST_INT_HYBRID, COST_ITEM_T>(src_buf, tar_buf);

    // params struct
    uint32_t sizeValue = DIM * DIM;
    Params params{.size = sizeValue};

    /* Load Shader Modules */
    // TODO: read shader files after impl

    // Solver shader
    wgpu::ShaderSourceWGSL solverWgsl{{.code=solverShaderCode.c_str()}};
    wgpu::ShaderModuleDescriptor solverShaderModuleDescriptor{.nextInChain = &solverWgsl};
    wgpu::ShaderModule solverShaderModule = device.CreateShaderModule(&solverShaderModuleDescriptor);

    // Physics shader
    wgpu::ShaderSourceWGSL physicsWgsl{{.code=physicsShaderCode.c_str()}};
    wgpu::ShaderModuleDescriptor physicsShaderModuleDescriptor{.nextInChain = &physicsWgsl};
    wgpu::ShaderModule physicsShaderModule = device.CreateShaderModule(&physicsShaderModuleDescriptor);

    // Render shader
    wgpu::ShaderSourceWGSL renderWgsl{{.code=renderShaderCode.c_str()}};
    wgpu::ShaderModuleDescriptor renderShaderModuleDescriptor{.nextInChain = &renderWgsl};
    wgpu::ShaderModule renderShaderModule = device.CreateShaderModule(&renderShaderModuleDescriptor);

    /* Create Buffers */
    // TODO: abstract buffer creation into util
    // index buffer
    wgpu::BufferDescriptor indexBufferDesc{};
    indexBufferDesc.size = indexData.size() * sizeof(uint16_t);
    indexBufferDesc.usage = wgpu::BufferUsage::Index | wgpu::BufferUsage::CopyDst;

    indexBuffer = device.CreateBuffer(&indexBufferDesc);

    // particle buffer
    wgpu::BufferDescriptor particleBufferDesc{};
    particleBufferDesc.size = MAX_CPU_PARTICLES * sizeof(ParticleCPU);
    particleBufferDesc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;

    particleBuffer = device.CreateBuffer(&particleBufferDesc);


    // params
    wgpu::BufferDescriptor paramsDesc{};
    paramsDesc.size = sizeof(uint32_t);
    paramsDesc.usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst;
    paramsBuffer = device.CreateBuffer(&paramsDesc);

    // Solver

    // cost buffer
    wgpu::BufferDescriptor costBufferDesc{};
    costBufferDesc.size = cost_buffer.size() * sizeof(int);
    costBufferDesc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;

    costBuffer = device.CreateBuffer(&costBufferDesc);

    // assignments buffer
    wgpu::BufferDescriptor assignmentsBufferDesc{};
    assignmentsBufferDesc.size = cost_buffer.size() * sizeof(int); // tmp code smell, but same dim as cost buffer
    assignmentsBufferDesc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;

    assignmentsBuffer = device.CreateBuffer(&assignmentsBufferDesc);

    // radix
    size_t sz = particleCPUData.size();
    // local prefix sum buffer
    wgpu::BufferDescriptor localPrefixSumBufferDesc{};
    localPrefixSumBufferDesc.size = sz * sizeof(int);
    localPrefixSumBufferDesc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;

    localPrefixSumBuffer = device.CreateBuffer(&localPrefixSumBufferDesc);

    // block sum buffer
    wgpu::BufferDescriptor prefixBlockSumBufferDesc{};
    prefixBlockSumBufferDesc.size = sz * sizeof(int);
    prefixBlockSumBufferDesc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;

    prefixBlockSumBuffer = device.CreateBuffer(&prefixBlockSumBufferDesc);

    // aux particle sort buffer
    wgpu::BufferDescriptor auxParticleSortBufferDesc{};
    auxParticleSortBufferDesc.size = MAX_CPU_PARTICLES * sizeof(ParticleCPU);
    auxParticleSortBufferDesc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;

    auxParticleSortBuffer = device.CreateBuffer(&auxParticleSortBufferDesc);

    // PBF/Physics
    // lambda
    wgpu::BufferDescriptor lambdasBufferDesc{};
    lambdasBufferDesc.size = sz * sizeof(int);
    lambdasBufferDesc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;

    lambdasBuffer = device.CreateBuffer(&lambdasBufferDesc);

    // deltaPos
    wgpu::BufferDescriptor deltaPosBufferDesc{};
    deltaPosBufferDesc.size = sz * sizeof(int);
    deltaPosBufferDesc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;

    deltaPosBuffer = device.CreateBuffer(&deltaPosBufferDesc);

    // posStar
    wgpu::BufferDescriptor posStarBufferDesc{};
    posStarBufferDesc.size = sz * sizeof(int);
    posStarBufferDesc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;

    posStarBuffer = device.CreateBuffer(&posStarBufferDesc);

    // binStart
    wgpu::BufferDescriptor binStartBufferDesc{};
    binStartBufferDesc.size = sz * sizeof(int);
    binStartBufferDesc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;

    binStartBuffer = device.CreateBuffer(&binStartBufferDesc);

    // binEnd
    wgpu::BufferDescriptor binEndBufferDesc{};
    binEndBufferDesc.size = sz * sizeof(int);
    binEndBufferDesc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;

    binEndBuffer = device.CreateBuffer(&binEndBufferDesc);

    // omegaBuffer
    wgpu::BufferDescriptor omegaBufferDesc{};
    omegaBufferDesc.size = sz * sizeof(int);
    omegaBufferDesc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;

    binEndBuffer = device.CreateBuffer(&omegaBufferDesc);

    /* Bind Group Layouts */
    // Group 0: Global Bind Group
    std::vector<wgpu::BindGroupLayoutEntry> globalEntries{};

    // particles
    globalEntries[0].binding = 0;
    globalEntries[0].visibility = wgpu::ShaderStage::Compute | wgpu::ShaderStage::Vertex;
    globalEntries[0].buffer.type = wgpu::BufferBindingType::Storage;
    globalEntries[0].buffer.minBindingSize = MAX_CPU_PARTICLES * sizeof(ParticleCPU);

    // params
    globalEntries[1].binding = 1;
    globalEntries[1].visibility = wgpu::ShaderStage::Compute | wgpu::ShaderStage::Vertex;
    globalEntries[1].buffer.type = wgpu::BufferBindingType::Uniform;
    globalEntries[1].buffer.minBindingSize = sizeof(Params);

    wgpu::BindGroupLayoutDescriptor paramsLayoutDesc{};
    paramsLayoutDesc.entryCount = globalEntries.size();
    paramsLayoutDesc.entries = globalEntries.data();

    auto globalBGL = device.CreateBindGroupLayout(&paramsLayoutDesc);

    // Group 1: Solver BindGroup (assignments, cost)
    std::array<wgpu::BindGroupLayoutEntry, 2> solverEntries{};

    // assignments
    solverEntries[0].binding = 0;
    solverEntries[0].visibility = wgpu::ShaderStage::Compute;
    solverEntries[0].buffer.type = wgpu::BufferBindingType::Storage;

    // cost
    solverEntries[1].binding = 1;
    solverEntries[1].visibility = wgpu::ShaderStage::Compute;
    solverEntries[1].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;

    wgpu::BindGroupLayoutDescriptor solverLayoutDesc{};
    solverLayoutDesc.entryCount = solverEntries.size();
    solverLayoutDesc.entries = solverEntries.data();

    auto solverBGL = device.CreateBindGroupLayout(&solverLayoutDesc);

    // Group 2: Physics (Unique PBF buffers)
    std::array<wgpu::BindGroupLayoutEntry, 6> physicsEntries;
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

    wgpu::BindGroupLayoutDescriptor physicsLayoutDesc{};
    physicsLayoutDesc.entryCount = physicsEntries.size();
    physicsLayoutDesc.entries = physicsEntries.data();

    auto physicsBGL = device.CreateBindGroupLayout(&physicsLayoutDesc);

    // Group 4: Radix Sort
    std::array<wgpu::BindGroupLayoutEntry, 4> radixEntries;

    // local prefix sum
    radixEntries[0].binding = 0;
    radixEntries[0].visibility = wgpu::ShaderStage::Compute;
    radixEntries[0].buffer.type = wgpu::BufferBindingType::Storage;

    // block sum
    radixEntries[1].binding = 1;
    radixEntries[1].visibility = wgpu::ShaderStage::Compute;
    radixEntries[1].buffer.type = wgpu::BufferBindingType::Storage;

    // output particles - pseudo swap buffer
    // TODO: I don't like this binding - replace it with something better
    radixEntries[2].binding = 2;
    radixEntries[2].visibility = wgpu::ShaderStage::Compute;
    radixEntries[2].buffer.type = wgpu::BufferBindingType::Storage;

    // output hashes (for bin start and bin end accumulation)
    radixEntries[3].binding = 3;
    radixEntries[3].visibility = wgpu::ShaderStage::Compute;
    radixEntries[3].buffer.type = wgpu::BufferBindingType::Storage;

    wgpu::BindGroupLayoutDescriptor radixLayoutDesc{};
    radixLayoutDesc.entryCount = radixEntries.size();
    radixLayoutDesc.entries = radixEntries.data();

    auto radixBGL = device.CreateBindGroupLayout(&radixLayoutDesc);

    /* Pipeline Layouts */
    // Solver pipeline
    std::array<wgpu::BindGroupLayout, 2> solverLayouts = {
            globalBGL,  // group 0
            solverBGL,  // group 1
    };

    wgpu::PipelineLayoutDescriptor solverPLDesc{};
    solverPLDesc.bindGroupLayoutCount = solverLayouts.size();
    solverPLDesc.bindGroupLayouts = solverLayouts.data();

    auto solverPipelineLayout = device.CreatePipelineLayout(&solverPLDesc);

    // Radix
    std::array<wgpu::BindGroupLayout, 2> radixLayouts = {
            globalBGL,          // group 0
            radixBGL,           // group 1
    };
    wgpu::PipelineLayoutDescriptor radixPLDesc{};
    radixPLDesc.bindGroupLayoutCount = radixLayouts.size();
    radixPLDesc.bindGroupLayouts = radixLayouts.data();

    auto radixPipelineLayout = device.CreatePipelineLayout(&radixPLDesc);

    // Physics (PBF) pipeline
    std::array<wgpu::BindGroupLayout, 4> physicsLayouts = {
            globalBGL,          // group 0
            solverBGL,          // group 1
            physicsBGL,         // group 2
    };

    wgpu::PipelineLayoutDescriptor physicsPLDesc{};
    physicsPLDesc.bindGroupLayoutCount = physicsLayouts.size();
    physicsPLDesc.bindGroupLayouts = physicsLayouts.data();

    auto physicsPipelineLayout = device.CreatePipelineLayout(&physicsPLDesc);

    // Render pipeline (vertex + fragment shader)
    wgpu::PipelineLayoutDescriptor renderPLDesc{};
    renderPLDesc.bindGroupLayoutCount = 1;
    renderPLDesc.bindGroupLayouts = &globalBGL; // contains both size + particles

    auto renderPipelineLayout = device.CreatePipelineLayout(&renderPLDesc);

    // TODO: update BG grouping
    // TODO: abstract verbose binding setup into functions for readability
    /* Bind Groups */
    // group 0 - params
    std::array<wgpu::BindGroupEntry, 2> globalBGEntries;
    globalBGEntries[0].binding = 0;
    globalBGEntries[0].buffer = particleBuffer;
    globalBGEntries[0].offset = 0;

    globalBGEntries[1].binding = 1;
    globalBGEntries[1].buffer = paramsBuffer;
    globalBGEntries[1].offset = 0;
    globalBGEntries[1].size = sizeof(params);

    wgpu::BindGroupDescriptor globalBGDesc{};
    globalBGDesc.layout = globalBGL;
    globalBGDesc.entryCount = globalBGEntries.size();
    globalBGDesc.entries = globalBGEntries.data();

    globalBG = device.CreateBindGroup(&globalBGDesc);

    // group 1 - solver
    std::array<wgpu::BindGroupEntry, 2> solverBGEntries;

    solverBGEntries[0].binding = 0;
    solverBGEntries[0].buffer = assignmentsBuffer;

    solverBGEntries[1].binding = 1;
    solverBGEntries[1].buffer = costBuffer;

    wgpu::BindGroupDescriptor solverBGDesc{};
    solverBGDesc.layout = solverBGL;
    solverBGDesc.entryCount = solverBGEntries.size();
    solverBGDesc.entries = solverBGEntries.data();

    solverBG = device.CreateBindGroup(&solverBGDesc);

    // group 2 - physics (PBF)
    std::array<wgpu::BindGroupEntry, 6> physicsBGEntries;
    physicsBGEntries[0].binding = 0;
    physicsBGEntries[0].buffer = lambdasBuffer;

    physicsBGEntries[1].binding = 1;
    physicsBGEntries[1].buffer = deltaPosBuffer;

    physicsBGEntries[2].binding = 2;
    physicsBGEntries[2].buffer = posStarBuffer;

    physicsBGEntries[3].binding = 3;
    physicsBGEntries[3].buffer = binStartBuffer;

    physicsBGEntries[4].binding = 4;
    physicsBGEntries[4].buffer = binEndBuffer;

    physicsBGEntries[5].binding = 5;
    physicsBGEntries[5].buffer = omegaBuffer;

    wgpu::BindGroupDescriptor physicsBGDesc{};
    physicsBGDesc.layout = physicsBGL;
    physicsBGDesc.entryCount = physicsBGEntries.size();
    physicsBGDesc.entries = physicsBGEntries.data();

    physicsBG = device.CreateBindGroup(&physicsBGDesc);

    // group 3 - radix group
    std::array<wgpu::BindGroupEntry, 5> radixBGEntries;

    radixBGEntries[0].binding = 0;
    radixBGEntries[0].buffer = localPrefixSumBuffer;

    radixBGEntries[1].binding = 1;
    radixBGEntries[1].buffer = prefixBlockSumBuffer;

    radixBGEntries[2].binding = 2;
    radixBGEntries[2].buffer = auxParticleSortBuffer;

    radixBGEntries[3].binding = 3;
    radixBGEntries[3].buffer = binStartBuffer;

    radixBGEntries[4].binding = 4;
    radixBGEntries[4].buffer = binEndBuffer;

    radixBGEntries[5].binding = 5;
    radixBGEntries[5].buffer = outputHashBuffer;

    wgpu::BindGroupDescriptor radixBGDesc{};
    radixBGDesc.layout = radixBGL;
    radixBGDesc.entryCount = radixBGEntries.size();
    radixBGDesc.entries = radixBGEntries.data();

    radixBG = device.CreateBindGroup(&radixBGDesc);

    /* Pipelines */
    // TODO: compute pipeline entrypoint specification (especially reduction)
    // solver
    wgpu::ComputePipelineDescriptor solverPipelineDesc{};
    solverPipelineDesc.layout = solverPipelineLayout;
    solverPipeline = device.CreateComputePipeline(&solverPipelineDesc);

    // radix
    wgpu::ComputePipelineDescriptor radixPipelineDesc{};
    radixPipelineDesc.layout = renderPipelineLayout;
    radixPipeline = device.CreateComputePipeline(&radixPipelineDesc);

    // pbf
    wgpu::ComputePipelineDescriptor physicsPipelineDesc{};
    solverPipelineDesc.layout = physicsPipelineLayout;
    physicsPipeline = device.CreateComputePipeline(&physicsPipelineDesc);

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
            MAX_CPU_PARTICLES * sizeof(ParticleCPU)
    );

    queue.WriteBuffer(paramsBuffer, 0, &params, sizeof(params));

    queue.WriteBuffer(
            costBuffer,
            0,
            cost_buffer.data(),
            cost_buffer.size() * sizeof(COST_ITEM_T)
    );

    // radix
    queue.WriteBuffer(
            localPrefixSumBuffer,
            0,
            std::vector<int>(sz, 0).data(),
            sz * sizeof(int)
    );

    queue.WriteBuffer(
            prefixBlockSumBuffer,
            0,
            std::vector<int>(sz, 0).data(),
            sz * sizeof(int)
    );

    // TODO: rm this from radix BG, and treat this as a swap buffer
    queue.WriteBuffer(
            auxParticleSortBuffer,
            0,
            particleCPUData.data(),
            MAX_CPU_PARTICLES * sizeof(ParticleCPU)
    );

    queue.WriteBuffer(
            binStartBuffer,
            0,
            std::vector<int>(sz, 0).data(),
            sz * sizeof(int)
    );

    queue.WriteBuffer(
            binEndBuffer,
            0,
            std::vector<int>(sz, 0).data(),
            sz * sizeof(int)
    );

    // physics
    queue.WriteBuffer(
            lambdasBuffer,
            0,
            std::vector<int>(sz, 0).data(),
            sz * sizeof(int)
    );

    queue.WriteBuffer(
            deltaPosBuffer,
            0,
            std::vector<int>(sz, 0).data(),
            sz * sizeof(int)
    );

    queue.WriteBuffer(
            posStarBuffer,
            0,
            std::vector<int>(sz, 0).data(),
            sz * sizeof(int)
    );

    queue.WriteBuffer(
            binStartBuffer,
            0,
            std::vector<int>(sz, 0).data(),
            sz * sizeof(int)
    );

    queue.WriteBuffer(
            binEndBuffer,
            0,
            std::vector<int>(sz, 0).data(),
            sz * sizeof(int)
    );

    queue.WriteBuffer(
            omegaBuffer,
            0,
            std::vector<int>(sz, 0).data(),
            sz * sizeof(int)
    );
}

void InitGraphics() {
    ConfigureSurface();
    CreateRenderPipeline();
}

void Render() {
    particleCPUData = solver->getParticleCPUBuffer(); // TODO: this is stupid. Not sure if bind by ref is better
    wgpu::SurfaceTexture surfaceTexture;
    surface.GetCurrentTexture(&surfaceTexture);

    wgpu::RenderPassColorAttachment attachment{
            .view = surfaceTexture.texture.CreateView(),
            .loadOp = wgpu::LoadOp::Clear,
            .storeOp = wgpu::StoreOp::Store
    };

    wgpu::RenderPassDescriptor renderpass{.colorAttachmentCount = 1,
            .colorAttachments = &attachment};

    wgpu::CommandEncoder encoder = device.CreateCommandEncoder();

    // TODO: update BG grouping

    // Compute
    {
        // solver
        auto pass = encoder.BeginComputePass();
        pass.SetPipeline(solverPipeline);
        pass.SetBindGroup(0, globalBG);
        pass.SetBindGroup(1, solverBG);
        pass.End();
    }
    {
        // radix sort hashes (for bins)
        auto pass = encoder.BeginComputePass();
        pass.SetPipeline(radixPipeline);
        pass.SetBindGroup(0, globalBG);
        pass.SetBindGroup(1, radixBG);
    }
    {
        // physics
        auto pass = encoder.BeginComputePass();
        pass.SetPipeline(physicsPipeline);
        pass.SetBindGroup(0, globalBG);
        pass.SetBindGroup(1, solverBG);
        pass.SetBindGroup(2, physicsBG);
        pass.End();
    }

    // Render
    // Vertex + Fragment shader must use render pass
    wgpu::RenderPassEncoder pass = encoder.BeginRenderPass(&renderpass);
    pass.SetPipeline(renderPipeline);
    // apply index buffer & bind group
    pass.SetIndexBuffer(indexBuffer, wgpu::IndexFormat::Uint16);
    pass.SetBindGroup(0, globalBG);

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
    queue.WriteBuffer(
            particleBuffer,
            0,
            particleCPUData.data(),
            particleCPUData.size() * sizeof(ParticleCPU)
    );
    queue.Submit(1, &commands);
}

int main() {
    std::cout << shaderCode << std::endl;
    Init();
    InitParticles();
    Start();

    // Destroy WebGPU instance
    // TODO: release queue

    return 0;
}