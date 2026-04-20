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
wgpu::Queue queue;

wgpu::Buffer indexBuffer;
wgpu::Buffer particleBuffer;
wgpu::Buffer costBuffer;
wgpu::Buffer paramsBuffer;
wgpu::Buffer assignmentsBuffer;

wgpu::ComputePipeline solverPipeline, physicsPipeline;
wgpu::RenderPipeline renderPipeline;

wgpu::BindGroup paramsBG, solverBG, particleBG;

// TODO: cleanup after solver impl
std::vector<ParticleCPU> particleCPUData;

// ptr for runtime polymorphism
SolverBase *solver;

// TODO: populate
std::string renderShaderCode;
std::string physicsShaderCode;
std::string solverShaderCode;
std::string shaderCode;

std::vector<uint16_t> indexData = {
        0, 1, 3,
        1, 2, 3
};

struct Params {
    uint32_t size;
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
    puts("..");
    using IntegralHungarian = Hungarian<int64_t, COST_TYPE::RGB_DIST_INT_HYBRID>;
    solver = new IntegralHungarian("img_1.png", "img_6.png", 100, 100);
    particleCPUData = solver->getParticleCPUBuffer();
}

void Init() {
    shaderCode = read_wgsl_file("particle_shader.wgsl");
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
    using COST_ITEM_T = int;

    // cost buffer
    constexpr uint32_t DIM = 100;
    auto src_buf = ParticleBuffer("img_1.png", DIM, DIM);
    auto tar_buf = ParticleBuffer("img_6.png", DIM, DIM);
    auto cost_buffer = get_cost_buffer<COST_TYPE::RGB_DIST_INT_HYBRID, COST_ITEM_T>(src_buf, tar_buf);

    // params struct
    uint32_t sizeValue = DIM * DIM;
    Params params{.size = sizeValue};

    // particle buffer
    wgpu::BufferDescriptor particleBufferDesc{};
    particleBufferDesc.size = MAX_CPU_PARTICLES * sizeof(ParticleCPU);
    particleBufferDesc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;

    particleBuffer = device.CreateBuffer(&particleBufferDesc);

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

    // params
    wgpu::BufferDescriptor paramsDesc{};
    paramsDesc.size = sizeof(uint32_t);
    paramsDesc.usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst;
    paramsBuffer = device.CreateBuffer(&paramsDesc);

    // index buffer
    wgpu::BufferDescriptor indexBufferDesc{};
    indexBufferDesc.size = indexData.size() * sizeof(uint16_t);
    indexBufferDesc.usage = wgpu::BufferUsage::Index | wgpu::BufferUsage::CopyDst;

    indexBuffer = device.CreateBuffer(&indexBufferDesc);
    particleBuffer = device.CreateBuffer(&particleBufferDesc);
    costBuffer = device.CreateBuffer(&costBufferDesc);
    paramsBuffer = device.CreateBuffer(&paramsDesc);

    /* Bind Group Layouts */
    // Group 0: Params Bind Group
    wgpu::BindGroupLayoutEntry paramsEntry{};
    paramsEntry.binding = 0;
    paramsEntry.visibility = wgpu::ShaderStage::Compute | wgpu::ShaderStage::Vertex;
    paramsEntry.buffer.type = wgpu::BufferBindingType::Uniform;
    paramsEntry.buffer.minBindingSize = sizeof(uint32_t);

    wgpu::BindGroupLayoutDescriptor paramsLayoutDesc{};
    paramsLayoutDesc.entryCount = 1;
    paramsLayoutDesc.entries = &paramsEntry;

    auto paramsBGL = device.CreateBindGroupLayout(&paramsLayoutDesc);

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

    // Group 2: Particles Bind Group
    wgpu::BindGroupLayoutEntry particleEntry{};
    particleEntry.binding = 0;
    particleEntry.visibility = wgpu::ShaderStage::Compute | wgpu::ShaderStage::Vertex;
    particleEntry.buffer.type = wgpu::BufferBindingType::Storage;

    wgpu::BindGroupLayoutDescriptor particleLayoutDesc{};
    particleLayoutDesc.entryCount = 1;
    particleLayoutDesc.entries = &particleEntry;

    auto particleBGL = device.CreateBindGroupLayout(&particleLayoutDesc);

    /* Pipeline Layouts */
    // Solver pipeline
    std::array<wgpu::BindGroupLayout, 2> solverLayouts = {
            paramsBGL,  // group 0
            solverBGL   // group 1
    };

    wgpu::PipelineLayoutDescriptor solverPLDesc{};
    solverPLDesc.bindGroupLayoutCount = solverLayouts.size();
    solverPLDesc.bindGroupLayouts = solverLayouts.data();

    auto solverPipelineLayout = device.CreatePipelineLayout(&solverPLDesc);

    // Physics pipeline
    std::array<wgpu::BindGroupLayout, 3> physicsLayouts = {
            paramsBGL,  // group 0
            solverBGL,
            particleBGL
    };

    wgpu::PipelineLayoutDescriptor physicsPLDesc{};
    physicsPLDesc.bindGroupLayoutCount = physicsLayouts.size();
    physicsPLDesc.bindGroupLayouts = physicsLayouts.data();

    auto physicsPipelineLayout = device.CreatePipelineLayout(&physicsPLDesc);

    // Render pipeline (vertex + fragment shader)
    std::array<wgpu::BindGroupLayout, 2> renderLayouts = {
            paramsBGL,
            particleBGL
    };

    wgpu::PipelineLayoutDescriptor renderPLDesc{};
    renderPLDesc.bindGroupLayoutCount = renderLayouts.size();
    renderPLDesc.bindGroupLayouts = renderLayouts.data();

    auto renderPipelineLayout = device.CreatePipelineLayout(&renderPLDesc);

    /* Bind Groups */
    // group 0 - params
    wgpu::BindGroupEntry paramsEntryBG{};
    paramsEntryBG.binding = 0;
    paramsEntryBG.buffer = paramsBuffer;
    paramsEntryBG.offset = 0;
    paramsEntryBG.size = sizeof(params);

    wgpu::BindGroupDescriptor paramsBGDesc{};
    paramsBGDesc.layout = paramsBGL;
    paramsBGDesc.entryCount = 1;
    paramsBGDesc.entries = &paramsEntryBG;

    paramsBG = device.CreateBindGroup(&paramsBGDesc);

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

    // group 2 - particle
    wgpu::BindGroupEntry particleBGEntry{};
    particleBGEntry.binding = 0;
    particleBGEntry.buffer = particleBuffer;

    wgpu::BindGroupDescriptor particleBGDesc{};
    particleBGDesc.layout = particleBGL;
    particleBGDesc.entryCount = 1;
    particleBGDesc.entries = &particleBGEntry;

    particleBG = device.CreateBindGroup(&particleBGDesc);

    /* Pipelines */
    wgpu::ComputePipelineDescriptor solverPipelineDesc{};
    solverPipelineDesc.layout = solverPipelineLayout;
    solverPipeline = device.CreateComputePipeline(&solverPipelineDesc);

    wgpu::ComputePipelineDescriptor physicsPipelineDesc{};
    solverPipelineDesc.layout = physicsPipelineLayout;
    physicsPipeline = device.CreateComputePipeline(&physicsPipelineDesc);

    wgpu::RenderPipelineDescriptor renderPipelineDesc{};
    renderPipelineDesc.layout = renderPipelineLayout;
    renderPipeline = device.CreateRenderPipeline(&renderPipelineDesc);

    // Upload data
    queue.WriteBuffer(
            indexBuffer,
            0,
            indexData.data(),
            indexData.size() * sizeof(uint16_t)
    );

    queue.WriteBuffer(
            particleBuffer,
            0,
            particleCPUData.data(),
            MAX_CPU_PARTICLES * sizeof(ParticleCPU)
    );

    queue.WriteBuffer(
            costBuffer,
            0,
            cost_buffer.data(),
            cost_buffer.size() * sizeof(COST_ITEM_T)
    );

    queue.WriteBuffer(paramsBuffer, 0, &params, sizeof(params));
}

void InitGraphics() {
    ConfigureSurface();
    CreateRenderPipeline();
}

void Render() {
    particleCPUData = solver->getParticleCPUBuffer(); // this is stupid
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

    // Compute
    {
        auto pass = encoder.BeginComputePass();
        pass.SetPipeline(solverPipeline);
        pass.SetBindGroup(0, paramsBG);
        pass.SetBindGroup(1, solverBG);
        pass.DispatchWorkgroups(16); // TODO: set workgroup count
        pass.End();
    }
    {
        auto pass = encoder.BeginComputePass();
        pass.SetPipeline(physicsPipeline);
        pass.SetBindGroup(0, paramsBG);
        pass.SetBindGroup(1, solverBG);
        pass.SetBindGroup(2, particleBG);
        pass.DispatchWorkgroups(16);
        pass.End();
    }

    // Render
    // Vertex + Fragment shader must use render pass
    wgpu::RenderPassEncoder pass = encoder.BeginRenderPass(&renderpass);
    pass.SetPipeline(renderPipeline);
    // apply index buffer & bind group
    pass.SetIndexBuffer(indexBuffer, wgpu::IndexFormat::Uint16);
    pass.SetBindGroup(0, paramsBG);
    pass.SetBindGroup(1, particleBG);

    // TODO: modify indexCount later
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

    // step physics
    // todo: fire this off as some async/promise every N iters or timeframe if too expensive
    solver->iterate();
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