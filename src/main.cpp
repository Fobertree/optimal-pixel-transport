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
wgpu::Buffer particleBuffer;

wgpu::BindGroup bindGroup;

// TODO: cleanup after solver impl
std::vector<ParticleCPU> particleCPUData;

// ptr for runtime polymorphism
SolverBase *solver;

// TODO: [FAILED COMPILATION] - not sure if this is some file token/parsing or scope issue
//std::string shaderCode = read_wgsl_file("particle_shader.wgsl");

std::string shaderCode = R"(
// abstract-float
const particle_size = 0.1;

struct Particle {
    position: vec2f,
    color: vec4f
}

struct VertexInput {
    @builtin(vertex_index) vertex_index : u32,
    @builtin(instance_index) instance_index : u32
};

struct VertexOutput {
    @builtin(position) clip_position: vec4f,
    @location(0) color: vec4f
    // other inter-stage variables alongside @location
};

@group(0) @binding(0)
var<storage, read> particles : array<Particle>;

@vertex
fn vertexMain(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    // look up tesselation for circle in future
    let quad = array(
        vec2f(-particle_size,  particle_size),
        vec2f( particle_size,  particle_size),
        vec2f( particle_size, -particle_size),
        vec2f(-particle_size, -particle_size)
    );

    let particle = particles[in.instance_index];

    let offset = quad[in.vertex_index] * particle_size;

    let world_pos = particle.position + offset;

    out.clip_position = vec4f(world_pos, 0.0, 1.0);
    out.color = particle.color;

    return out;
}

@fragment
fn fragmentMain(in: VertexOutput) -> @location(0) vec4f {
    return in.color;
}
)";

std::vector<uint16_t> indexData = {
        0, 1, 3,
        1, 2, 3
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
    solver = new DefaultLAPJV("img_1.png", "img_6.png", 100, 100);
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
    /* Bind group layout */
    wgpu::BindGroupLayoutEntry entry{};
    entry.binding = 0;
    entry.visibility = wgpu::ShaderStage::Vertex;
    entry.buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
    entry.buffer.minBindingSize = sizeof(ParticleCPU);

    wgpu::BindGroupLayoutDescriptor layoutDesc{};
    layoutDesc.entryCount = 1;
    layoutDesc.entries = &entry;

    auto bindGroupLayout = device.CreateBindGroupLayout(&layoutDesc);

    /* Pipeline layout */
    wgpu::PipelineLayoutDescriptor pipelineLayoutDesc{};
    pipelineLayoutDesc.bindGroupLayoutCount = 1;
    pipelineLayoutDesc.bindGroupLayouts = &bindGroupLayout;

    auto pipelineLayout = device.CreatePipelineLayout(&pipelineLayoutDesc);

    /* Shader */
    wgpu::ShaderSourceWGSL wgsl{{.code=shaderCode.c_str()}};

    wgpu::ShaderModuleDescriptor shaderModuleDescriptor{.nextInChain = &wgsl};
    wgpu::ShaderModule shaderModule = device.CreateShaderModule(&shaderModuleDescriptor);

    /* Pipeline */
    wgpu::ColorTargetState colorTargetState{.format=format};

    wgpu::FragmentState fragmentState{
            .module = shaderModule,
            .entryPoint= "fragmentMain",
            .targetCount=1,
            .targets=&colorTargetState
    };
    wgpu::RenderPipelineDescriptor descriptor{
            .layout = pipelineLayout,
            .vertex={.module = shaderModule, .entryPoint = "vertexMain"},
            .primitive={.topology = wgpu::PrimitiveTopology::TriangleList},
            .fragment=&fragmentState,
    };

    pipeline = device.CreateRenderPipeline(&descriptor);

    /* Buffers */

    // particle buffer
    wgpu::BufferDescriptor particleBufferDesc{};
    particleBufferDesc.size = MAX_CPU_PARTICLES * sizeof(ParticleCPU);
    particleBufferDesc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;

    particleBuffer = device.CreateBuffer(&particleBufferDesc);

    // index buffer
    wgpu::BufferDescriptor indexBufferDesc{};
    indexBufferDesc.size = indexData.size() * sizeof(uint16_t);
    indexBufferDesc.usage = wgpu::BufferUsage::Index | wgpu::BufferUsage::CopyDst;

    indexBuffer = device.CreateBuffer(&indexBufferDesc);

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

    /* Bind Group */
    wgpu::BindGroupEntry bgEntry{};
    bgEntry.binding = 0;
    bgEntry.buffer = particleBuffer;
    bgEntry.offset = 0;
    bgEntry.size = MAX_CPU_PARTICLES * sizeof(ParticleCPU);

    wgpu::BindGroupDescriptor bgDesc{};
    bgDesc.layout = bindGroupLayout;
    bgDesc.entryCount = 1;
    bgDesc.entries = &bgEntry;

    bindGroup = device.CreateBindGroup(&bgDesc);

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
    wgpu::RenderPassEncoder pass = encoder.BeginRenderPass(&renderpass);
    pass.SetPipeline(pipeline);
    // apply index buffer & bind group
    pass.SetIndexBuffer(indexBuffer, wgpu::IndexFormat::Uint16);
    pass.SetBindGroup(0, bindGroup);

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