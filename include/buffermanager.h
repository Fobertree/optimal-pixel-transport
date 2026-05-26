//
// Created by Alexander Liu on 5/23/26.
//

#ifndef OPTIMALPIXELTRANSPORT_BUFFERMANAGER_H
#define OPTIMALPIXELTRANSPORT_BUFFERMANAGER_H

#include <dawn/webgpu_cpp_print.h>
#include <webgpu/webgpu_cpp.h>
#include <type_traits>

#include <span>

struct BGLEntryParams {
    BGLEntryParams(wgpu::ShaderStage visibility, wgpu::BufferBindingType bufferType, uint64_t minBindingSize)
            : visibility(visibility),
              bufferType(bufferType), minBindingSize(minBindingSize) {}

    wgpu::ShaderStage visibility;
    wgpu::BufferBindingType bufferType;
    uint64_t minBindingSize;
};

struct BGLParams {
    std::vector<BGLEntryParams> bglEntriesParams;
};

/*
 * Utilities for render pipeline creation, including:
 * - BG creation
 * - initial write to buffer queues
 *
 */
class BufferManager {
public:
    explicit BufferManager(wgpu::Device &device, wgpu::Queue &queue, wgpu::Buffer &bufA, wgpu::Buffer &bufB)
            : device_(device),
              queue_(queue),
              particleBufA_(bufA),
              particleBufB_(bufB) {

    }

    template<typename T, size_t N>
    [[nodiscard]] wgpu::BindGroupLayout createBGL(const std::array<T, N> &entries) {
        wgpu::BindGroupLayoutDescriptor bglDesc;

        bglDesc.entryCount = entries.size();
        bglDesc.entries = entries.data();

        return device_.CreateBindGroupLayout(&bglDesc);
    }

    template<typename T>
    [[nodiscard]] wgpu::Buffer createWGPUBuffer(wgpu::BufferUsage usage, size_t N) {
        // runtime instead of template arg for emscripten
        wgpu::BufferDescriptor bufDesc{};
        bufDesc.size = N * sizeof(T);
        bufDesc.usage = usage;

        return device_.CreateBuffer(&bufDesc);
    }

    template<std::integral T = int>
    void fillZero(wgpu::Buffer &buf, size_t N) {
        // this should probably be outside class in a namespace
        queue_.WriteBuffer(
                buf,
                0,
                std::vector<T>(N, 0).data(),
                N * sizeof(T)
        );
    }

    template<typename... Buffers>
//    requires (std::same_as<std::remove_cv_t<Buffers>, wgpu::Buffer> && ...)
    [[nodiscard]] std::vector<wgpu::BindGroupEntry> getBGEntries(Buffers &... args) {
        // using vector over array since array would require auto return type
        // can also sizeof...(args)
        const size_t N = sizeof...(args);
        // assume this is cheap enough?
        // worst-case, only particles need to be re-binded (buffer swap) so this should be fine
        const std::array<const wgpu::Buffer, N> list = {args...};
        std::vector<wgpu::BindGroupEntry> bgEntries(N);

        // C++23 has std enumerate
        for (size_t i = 0; i < N; i++) {
            bgEntries[i].binding = i;
            bgEntries[i].buffer = list[i];
        }

        return bgEntries;
    }

    template<typename... BindingType>
    [[nodiscard]] std::vector<wgpu::BindGroupLayoutEntry>
    getBGLayoutEntries(wgpu::ShaderStage shaderStage, BindingType &... args) {
        const size_t N = sizeof...(args);
        const std::array<wgpu::BufferBindingType, N> list = {args...};
        std::vector<wgpu::BindGroupLayoutEntry> bgLayoutEntries(N);

        for (size_t i = 0; i < N; i++) {
            bgLayoutEntries[i] = {
                    .binding = 0,
                    .visibility = shaderStage,
                    .buffer = {.type = list[i]}
            };
        }
        return bgLayoutEntries;
    }

private:
    void swapParticleBuffer(); // TODO: impl

    // might be arguably stupid to have the buffers with ambiguous ownership and just passing by ref
    // maybe just define them here, not in main.cpp
    wgpu::Buffer particleBufA_;
    wgpu::Buffer particleBufB_;
    bool particleSwapBufFlag_{false};

    // debatable whether it's good to have this as private member
    // opting into this bc I don't want to supply an arg everytime
    wgpu::Device device_;
    wgpu::Queue queue_;
};

#endif //OPTIMALPIXELTRANSPORT_BUFFERMANAGER_H
