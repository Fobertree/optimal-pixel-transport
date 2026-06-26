//
// Created by Alexander Liu on 5/23/26.
//

#ifndef OPTIMALPIXELTRANSPORT_BUFFERMANAGER_H
#define OPTIMALPIXELTRANSPORT_BUFFERMANAGER_H

#include <dawn/webgpu_cpp_print.h>
#include <webgpu/webgpu_cpp.h>
#include <type_traits>

#include <span>
#include <algorithm>
#include <array>
#include <cstring>

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
    explicit BufferManager() = default; // tmp, need to figure out better approach on global scope
    explicit BufferManager(const wgpu::Device &device, const wgpu::Queue &queue)
            : device_(device),
              queue_(queue) {
        assert(device_);
        printf("device=%p\n", device_.Get());
    }

    template<typename T, size_t N>
    [[nodiscard]] wgpu::BindGroupLayout createBGL(const std::array<T, N> &entries, const wgpu::StringView &label) {
        wgpu::BindGroupLayoutDescriptor bglDesc;

        bglDesc.entryCount = entries.size();
        bglDesc.entries = entries.data();
        bglDesc.label = label;

        return device_.CreateBindGroupLayout(&bglDesc);
    }

    template<typename T>
    [[nodiscard]] wgpu::Buffer
    createWGPUBuffer(wgpu::BufferUsage usage, size_t N, const wgpu::StringView &label = "NONE") {
        // runtime instead of template arg for emscripten
        assert(N > 0);
        if (!device_) {
            puts("device invalid");
        }

//        puts("create wgpu buffer");
        wgpu::BufferDescriptor bufDesc{};
        bufDesc.size = N * sizeof(T);
        bufDesc.usage = usage;
        bufDesc.label = label;

        return device_.CreateBuffer(&bufDesc);
    }

    void fillZeroBytes(wgpu::Buffer &buf, size_t byteSize) {
        constexpr size_t kChunkBytes = 65536;
        std::array<uint8_t, kChunkBytes> zeros{};
        size_t offset = 0;
        while (offset < byteSize) {
            const size_t chunk = std::min(kChunkBytes, byteSize - offset);
            queue_.WriteBuffer(buf, offset, zeros.data(), chunk);
            offset += chunk;
        }
    }

    template<std::integral T = int>
    void fillVal(wgpu::Buffer &buf, size_t N, T val) {
        constexpr size_t kMaxChunk = 16384;
        const size_t chunkCount = std::min(N, kMaxChunk);
        const std::vector<T> chunk(chunkCount, val);
        size_t written = 0;
        while (written < N) {
            const size_t count = std::min(N - written, chunk.size());
            queue_.WriteBuffer(buf, written * sizeof(T), chunk.data(), count * sizeof(T));
            written += count;
        }
    }

    template<std::integral T = int>
    void fillZero(wgpu::Buffer &buf, size_t N) {
        fillVal(buf, N, T{0});
    }

    static std::vector<wgpu::BindGroupEntry>
    getBGEntries(std::initializer_list<wgpu::Buffer> args) {
        std::vector<wgpu::BindGroupEntry> bgEntries(args.size());

        size_t i = 0;
        for (auto const &buffer: args) {
            bgEntries[i].binding = i;
            bgEntries[i].buffer = buffer;
            bgEntries[i].offset = 0;
            bgEntries[i].size = buffer.GetSize();
            ++i;
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
    // debatable whether it's good to have this as private member
    // opting into this bc I don't want to supply an arg everytime
    wgpu::Device device_;
    wgpu::Queue queue_;
};

#endif //OPTIMALPIXELTRANSPORT_BUFFERMANAGER_H
