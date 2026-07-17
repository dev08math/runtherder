#pragma once

#include <memory>
#include <type_traits>
#include <utility>

#include <cuda_runtime.h>

#include <runtherder/check.h>
#include <runtherder/device/check.cuh>

namespace runtherder::device {

/**
 * @brief Owns a non blocking CUDA stream.
 * @note The legacy default stream cannot be captured. Work destined for a graph
 *       launches here.
 */
class CudaStream {
public:
    CudaStream() {
        cudaStream_t raw = nullptr;
        RUNTHERDER_CUDA_CHECK(cudaStreamCreateWithFlags(&raw, cudaStreamNonBlocking));
        stream_.reset(raw);
    }

    CudaStream(CudaStream&&) noexcept            = default;
    CudaStream& operator=(CudaStream&&) noexcept = default;
    CudaStream(const CudaStream&)                = delete;
    CudaStream& operator=(const CudaStream&)     = delete;

    [[nodiscard]] cudaStream_t get() const noexcept { return stream_.get(); }

    void synchronize() const {
        RUNTHERDER_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));
    }

private:
    struct StreamDeleter {
        void operator()(cudaStream_t stream) const noexcept {
            if (stream != nullptr) {
                cudaStreamDestroy(stream);
            }
        }
    };
    std::unique_ptr<std::remove_pointer_t<cudaStream_t>, StreamDeleter> stream_;
};

/**
 * @brief One recorded stream capture, instantiated once and replayed per launch.
 */
class CudaGraph {
public:
    CudaGraph()                                = default;
    CudaGraph(CudaGraph&&) noexcept            = default;
    CudaGraph& operator=(CudaGraph&&) noexcept = default;
    CudaGraph(const CudaGraph&)                = delete;
    CudaGraph& operator=(const CudaGraph&)     = delete;

    [[nodiscard]] bool captured() const noexcept { return exec_ != nullptr; }

    /**
     * @brief Records body onto stream without running it, then instantiates.
     * @note body must issue device work only. A synchronous CUDA call inside it
     *       fails the capture and leaves stream unusable. Every pointer and
     *       launch dimension body passes is frozen into the graph. Follow with
     *       launch() to execute.
     */
    template <typename F>
    void capture(cudaStream_t stream, F&& body) {
        RUNTHERDER_CHECK(!captured(), "CudaGraph captured twice");

        RUNTHERDER_CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        std::forward<F>(body)();

        cudaGraph_t raw_graph = nullptr;
        RUNTHERDER_CUDA_CHECK(cudaStreamEndCapture(stream, &raw_graph));
        graph_.reset(raw_graph);

        cudaGraphExec_t raw_exec = nullptr;
        RUNTHERDER_CUDA_CHECK(cudaGraphInstantiate(&raw_exec, raw_graph, 0));
        exec_.reset(raw_exec);
    }

    void launch(cudaStream_t stream) const {
        RUNTHERDER_CHECK(captured(), "CudaGraph launch before capture");
        RUNTHERDER_CUDA_CHECK(cudaGraphLaunch(exec_.get(), stream));
    }

private:
    struct GraphDeleter {
        void operator()(cudaGraph_t graph) const noexcept {
            if (graph != nullptr) {
                cudaGraphDestroy(graph);
            }
        }
    };
    struct GraphExecDeleter {
        void operator()(cudaGraphExec_t exec) const noexcept {
            if (exec != nullptr) {
                cudaGraphExecDestroy(exec);
            }
        }
    };

    std::unique_ptr<std::remove_pointer_t<cudaGraph_t>, GraphDeleter>         graph_;
    std::unique_ptr<std::remove_pointer_t<cudaGraphExec_t>, GraphExecDeleter> exec_;
};

}  // namespace runtherder::device
