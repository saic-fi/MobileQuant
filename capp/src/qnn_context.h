#ifndef LIBLLMOD_QNN_CONTEXT_H
#define LIBLLMOD_QNN_CONTEXT_H

#include "utils.h"

#include <list>
#include <span>
#include <vector>
#include <string>
#include <memory>
#include <optional>
#include <functional>
#include <mutex>

#include <QnnInterface.h>
#include <QnnWrapperUtils.hpp>
#include <System/QnnSystemInterface.h>
#include <HTP/QnnHtpDevice.h>


namespace libllmod {

class QnnApi;
class QnnTensor;
class QnnGraph;
class QnnContext;
class QnnBackend;

struct graph_slot {
    QnnGraph& graph;
    Qnn_Tensor_t& target;
    const QnnTensor* current_tensor;
};

template <class T>
using qnn_hnd = std::shared_ptr<std::remove_pointer_t<T>>;
using graph_ref = std::add_lvalue_reference_t<QnnGraph>;
using graph_refs = std::list<std::reference_wrapper<QnnGraph>>;
using graph_list = std::list<QnnGraph>;
using tensor_list = std::list<QnnTensor>;
using graph_slots = std::vector<graph_slot>;


#define RPCMEM_HEAP_ID_SYSTEM 25
#define RPCMEM_DEFAULT_FLAGS 1

typedef void* (*RpcMemAllocFn_t)(int, uint32_t, int);
typedef void (*RpcMemFreeFn_t)(void*);
typedef int (*RpcMemToFdFn_t)(void*);


// Graph Related Function Handle Types
typedef qnn_wrapper_api::ModelError_t (*ComposeGraphsFnHandleType_t)(Qnn_BackendHandle_t, QNN_INTERFACE_VER_TYPE, Qnn_ContextHandle_t, const qnn_wrapper_api::GraphConfigInfo_t **, const uint32_t, qnn_wrapper_api::GraphInfo_t ***, uint32_t *, bool, QnnLog_Callback_t, QnnLog_Level_t);
typedef Qnn_ErrorHandle_t (*FreeGraphInfoFnHandleType_t)(qnn_wrapper_api::GraphInfo_t***, uint32_t);


enum class QnnBackendType : int {
    CPU,
    GPU,
    DSP,
    HTP,
    HTA
};


class QnnApi {
public:
    static std::shared_ptr<QnnApi> get(QnnBackendType backend);
    virtual ~QnnApi();

    auto get_backend_type() const { return backend; }
    auto get_interface() const { return interface; }

    QnnDevice_Infrastructure_t get_device_infrastructure() const;

    qnn_hnd<Qnn_BackendHandle_t> create_backend(const QnnBackend_Config_t** cfg) const;
    qnn_hnd<Qnn_DeviceHandle_t> create_device(const QnnDevice_Config_t** cfg) const;
    qnn_hnd<Qnn_ContextHandle_t> create_context(Qnn_BackendHandle_t backend, Qnn_DeviceHandle_t device, const QnnContext_Config_t** cfg) const;
    qnn_hnd<Qnn_ContextHandle_t> create_context(std::vector<unsigned char> const& buffer, Qnn_BackendHandle_t backend, Qnn_DeviceHandle_t device, const QnnContext_Config_t** cfg) const;
    qnn_hnd<Qnn_ContextHandle_t> create_context(mmap_t const& buffer, Qnn_BackendHandle_t backend, Qnn_DeviceHandle_t device, const QnnContext_Config_t** cfg) const;

    void register_op_package(std::string const& package_path, std::string const& package_interface_provider) const;

    qnn_hnd<QnnSystemContext_Handle_t> create_system_context();
    QnnSystemContext_BinaryInfo_t const& get_binary_info(QnnSystemContext_Handle_t ctx, std::vector<unsigned char>& buffer) const;
    QnnSystemContext_BinaryInfo_t const& get_binary_info(QnnSystemContext_Handle_t ctx, mmap_t& buffer) const;
    Qnn_GraphHandle_t retrieve_graph(Qnn_ContextHandle_t context, const char* graph_name) const;
    void finalize_graph(Qnn_GraphHandle_t hnd) const;

    void set_graph_config(Qnn_GraphHandle_t graph, const QnnGraph_Config_t** cfg) const;

    std::pair<std::shared_ptr<void>,int> allocate_ion(uint32_t size);
    qnn_hnd<Qnn_MemHandle_t> mem_register(Qnn_ContextHandle_t ctx, Qnn_MemDescriptor_t desc);

    bool has_ion() const { return bool(cdsp_dl); }

    void execute_graph(Qnn_GraphHandle_t graph, std::span<Qnn_Tensor_t> const& inputs, std::span<Qnn_Tensor_t>& outputs);
    void execute_graph_async(Qnn_GraphHandle_t graph, std::span<Qnn_Tensor_t> const& inputs, std::span<Qnn_Tensor_t>& outputs, Qnn_NotifyFn_t notify, void* notify_params);

private:
    QnnApi(QnnBackendType backend);

    QnnBackendType backend;
    std::shared_ptr<void> dl;
    std::shared_ptr<void> system_dl;
    std::shared_ptr<void> cdsp_dl ;
    qnn_hnd<Qnn_LogHandle_t> log_hnd;
    bool has_system_interface = false;

    QNN_INTERFACE_VER_TYPE interface;
    QNN_SYSTEM_INTERFACE_VER_TYPE system_interface;
    RpcMemAllocFn_t rpcmem_alloc = nullptr;
    RpcMemFreeFn_t rpcmem_free = nullptr;
    RpcMemToFdFn_t rpcmem_to_fd = nullptr;
};


class QnnTensor {
    friend class QnnGraph;
public:
    QnnTensor(QnnTensor&& other);
    QnnTensor(QnnTensor const& ohter) = delete;
    ~QnnTensor();

    static uint32_t get_num_elements(Qnn_Tensor_t const& t, unsigned int batch_size=1);
    static uint8_t get_element_size(Qnn_Tensor_t const& t);
    static bool is_quantized(Qnn_Tensor_t const& t);
    static bool is_floating_point(Qnn_Tensor_t const& t);

    void activate() const;
    void deactivate() const;

    uint32_t get_num_elements(unsigned int batch_size) const { return get_num_elements(slot.target, batch_size); }
    uint8_t get_element_size() const { return get_element_size(slot.target); }
    bool is_quantized() const { return is_quantized(slot.target); }
    bool is_floating_point() const { return is_floating_point(slot.target); }

    void set_data(std::vector<float>    const& buffer, bool accum=false, int src_start=0, int src_end=-1, int dst_start=0);
    void set_data(std::vector<uint16_t> const& buffer, bool accum=false, int src_start=0, int src_end=-1, int dst_start=0);
    void set_data(std::vector<uint32_t> const& buffer, bool accum=false, int src_start=0, int src_end=-1, int dst_start=0);
    void set_data(std::vector<float> const& buffer, float scale, bool accum=false, int src_start=0, int src_end=-1, int dst_start=0);

    void get_data(std::vector<float>& buffer,       bool accum=false, int src_start=0, int src_end=-1, int dst_start=0) const; 
    void get_data(std::vector<uint16_t>& buffer,    bool accum=false, int src_start=0, int src_end=-1, int dst_start=0) const;
    void get_data(std::vector<uint32_t>& buffer,    bool accum=false, int src_start=0, int src_end=-1, int dst_start=0) const;
    void get_data(std::vector<float>& buffer, float scale, bool accum=false, int src_start=0, int src_end=-1, int dst_start=0) const;

    std::string get_slot_name() const;

    void* get_data_ptr() { return (void *) data.get(); }

private:
    QnnTensor(QnnApi& api, Qnn_ContextHandle_t ctx, graph_slot& slot, unsigned int batch_size=1); //allocate new
    QnnTensor(QnnTensor const& other, graph_slot& slot, bool strict_shape); //reuse the same allocation for different input/output slot

    bool is_ion = false;
    unsigned int batch_size = 0;

    std::shared_ptr<void> data;
    uint32_t data_size = 0;
    int data_fd = -1;
    qnn_hnd<Qnn_MemHandle_t> data_hnd;

    graph_slot& slot;
};


class QnnGraph {
    friend class QnnBackend;
private:
    struct CtorToken {
        Qnn_GraphHandle_t graph;
        std::shared_ptr<QnnContext> ctx;
        std::shared_ptr<QnnApi> api;
        const char* orig_name;
        std::span<Qnn_Tensor_t> inputs;
        std::span<Qnn_Tensor_t> outputs;
    };

public:
    explicit QnnGraph(CtorToken&& token);
    QnnGraph(QnnGraph&& other) = delete;
    ~QnnGraph();

    QnnTensor allocate_input(unsigned int idx, unsigned batch=1, bool activate=true);
    QnnTensor attach_input(unsigned int idx, QnnTensor const& t, bool activate=true, bool strict_shape=true);

    QnnTensor allocate_output(unsigned int idx, unsigned batch=1, bool activate=true);
    QnnTensor attach_output(unsigned int idx, QnnTensor const& t, bool activate=true, bool strict_shape=true);

    auto get_num_inputs() const { return inputs.size(); }
    auto get_num_outputs() const { return outputs.size(); }

    void verify();
    void execute();
    void execute_async(std::function<void(void*, Qnn_NotifyStatus_t)> notify = std::function<void(void*, Qnn_NotifyStatus_t)>(), void* notify_param = nullptr);

    void set_name(std::string s) { name.swap(s); }
    auto const& get_name() const { return name; }

private:
    const char* orig_name;
    std::span<Qnn_Tensor_t> inputs;
    std::span<Qnn_Tensor_t> outputs;

    graph_slots input_slots;
    graph_slots output_slots;

    Qnn_GraphHandle_t graph;
    std::shared_ptr<QnnContext> ctx;
    std::shared_ptr<QnnApi> api;

    std::string name;
};


class QnnContext {
    friend class QnnBackend;
public:
    QnnContext(QnnContext const& other) = delete;
    QnnContext(QnnContext&& other) = default;
    ~QnnContext();

    Qnn_ContextHandle_t get_handle() const { return ctx.get(); }

private:
    QnnContext(qnn_hnd<Qnn_ContextHandle_t> ctx, std::shared_ptr<void> dl = nullptr, std::function<void()> free_fn = nullptr);

    qnn_hnd<Qnn_ContextHandle_t> ctx;
    std::shared_ptr<void> dl;
    std::function<void()> free_fn;
};



class QnnBackend {
public:
    QnnBackend(QnnBackendType backend, std::string device_type = "8gen3", std::list<std::string> const& op_packages = std::list<std::string>(), bool burst = true);
    ~QnnBackend();

    graph_list load_context(std::string const& context_blob);
    graph_list load_model(std::string const& model_so);

    graph_list load_graphs(std::string const& file, bool is_cached);

    void start_burst();
    void end_burst();
    auto get_backend_type() const { return api->get_backend_type(); }

private:
    std::shared_ptr<QnnApi> api;
    std::mutex api_mutex;

    qnn_hnd<Qnn_BackendHandle_t> backend_hnd;
    qnn_hnd<Qnn_DeviceHandle_t> device_hnd;

    std::optional<QnnHtpDevice_PerfInfrastructure_t> _htp_perf_infra;
    std::optional<uint32_t> _htp_power_config_id;
    std::optional<QnnHtpPerfInfrastructure_PowerConfig_t> _htp_burst_power_config;
    std::optional<QnnHtpPerfInfrastructure_PowerConfig_t> _htp_normal_power_config;

    bool burst = true;
    std::string device_type;

    void _init_backend();
    void _init_device();
    void _init_performance();
};

}

#endif // LIBLLMOD_QNN_CONTEXT_H
