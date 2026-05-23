#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cctype>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits.h>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <dirent.h>
#include <thread>
#include <unistd.h>
#include <vector>

namespace {

constexpr double kMaxMemoryUsagePercent = 80.0;
constexpr unsigned int kMaxGpuUtilizationPercent = 30;
constexpr double kTargetFreeMemoryRatio = 0.60;
constexpr std::size_t kAlignmentBytes = 256ULL * 1024ULL * 1024ULL;
constexpr int kNvmlSuccess = 0;
constexpr const char *kDaemonFlag = "--internal-daemon";

std::atomic<bool> keep_running{true};

using nvmlReturn_t = int;
using nvmlDevice_t = struct nvmlDevice_st *;

struct nvmlMemory_t {
    unsigned long long total;
    unsigned long long free;
    unsigned long long used;
};

struct nvmlUtilization_t {
    unsigned int gpu;
    unsigned int memory;
};

struct Options {
    enum class Mode {
        Foreground,
        Up,
        Down,
        Status,
    };

    Mode mode = Mode::Foreground;
    std::vector<unsigned int> gpus;
    std::string binary_name;
    std::string binary_path;
    bool internal_daemon = false;
};

struct GpuInfo {
    unsigned int physical_index = 0;
    int cuda_device = 0;
    double total_gb = 0.0;
    double free_gb = 0.0;
    double target_gb = 0.0;
    unsigned int utilization = 0;
};

struct DaemonProcess {
    pid_t pid = 0;
    std::vector<unsigned int> gpus;
};

struct GpuMemoryUsage {
    unsigned int physical_index = 0;
    double used_gb = 0.0;
    double total_gb = 0.0;
    bool valid = false;
};

struct NvmlApi {
    void *handle = nullptr;
    nvmlReturn_t (*init)() = nullptr;
    nvmlReturn_t (*shutdown)() = nullptr;
    nvmlReturn_t (*device_get_count)(unsigned int *) = nullptr;
    nvmlReturn_t (*device_get_handle_by_index)(unsigned int, nvmlDevice_t *) = nullptr;
    nvmlReturn_t (*device_get_memory_info)(nvmlDevice_t, nvmlMemory_t *) = nullptr;
    nvmlReturn_t (*device_get_utilization_rates)(nvmlDevice_t, nvmlUtilization_t *) = nullptr;
    const char *(*error_string)(nvmlReturn_t) = nullptr;
};

__global__ void burn_kernel(float *data, std::size_t elements, unsigned long long seed) {
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    std::size_t stride = blockDim.x * gridDim.x;
    float x = static_cast<float>((idx + seed) % 1024) * 0.001f;

    for (std::size_t i = idx; i < elements; i += stride) {
        float v = data[i] + x;
        #pragma unroll 64
        for (int j = 0; j < 1024; ++j) {
            v = v * 1.000001f + 0.000001f;
            v = v - static_cast<float>(j & 7) * 0.0000001f;
        }
        data[i] = v;
    }
}

std::string basename(const char *path) {
    const char *slash = std::strrchr(path, '/');
    return slash == nullptr ? std::string(path) : std::string(slash + 1);
}

std::string resolve_path(const std::string &path) {
    char resolved[PATH_MAX] = {};
    if (realpath(path.c_str(), resolved) != nullptr) {
        return resolved;
    }
    return path;
}

std::string resolve_executable_path(const char *argv0) {
    if (argv0 == nullptr || *argv0 == '\0') {
        return "";
    }

    std::string name(argv0);
    if (name.find('/') != std::string::npos) {
        return resolve_path(name);
    }

    const char *path_env = std::getenv("PATH");
    if (path_env == nullptr) {
        return name;
    }

    std::stringstream ss(path_env);
    std::string dir;
    while (std::getline(ss, dir, ':')) {
        if (dir.empty()) {
            dir = ".";
        }
        std::string candidate = dir + "/" + name;
        if (access(candidate.c_str(), X_OK) == 0) {
            return resolve_path(candidate);
        }
    }

    return name;
}

std::string sanitize_binary_name(std::string name) {
    if (name.empty()) {
        return "gpu-dryrun";
    }
    for (char &ch : name) {
        if (!(std::isalnum(static_cast<unsigned char>(ch)) || ch == '-' || ch == '_' || ch == '.')) {
            ch = '_';
        }
    }
    return name;
}

void handle_signal(int) {
    keep_running.store(false);
}

void fail(const std::string &message) {
    std::cerr << message << std::endl;
    std::exit(1);
}

template <typename T>
T load_symbol(void *handle, const char *name) {
    dlerror();
    void *symbol = dlsym(handle, name);
    const char *error = dlerror();
    if (error != nullptr || symbol == nullptr) {
        fail(std::string("failed to load NVML symbol ") + name + ": " + (error == nullptr ? "not found" : error));
    }
    return reinterpret_cast<T>(symbol);
}

template <typename T>
T load_symbol_any(void *handle, std::initializer_list<const char *> names) {
    for (const char *name : names) {
        dlerror();
        void *symbol = dlsym(handle, name);
        if (dlerror() == nullptr && symbol != nullptr) {
            return reinterpret_cast<T>(symbol);
        }
    }
    fail("failed to load required NVML symbol");
    return nullptr;
}

NvmlApi load_nvml() {
    NvmlApi api;
    api.handle = dlopen("libnvidia-ml.so.1", RTLD_NOW);
    if (api.handle == nullptr) {
        api.handle = dlopen("libnvidia-ml.so", RTLD_NOW);
    }
    if (api.handle == nullptr) {
        fail(std::string("failed to load NVML: ") + dlerror());
    }

    api.init = load_symbol_any<nvmlReturn_t (*)()>(api.handle, {"nvmlInit_v2", "nvmlInit"});
    api.shutdown = load_symbol<nvmlReturn_t (*)()>(api.handle, "nvmlShutdown");
    api.device_get_count = load_symbol_any<nvmlReturn_t (*)(unsigned int *)>(
        api.handle, {"nvmlDeviceGetCount_v2", "nvmlDeviceGetCount"});
    api.device_get_handle_by_index = load_symbol_any<nvmlReturn_t (*)(unsigned int, nvmlDevice_t *)>(
        api.handle, {"nvmlDeviceGetHandleByIndex_v2", "nvmlDeviceGetHandleByIndex"});
    api.device_get_memory_info = load_symbol<nvmlReturn_t (*)(nvmlDevice_t, nvmlMemory_t *)>(
        api.handle, "nvmlDeviceGetMemoryInfo");
    api.device_get_utilization_rates = load_symbol<nvmlReturn_t (*)(nvmlDevice_t, nvmlUtilization_t *)>(
        api.handle, "nvmlDeviceGetUtilizationRates");
    api.error_string = load_symbol<const char *(*)(nvmlReturn_t)>(api.handle, "nvmlErrorString");
    return api;
}

void check_nvml(const NvmlApi &api, nvmlReturn_t result, const std::string &what) {
    if (result != kNvmlSuccess) {
        fail(what + ": " + api.error_string(result));
    }
}

void close_nvml(NvmlApi &api) {
    if (api.handle != nullptr) {
        dlclose(api.handle);
        api.handle = nullptr;
    }
}

void check_cuda(cudaError_t result, const std::string &what) {
    if (result != cudaSuccess) {
        fail(what + ": " + cudaGetErrorString(result));
    }
}

std::vector<unsigned int> parse_gpu_list(const std::string &value) {
    std::vector<unsigned int> devices;
    std::stringstream ss(value);
    std::string item;

    while (std::getline(ss, item, ',')) {
        if (item.empty()) {
            continue;
        }
        char *end = nullptr;
        errno = 0;
        unsigned long parsed = std::strtoul(item.c_str(), &end, 10);
        if (errno != 0 || end == item.c_str() || *end != '\0') {
            fail("GPU list must contain numeric indices, got: " + item);
        }
        devices.push_back(static_cast<unsigned int>(parsed));
    }

    return devices;
}

void usage(const char *argv0) {
    std::string name = basename(argv0);
    std::cout
        << "Usage:\n"
        << "  " << name << " [--gpus 0,1]\n"
        << "  " << name << " up <GPU_IDs>\n"
        << "  " << name << " down\n"
        << "  " << name << " status\n\n"
        << "Examples:\n"
        << "  " << name << " --gpus 0,1\n"
        << "  " << name << " up 0,1\n"
        << "  " << name << " down\n";
}

Options parse_args(int argc, char **argv) {
    Options options;
    options.binary_name = sanitize_binary_name(basename(argv[0]));
    options.binary_path = resolve_executable_path(argv[0]);

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            usage(argv[0]);
            std::exit(0);
        }
        if (arg == "up") {
            options.mode = Options::Mode::Up;
            if (i + 1 >= argc) {
                fail("missing GPU list after 'up'");
            }
            options.gpus = parse_gpu_list(argv[++i]);
            continue;
        }
        if (arg == "down") {
            options.mode = Options::Mode::Down;
            continue;
        }
        if (arg == "status") {
            options.mode = Options::Mode::Status;
            continue;
        }
        if (arg == "--gpus") {
            if (i + 1 >= argc) {
                fail("missing value after --gpus");
            }
            options.gpus = parse_gpu_list(argv[++i]);
            continue;
        }
        if (arg == kDaemonFlag) {
            options.internal_daemon = true;
            options.mode = Options::Mode::Foreground;
            continue;
        }
        fail("unknown argument: " + arg);
    }

    return options;
}

std::vector<unsigned int> all_physical_gpus(const NvmlApi &nvml) {
    unsigned int count = 0;
    check_nvml(nvml, nvml.device_get_count(&count), "nvmlDeviceGetCount failed");

    std::vector<unsigned int> gpus;
    gpus.reserve(count);
    for (unsigned int i = 0; i < count; ++i) {
        gpus.push_back(i);
    }
    return gpus;
}

std::vector<GpuInfo> get_available_gpus(const std::vector<unsigned int> &requested) {
    NvmlApi nvml = load_nvml();
    check_nvml(nvml, nvml.init(), "nvmlInit failed");

    std::vector<unsigned int> candidates = requested.empty() ? all_physical_gpus(nvml) : requested;
    std::vector<GpuInfo> gpus;

    unsigned int count = 0;
    check_nvml(nvml, nvml.device_get_count(&count), "nvmlDeviceGetCount failed");

    for (unsigned int physical_index : candidates) {
        if (physical_index >= count) {
            std::cerr << "GPU index out of range: " << physical_index << " (count: " << count << ")" << std::endl;
            continue;
        }

        nvmlDevice_t handle;
        nvmlMemory_t memory;
        nvmlUtilization_t utilization;
        check_nvml(nvml, nvml.device_get_handle_by_index(physical_index, &handle), "nvmlDeviceGetHandleByIndex failed");
        check_nvml(nvml, nvml.device_get_memory_info(handle, &memory), "nvmlDeviceGetMemoryInfo failed");
        check_nvml(nvml, nvml.device_get_utilization_rates(handle, &utilization), "nvmlDeviceGetUtilizationRates failed");

        double total_gb = static_cast<double>(memory.total) / (1024.0 * 1024.0 * 1024.0);
        double free_gb = static_cast<double>(memory.free) / (1024.0 * 1024.0 * 1024.0);
        double memory_usage = (total_gb - free_gb) / total_gb * 100.0;

        if (memory_usage < kMaxMemoryUsagePercent && utilization.gpu < kMaxGpuUtilizationPercent) {
            gpus.push_back({
                physical_index,
                static_cast<int>(physical_index),
                total_gb,
                free_gb,
                free_gb * kTargetFreeMemoryRatio,
                utilization.gpu,
            });
        }
    }

    check_nvml(nvml, nvml.shutdown(), "nvmlShutdown failed");
    close_nvml(nvml);

    std::sort(gpus.begin(), gpus.end(), [](const GpuInfo &a, const GpuInfo &b) {
        return a.free_gb > b.free_gb;
    });
    return gpus;
}

std::size_t target_bytes(double target_gb) {
    std::size_t bytes = static_cast<std::size_t>(target_gb * 1024.0 * 1024.0 * 1024.0);
    bytes = bytes / kAlignmentBytes * kAlignmentBytes;
    return std::max<std::size_t>(bytes, kAlignmentBytes);
}

float *allocate_with_backoff(std::size_t &bytes) {
    float *ptr = nullptr;
    while (bytes >= kAlignmentBytes) {
        cudaError_t result = cudaMalloc(&ptr, bytes);
        if (result == cudaSuccess) {
            return ptr;
        }
        cudaGetLastError();
        bytes = bytes * 9 / 10;
        bytes = bytes / kAlignmentBytes * kAlignmentBytes;
    }
    check_cuda(cudaErrorMemoryAllocation, "cudaMalloc failed");
    return nullptr;
}

void worker(GpuInfo gpu) {
    check_cuda(cudaSetDevice(gpu.cuda_device), "cudaSetDevice failed");

    std::size_t bytes = target_bytes(gpu.target_gb);
    float *buffer = allocate_with_backoff(bytes);
    std::size_t elements = bytes / sizeof(float);
    check_cuda(cudaMemset(buffer, 0, bytes), "cudaMemset failed");

    int multiprocessors = 0;
    check_cuda(cudaDeviceGetAttribute(&multiprocessors, cudaDevAttrMultiProcessorCount, gpu.cuda_device),
               "cudaDeviceGetAttribute failed");

    int block_size = 256;
    int grid_size = std::max(1, multiprocessors * 8);
    double allocated_gb = static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0);

    std::cout << "GPU " << gpu.physical_index << " started: allocated "
              << std::fixed << std::setprecision(2) << allocated_gb
              << "GB, CUDA device " << gpu.cuda_device << std::endl;

    unsigned long long iter = 0;
    while (keep_running.load()) {
        burn_kernel<<<grid_size, block_size>>>(buffer, elements, iter++);
        cudaError_t launch_status = cudaGetLastError();
        if (launch_status != cudaSuccess) {
            std::cerr << "GPU " << gpu.physical_index << " kernel launch failed: "
                      << cudaGetErrorString(launch_status) << std::endl;
            keep_running.store(false);
            break;
        }
        cudaError_t sync_status = cudaDeviceSynchronize();
        if (sync_status != cudaSuccess) {
            std::cerr << "GPU " << gpu.physical_index << " kernel sync failed: "
                      << cudaGetErrorString(sync_status) << std::endl;
            keep_running.store(false);
            break;
        }
    }

    cudaFree(buffer);
    cudaDeviceReset();
}

int run_foreground(const Options &options) {
    std::signal(SIGINT, handle_signal);
    std::signal(SIGTERM, handle_signal);

    std::vector<GpuInfo> gpus = get_available_gpus(options.gpus);
    if (gpus.empty()) {
        std::cout << "No suitable GPUs found!" << std::endl;
        return 0;
    }

    std::cout << "Found " << gpus.size() << " suitable GPUs:" << std::endl;
    for (const GpuInfo &gpu : gpus) {
        std::cout << "GPU " << gpu.physical_index << ": "
                  << std::fixed << std::setprecision(2) << gpu.total_gb << "GB total, "
                  << gpu.free_gb << "GB free, Utilization: "
                  << gpu.utilization << "%" << std::endl;
    }

    std::vector<std::thread> threads;
    threads.reserve(gpus.size());
    for (const GpuInfo &gpu : gpus) {
        threads.emplace_back(worker, gpu);
    }

    for (std::thread &thread : threads) {
        thread.join();
    }

    return 0;
}

bool is_digits_only(const char *s) {
    if (s == nullptr || *s == '\0') {
        return false;
    }
    for (const char *p = s; *p != '\0'; ++p) {
        if (!std::isdigit(static_cast<unsigned char>(*p))) {
            return false;
        }
    }
    return true;
}

std::vector<std::string> read_cmdline(pid_t pid) {
    std::vector<std::string> args;
    std::string path = "/proc/" + std::to_string(pid) + "/cmdline";
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        return args;
    }

    std::string data((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    std::size_t start = 0;
    while (start < data.size()) {
        std::size_t stop = data.find('\0', start);
        if (stop == std::string::npos) {
            stop = data.size();
        }
        if (stop > start) {
            args.emplace_back(data.substr(start, stop - start));
        }
        start = stop + 1;
    }
    return args;
}

std::string read_proc_exe(pid_t pid) {
    std::string link = "/proc/" + std::to_string(pid) + "/exe";
    char buffer[PATH_MAX] = {};
    ssize_t len = readlink(link.c_str(), buffer, sizeof(buffer) - 1);
    if (len < 0) {
        return "";
    }
    buffer[len] = '\0';
    return std::string(buffer);
}

bool has_daemon_flag(const std::vector<std::string> &args) {
    for (const std::string &arg : args) {
        if (arg == kDaemonFlag) {
            return true;
        }
    }
    return false;
}

std::vector<unsigned int> parse_gpus_from_args(const std::vector<std::string> &args) {
    for (std::size_t i = 0; i + 1 < args.size(); ++i) {
        if (args[i] == "--gpus") {
            return parse_gpu_list(args[i + 1]);
        }
    }
    return {};
}

std::vector<DaemonProcess> find_daemons(const Options &options) {
    std::vector<DaemonProcess> daemons;
    DIR *proc = opendir("/proc");
    if (proc == nullptr) {
        fail("failed to open /proc: " + std::string(std::strerror(errno)));
    }

    struct dirent *entry = nullptr;
    while ((entry = readdir(proc)) != nullptr) {
        if (!is_digits_only(entry->d_name)) {
            continue;
        }

        pid_t pid = static_cast<pid_t>(std::strtol(entry->d_name, nullptr, 10));
        if (pid <= 0 || pid == getpid()) {
            continue;
        }

        if (resolve_path(read_proc_exe(pid)) != options.binary_path) {
            continue;
        }

        std::vector<std::string> args = read_cmdline(pid);
        if (has_daemon_flag(args)) {
            daemons.push_back({pid, parse_gpus_from_args(args)});
        }
    }

    closedir(proc);
    return daemons;
}

std::vector<GpuMemoryUsage> get_gpu_memory_usage(const std::vector<unsigned int> &gpus) {
    std::vector<GpuMemoryUsage> usage;
    usage.reserve(gpus.size());
    if (gpus.empty()) {
        return usage;
    }

    NvmlApi nvml = load_nvml();
    check_nvml(nvml, nvml.init(), "nvmlInit failed");

    unsigned int count = 0;
    check_nvml(nvml, nvml.device_get_count(&count), "nvmlDeviceGetCount failed");
    for (unsigned int physical_index : gpus) {
        if (physical_index >= count) {
            usage.push_back({physical_index, 0.0, 0.0, false});
            continue;
        }

        nvmlDevice_t handle;
        nvmlMemory_t memory;
        check_nvml(nvml, nvml.device_get_handle_by_index(physical_index, &handle), "nvmlDeviceGetHandleByIndex failed");
        check_nvml(nvml, nvml.device_get_memory_info(handle, &memory), "nvmlDeviceGetMemoryInfo failed");
        double used_gb = static_cast<double>(memory.used) / (1024.0 * 1024.0 * 1024.0);
        double total_gb = static_cast<double>(memory.total) / (1024.0 * 1024.0 * 1024.0);
        usage.push_back({physical_index, used_gb, total_gb, true});
    }

    check_nvml(nvml, nvml.shutdown(), "nvmlShutdown failed");
    close_nvml(nvml);
    return usage;
}

void redirect_stdio_to_null() {
    int fd = open("/dev/null", O_RDWR);
    if (fd < 0) {
        fail("failed to open /dev/null: " + std::string(std::strerror(errno)));
    }
    dup2(fd, STDIN_FILENO);
    dup2(fd, STDOUT_FILENO);
    dup2(fd, STDERR_FILENO);
    close(fd);
}

int daemon_up(const Options &options, char **argv) {
    (void)argv;

    std::vector<DaemonProcess> existing = find_daemons(options);
    if (!existing.empty()) {
        std::cout << options.binary_name << " is already running with PID " << existing.front().pid << std::endl;
        return 1;
    }

    pid_t pid = fork();
    if (pid < 0) {
        fail("fork failed: " + std::string(std::strerror(errno)));
    }
    if (pid > 0) {
        std::cout << "Started " << options.binary_name << " with PID " << pid << std::endl;
        return 0;
    }

    if (setsid() < 0) {
        fail("setsid failed: " + std::string(std::strerror(errno)));
    }
    redirect_stdio_to_null();

    std::string gpu_list;
    for (std::size_t i = 0; i < options.gpus.size(); ++i) {
        if (i > 0) {
            gpu_list += ",";
        }
        gpu_list += std::to_string(options.gpus[i]);
    }

    std::vector<std::string> arg_storage{
        options.binary_path,
        kDaemonFlag,
        "--gpus",
        gpu_list,
    };

    std::vector<char *> exec_args;
    exec_args.reserve(arg_storage.size() + 1);
    for (std::string &arg : arg_storage) {
        exec_args.push_back(const_cast<char *>(arg.c_str()));
    }
    exec_args.push_back(nullptr);

    execv(options.binary_path.c_str(), exec_args.data());
    _exit(127);
}

int daemon_down(const Options &options) {
    std::vector<DaemonProcess> daemons = find_daemons(options);
    if (daemons.empty()) {
        std::cout << options.binary_name << " is not running." << std::endl;
        return 0;
    }

    for (const DaemonProcess &daemon : daemons) {
        pid_t pid = daemon.pid;
        if (kill(pid, SIGTERM) != 0 && errno != ESRCH) {
            fail("failed to stop PID " + std::to_string(pid) + ": " + std::strerror(errno));
        }
    }

    for (int i = 0; i < 100; ++i) {
        if (find_daemons(options).empty()) {
            std::cout << "Stopped " << options.binary_name << "." << std::endl;
            return 0;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    std::cout << "SIGTERM sent; some " << options.binary_name << " processes are still exiting." << std::endl;
    return 0;
}

int daemon_status(const Options &options) {
    std::vector<DaemonProcess> daemons = find_daemons(options);
    if (!daemons.empty()) {
        const DaemonProcess &daemon = daemons.front();
        std::cout << options.binary_name << " is running with PID " << daemon.pid << std::endl;
        if (daemon.gpus.empty()) {
            std::cout << "GPUs: unspecified" << std::endl;
            return 0;
        }

        std::cout << "GPUs: ";
        for (std::size_t i = 0; i < daemon.gpus.size(); ++i) {
            if (i > 0) {
                std::cout << ",";
            }
            std::cout << daemon.gpus[i];
        }
        std::cout << std::endl;

        std::vector<GpuMemoryUsage> usage = get_gpu_memory_usage(daemon.gpus);
        for (const GpuMemoryUsage &u : usage) {
            if (!u.valid || u.total_gb <= 0.0) {
                std::cout << "GPU " << u.physical_index << ": unavailable" << std::endl;
                continue;
            }
            double percent = u.used_gb / u.total_gb * 100.0;
            std::cout << "GPU " << u.physical_index << " memory used: "
                      << std::fixed << std::setprecision(2) << u.used_gb
                      << "GB / " << u.total_gb << "GB (" << percent << "%)" << std::endl;
        }
        return 0;
    }

    std::cout << options.binary_name << " is not running." << std::endl;
    return 1;
}

}  // namespace

int main(int argc, char **argv) {
    Options options = parse_args(argc, argv);

    if (options.internal_daemon) {
        return run_foreground(options);
    }

    switch (options.mode) {
        case Options::Mode::Foreground:
            return run_foreground(options);
        case Options::Mode::Up:
            return daemon_up(options, argv);
        case Options::Mode::Down:
            return daemon_down(options);
        case Options::Mode::Status:
            return daemon_status(options);
    }

    return 1;
}
