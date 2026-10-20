// Experimental host-side tiling adapter for unchanged CANN native binaries.
// No device kernel is generated. Restricts changes to two audited BF16 M8 shapes.
#include <acl/acl_rt.h>
#include <dlfcn.h>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <vector>

namespace {
std::mutex mutex_;
std::map<void*, uint64_t> function_keys;
std::atomic<int> mode{-1};
std::atomic<uint64_t> counts[2]{};
int get_mode() {
    int value = mode.load();
    if (value < 0) {
        const char* p = std::getenv("NATIVE_M8_RETILE");
        value = p ? std::atoi(p) : 0;
        mode.store(value);
    }
    return value;
}
bool patch(uint64_t key, uint32_t blocks, void* data, size_t bytes,
           aclrtPlaceHolderInfo* placeholders, size_t num) {
    const int plan = get_mode();
    if (!plan || !num || !placeholders || bytes != 200 || blocks != 23) return false;
    const size_t offset = placeholders[num - 1].dataOffset;
    if (offset != 40 || offset + 160 > bytes) return false;
    uint32_t t[40];
    std::memcpy(t, static_cast<char*>(data) + offset, sizeof(t));
    if (t[1] != 8 || t[32] != 1) return false;
    // v1 enlarged L1 alone. v2 also balances L0 to avoid extra MMAD/issue work.
    // Both L0B ping-pong allocations fit the audited 64 KiB capacity.
    const char* balance_env = std::getenv("NATIVE_M8_BALANCE_L0");
    const bool balance_l0 = !balance_env || std::atoi(balance_env) != 0;
    const char* preserve_env = std::getenv("NATIVE_M8_PRESERVE_L1");
    const bool preserve_l1 = preserve_env && std::atoi(preserve_env) != 0;
    int which = -1;
    if ((plan & 1) && key == 98513 && t[0] == 17408 && t[2] == 5120) {
        // Equal A/B K coverage and double buffers remain equal. L1=512 KiB.
        if (t[17] != 4 || t[18] != 7 || t[19] != 8 || t[20] != 8 ||
            t[21] != 34 || t[22] != 34 || t[23] != 32 || t[24] != 32 ||
            t[33] != 1 || t[34] != 1) std::abort();
        t[19] = t[20] = 16;
        t[21] = t[22] = 17;
        t[23] = t[24] = 64;
        if (balance_l0) {
            t[17] = 8;
            t[19] = t[20] = 8;
        }
        if (preserve_l1) {
            // Preserve original K512 transfers and K traversal order.
            t[17] = 8;
            t[19] = t[20] = 4;
            t[21] = t[22] = 34;
            t[23] = t[24] = 32;
        }
        which = 1;
    } else if ((plan & 2) && key == 98499 && t[0] == 5120 && t[2] == 34816) {
        // Keep 128 KiB B buffers: narrower N, longer contiguous K source rows.
        if (t[8] != 6 || t[17] != 4 || t[18] != 16 || t[20] != 4 ||
            t[22] != 20 || t[24] != 16 || t[25] != 20 ||
            t[33] != 0 || t[34] != 1) std::abort();
        t[8] = 12;
        t[18] = t[20] = 8;
        t[22] = t[25] = 10;
        t[24] = 32;
        if (balance_l0) {
            t[17] = 8;
            t[19] = 40;
            t[20] = 4;
        }
        if (preserve_l1) {
            // The original N256/K256 L1 tile feeds two N128/K128 L0 tiles.
            t[8] = 6;
            t[14] = 2;
            t[17] = 8;
            t[18] = 8;
            t[19] = 40;
            t[20] = 2;
            t[22] = t[25] = 20;
            t[24] = 16;
        }
        which = 0;
    }
    if (which < 0) return false;
    std::memcpy(static_cast<char*>(data) + offset, t, sizeof(t));
    if (counts[which].fetch_add(1) == 0)
        std::fprintf(stderr, "NATIVE_M8_RETILE_APPLIED key=%llu mode=%d balance_l0=%d preserve_l1=%d M=%u N=%u K=%u\n",
                     static_cast<unsigned long long>(key), plan, balance_l0, preserve_l1, t[1], t[2], t[0]);
    return true;
}
}

extern "C" void native_m8_set_mode(int value) { mode.store(value); }
extern "C" uint64_t native_m8_get_count(int which) {
    return which >= 0 && which < 2 ? counts[which].load() : 0;
}
extern "C" aclError aclrtBinaryGetFunctionByEntry(aclrtBinHandle h, uint64_t key,
                                                  aclrtFuncHandle* function) {
    using Fn = aclError(*)(aclrtBinHandle, uint64_t, aclrtFuncHandle*);
    static Fn real = reinterpret_cast<Fn>(dlsym(RTLD_NEXT, "aclrtBinaryGetFunctionByEntry"));
    auto result = real(h, key, function);
    if (!result && function) {
        std::lock_guard<std::mutex> guard(mutex_);
        function_keys[*function] = key;
    }
    return result;
}
extern "C" aclError aclrtBinaryGetFunction(aclrtBinHandle h, const char* name,
                                           aclrtFuncHandle* function) {
    using Fn = aclError(*)(aclrtBinHandle, const char*, aclrtFuncHandle*);
    static Fn real = reinterpret_cast<Fn>(dlsym(RTLD_NEXT, "aclrtBinaryGetFunction"));
    auto result = real(h, name, function);
    if (!result && function && name && std::strstr(name, "MatMul")) {
        const char* end = std::strrchr(name, '_');
        std::lock_guard<std::mutex> guard(mutex_);
        function_keys[*function] = end ? std::strtoull(end + 1, nullptr, 10) : 0;
    }
    return result;
}
extern "C" aclError aclrtLaunchKernelWithHostArgs(
    aclrtFuncHandle f, uint32_t blocks, aclrtStream stream, aclrtLaunchKernelCfg* config,
    void* args, size_t size, aclrtPlaceHolderInfo* placeholders, size_t num) {
    using Fn = aclError(*)(aclrtFuncHandle, uint32_t, aclrtStream, aclrtLaunchKernelCfg*,
                           void*, size_t, aclrtPlaceHolderInfo*, size_t);
    static Fn real = reinterpret_cast<Fn>(dlsym(RTLD_NEXT, "aclrtLaunchKernelWithHostArgs"));
    uint64_t key = 0;
    { std::lock_guard<std::mutex> guard(mutex_); key = function_keys[f]; }
    if (get_mode() && (key == 98499 || key == 98513)) {
        std::vector<unsigned char> copy(size);
        std::memcpy(copy.data(), args, size);
        if (patch(key, blocks, copy.data(), size, placeholders, num))
            return real(f, blocks, stream, config, copy.data(), size, placeholders, num);
    }
    return real(f, blocks, stream, config, args, size, placeholders, num);
}
