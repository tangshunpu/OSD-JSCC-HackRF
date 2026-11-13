// Continuous HackRF TX stream through libhackrf directly (hackrf_io.NativeTxStream), in place of
// `hackrf_transfer -t -` fed through a pipe (hackrf_io.TxStream).
//
// libhackrf keeps TRANSFER_COUNT = 4 USB transfers in flight, and hackrf_transfer always fills
// each one whole (TRANSFER_BUFFER_SIZE = 256 KiB), so a burst spliced into that stream waits
// behind ~1 MiB of zeros (~26 ms at 20 MS/s), plus the 64 KiB pipe. Here the TX callback sets
// valid_length to a shorter transfer (a multiple of 512 bytes), which libhackrf honours since
// 2022.09 (hackrf_libusb_transfer_callback resubmits with length = valid_length). The queue
// ahead of a burst is then 4 short transfers plus the 32 KiB device buffer, e.g. 4 x 32 KiB
// + 32 KiB = 4.1 ms. A burst always starts at a transfer boundary, at TX stream byte
// `start_byte`; between bursts the stream carries zeros, as before.
//
// The callbacks run on libhackrf's transfer thread and never touch Python. Python writes a
// burst into txn_burst_buffer(), calls txn_submit() and txn_wait(seq, 0) for its start.
// A poll thread reads the M0 state every 200 ms (shortfalls = TX underruns, as printed by
// `hackrf_transfer -B`).
//
// Built by hackrf_io.native_tx_lib() through ctypes against the runtime library:
//     c++ -O2 -shared -fPIC -std=c++17 -pthread hackrf_tx_native.cpp -o X.so -l:libhackrf.so.0
// The runtime package ships no header, so the declarations below are copied from hackrf.h of
// release 2023.01.1 (the AGX and the RTX host; Homebrew's 2024.02.1 has the same layouts).

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <thread>
#include <vector>

#ifdef __linux__
#include <pthread.h>
#include <sched.h>
#endif

extern "C" {
struct hackrf_device;
typedef struct {
    hackrf_device* device;
    uint8_t* buffer;
    int buffer_length;
    int valid_length;
    void* rx_ctx;
    void* tx_ctx;
} hackrf_transfer;
typedef struct {
    uint16_t requested_mode;
    uint16_t request_flag;
    uint32_t active_mode;
    uint32_t m0_count;
    uint32_t m4_count;
    uint32_t num_shortfalls;
    uint32_t longest_shortfall;
    uint32_t shortfall_limit;
    uint32_t threshold;
    uint32_t next_mode;
    uint32_t error;
} hackrf_m0_state;
typedef int (*hackrf_sample_block_cb_fn)(hackrf_transfer* transfer);
typedef void (*hackrf_tx_block_complete_cb_fn)(hackrf_transfer* transfer, int success);
int hackrf_init();
int hackrf_exit();
const char* hackrf_library_release();
int hackrf_open_by_serial(const char* const desired_serial_number, hackrf_device** device);
int hackrf_close(hackrf_device* device);
int hackrf_set_sample_rate(hackrf_device* device, const double freq_hz);
uint32_t hackrf_compute_baseband_filter_bw(const uint32_t bandwidth_hz);
int hackrf_set_baseband_filter_bandwidth(hackrf_device* device, const uint32_t bandwidth_hz);
int hackrf_set_hw_sync_mode(hackrf_device* device, const uint8_t value);
int hackrf_set_freq(hackrf_device* device, const uint64_t freq_hz);
int hackrf_set_amp_enable(hackrf_device* device, const uint8_t value);
int hackrf_set_antenna_enable(hackrf_device* device, const uint8_t value);
int hackrf_set_txvga_gain(hackrf_device* device, uint32_t value);
int hackrf_set_tx_block_complete_callback(hackrf_device* device, hackrf_tx_block_complete_cb_fn callback);
int hackrf_start_tx(hackrf_device* device, hackrf_sample_block_cb_fn callback, void* tx_ctx);
int hackrf_stop_tx(hackrf_device* device);
int hackrf_is_streaming(hackrf_device* device);
int hackrf_get_m0_state(hackrf_device* device, hackrf_m0_state* value);
const char* hackrf_error_name(int errcode);
}

namespace {

constexpr int HACKRF_SUCCESS = 0;
constexpr int HACKRF_TRUE = 1;
constexpr int MAX_TRANSFER = 262144;  // libhackrf TRANSFER_BUFFER_SIZE

int64_t now_ns() {  // same clock as Python's time.time_ns()
    timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return int64_t(ts.tv_sec) * 1000000000 + ts.tv_nsec;
}

struct Stream {
    hackrf_device* dev = nullptr;
    int transfer_bytes = 0;
    std::vector<uint8_t> burst;
    // Python -> transfer thread
    std::atomic<int64_t> pending_len{0}, pending_seq{0};
    // transfer thread only
    bool active = false;
    int64_t pos = 0, len = 0, seq = 0;
    // transfer thread -> Python
    std::atomic<int64_t> queued{0};     // bytes handed to libhackrf (TX stream byte counter)
    std::atomic<int64_t> completed{0};  // bytes of completed USB transfers (accepted by the device)
    std::atomic<int64_t> callbacks{0}, failed_transfers{0}, max_callback_ns{0};
    std::atomic<int64_t> started_seq{0}, done_seq{0}, accepted_seq{0};
    std::atomic<int64_t> start_byte{-1}, end_byte{-1}, t_start{0}, t_end{0}, t_accepted{0};
    // poll thread -> Python
    std::atomic<int64_t> shortfalls{0}, longest_shortfall{0}, polls{0}, poll_errors{0};
    std::atomic<bool> stop{false};
    std::thread poller;
};

Stream* S = nullptr;

int tx_callback(hackrf_transfer* transfer) {
    const int64_t t0 = now_ns();
    uint8_t* out = transfer->buffer;
    const int n = S->transfer_bytes;
    int filled = 0;
    if (!S->active) {
        const int64_t len = S->pending_len.load(std::memory_order_acquire);
        if (len > 0) {
            S->active = true;
            S->pos = 0;
            S->len = len;
            S->seq = S->pending_seq.load(std::memory_order_relaxed);
            S->pending_len.store(0, std::memory_order_relaxed);
            S->start_byte.store(S->queued.load(std::memory_order_relaxed), std::memory_order_relaxed);
            S->t_start.store(t0, std::memory_order_relaxed);
            S->started_seq.store(S->seq, std::memory_order_release);
        }
    }
    if (S->active) {
        const int64_t take = std::min<int64_t>(n, S->len - S->pos);
        std::memcpy(out, S->burst.data() + S->pos, size_t(take));
        S->pos += take;
        filled = int(take);
        if (S->pos == S->len) {
            S->active = false;
            S->end_byte.store(S->queued.load(std::memory_order_relaxed) + take, std::memory_order_relaxed);
            S->t_end.store(now_ns(), std::memory_order_relaxed);
            S->done_seq.store(S->seq, std::memory_order_release);
        }
    }
    if (filled < n) std::memset(out + filled, 0, size_t(n - filled));
    transfer->valid_length = n;
    S->queued.fetch_add(n, std::memory_order_relaxed);
    S->callbacks.fetch_add(1, std::memory_order_relaxed);
    const int64_t took = now_ns() - t0;
    if (took > S->max_callback_ns.load(std::memory_order_relaxed))
        S->max_callback_ns.store(took, std::memory_order_relaxed);
    return 0;
}

void tx_complete(hackrf_transfer* transfer, int success) {
    if (!success) {
        S->failed_transfers.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    const int64_t done = S->completed.fetch_add(transfer->valid_length, std::memory_order_relaxed)
                         + transfer->valid_length;
    const int64_t seq = S->done_seq.load(std::memory_order_acquire);
    if (seq > S->accepted_seq.load(std::memory_order_relaxed)
        && done >= S->end_byte.load(std::memory_order_relaxed)) {
        S->t_accepted.store(now_ns(), std::memory_order_relaxed);
        S->accepted_seq.store(seq, std::memory_order_release);
    }
}

void poll_loop() {
    while (!S->stop.load()) {
        hackrf_m0_state state;
        if (hackrf_get_m0_state(S->dev, &state) == HACKRF_SUCCESS) {
            S->shortfalls.store(state.num_shortfalls);
            S->longest_shortfall.store(state.longest_shortfall);
            S->polls.fetch_add(1);
        } else {
            S->poll_errors.fetch_add(1);
        }
        for (int i = 0; i < 20 && !S->stop.load(); ++i) std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

void set_error(char* err, int errlen, const char* what, int code) {
    if (err && errlen > 0) std::snprintf(err, size_t(errlen), "%s failed: %s (%d)", what, hackrf_error_name(code), code);
}

}  // namespace

extern "C" {

const char* txn_library_release() { return hackrf_library_release(); }

// Opens the radio by serial, configures it as `hackrf_transfer -t - -f freq -s rate -b bandwidth
// -a amp -p antenna -x txvga` does and starts streaming. `cores`: CPUs for libhackrf's transfer
// thread and the poll thread (Linux; ncores 0: no pinning). Returns 0 or a libhackrf error code
// (message in `err`).
int txn_open(const char* serial, uint64_t freq_hz, double rate, uint32_t bandwidth, int amp, int antenna,
             int txvga, int transfer_bytes, int64_t max_burst_bytes, const int* cores, int ncores,
             char* err, int errlen) {
    if (S) {
        std::snprintf(err, size_t(errlen), "a TX stream is already open in this process");
        return -1;
    }
    if (transfer_bytes < 512 || transfer_bytes > MAX_TRANSFER || transfer_bytes % 512) {
        std::snprintf(err, size_t(errlen), "transfer size must be a multiple of 512 bytes up to %d", MAX_TRANSFER);
        return -1;
    }
    S = new Stream();
    S->transfer_bytes = transfer_bytes;
    S->burst.assign(size_t(max_burst_bytes), 0);
#ifdef __linux__
    cpu_set_t saved;
    bool pinned = false;
    if (ncores > 0 && pthread_getaffinity_np(pthread_self(), sizeof(saved), &saved) == 0) {
        cpu_set_t want;
        CPU_ZERO(&want);
        for (int i = 0; i < ncores; ++i) CPU_SET(cores[i], &want);
        pinned = pthread_setaffinity_np(pthread_self(), sizeof(want), &want) == 0;  // inherited below
    }
#endif
    int code = hackrf_init();
    const char* step = "hackrf_init";
    if (code == HACKRF_SUCCESS) { step = "hackrf_open_by_serial"; code = hackrf_open_by_serial(serial, &S->dev); }
    if (code == HACKRF_SUCCESS) { step = "hackrf_set_sample_rate"; code = hackrf_set_sample_rate(S->dev, rate); }
    if (code == HACKRF_SUCCESS) {
        step = "hackrf_set_baseband_filter_bandwidth";
        code = hackrf_set_baseband_filter_bandwidth(S->dev, hackrf_compute_baseband_filter_bw(bandwidth));
    }
    if (code == HACKRF_SUCCESS) { step = "hackrf_set_hw_sync_mode"; code = hackrf_set_hw_sync_mode(S->dev, 0); }
    if (code == HACKRF_SUCCESS) { step = "hackrf_set_freq"; code = hackrf_set_freq(S->dev, freq_hz); }
    if (code == HACKRF_SUCCESS) { step = "hackrf_set_amp_enable"; code = hackrf_set_amp_enable(S->dev, uint8_t(amp)); }
    if (code == HACKRF_SUCCESS) {
        step = "hackrf_set_antenna_enable";
        code = hackrf_set_antenna_enable(S->dev, uint8_t(antenna));
    }
    if (code == HACKRF_SUCCESS) { step = "hackrf_set_txvga_gain"; code = hackrf_set_txvga_gain(S->dev, uint32_t(txvga)); }
    if (code == HACKRF_SUCCESS) {
        step = "hackrf_set_tx_block_complete_callback";
        code = hackrf_set_tx_block_complete_callback(S->dev, tx_complete);
    }
    if (code == HACKRF_SUCCESS) { step = "hackrf_start_tx"; code = hackrf_start_tx(S->dev, tx_callback, nullptr); }
    if (code == HACKRF_SUCCESS) S->poller = std::thread(poll_loop);
#ifdef __linux__
    if (pinned) pthread_setaffinity_np(pthread_self(), sizeof(saved), &saved);
#endif
    if (code != HACKRF_SUCCESS) {
        set_error(err, errlen, step, code);
        if (S->dev) hackrf_close(S->dev);
        hackrf_exit();
        delete S;
        S = nullptr;
    }
    return code;
}

uint8_t* txn_burst_buffer() { return S ? S->burst.data() : nullptr; }

// Queues the burst written into txn_burst_buffer(); it starts with the next transfer the
// callback fills. 1: a burst is still being sent; 2: longer than the buffer.
int txn_submit(int64_t len, int64_t seq) {
    if (!S) return -1;
    if (len > int64_t(S->burst.size())) return 2;
    if (S->pending_len.load() != 0 || S->started_seq.load() != S->done_seq.load()) return 1;
    S->pending_seq.store(seq, std::memory_order_relaxed);
    S->pending_len.store(len, std::memory_order_release);
    return 0;
}

// Waits until burst `seq` has started (what 0), has been handed to libhackrf whole (1), or its
// last transfer has completed on the USB (2). 0: done; 1: timeout; 2: the stream stopped.
int txn_wait(int64_t seq, int what, int timeout_ms) {
    if (!S) return -1;
    std::atomic<int64_t>& marker = what == 0 ? S->started_seq : what == 1 ? S->done_seq : S->accepted_seq;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    while (marker.load(std::memory_order_acquire) < seq) {
        if (hackrf_is_streaming(S->dev) != HACKRF_TRUE) return 2;
        if (std::chrono::steady_clock::now() > deadline) return 1;
        std::this_thread::sleep_for(std::chrono::microseconds(20));
    }
    return 0;
}

// [queued, completed, start_byte, end_byte, t_start, t_end, t_accepted, started_seq, done_seq,
//  accepted_seq, shortfalls, longest_shortfall, streaming, callbacks, max_callback_ns,
//  failed_transfers, polls, poll_errors, transfer_bytes]
void txn_stats(int64_t* out) {
    if (!S) return;
    out[0] = S->queued.load();
    out[1] = S->completed.load();
    out[7] = S->started_seq.load(std::memory_order_acquire);
    out[2] = S->start_byte.load();
    out[4] = S->t_start.load();
    out[8] = S->done_seq.load(std::memory_order_acquire);
    out[3] = S->end_byte.load();
    out[5] = S->t_end.load();
    out[9] = S->accepted_seq.load(std::memory_order_acquire);
    out[6] = S->t_accepted.load();
    out[10] = S->shortfalls.load();
    out[11] = S->longest_shortfall.load();
    out[12] = hackrf_is_streaming(S->dev) == HACKRF_TRUE;
    out[13] = S->callbacks.load();
    out[14] = S->max_callback_ns.load();
    out[15] = S->failed_transfers.load();
    out[16] = S->polls.load();
    out[17] = S->poll_errors.load();
    out[18] = S->transfer_bytes;
}

int txn_close() {
    if (!S) return 0;
    S->stop.store(true);
    if (S->poller.joinable()) S->poller.join();
    int code = hackrf_stop_tx(S->dev);
    hackrf_close(S->dev);
    hackrf_exit();
    delete S;
    S = nullptr;
    return code;
}

}  // extern "C"
