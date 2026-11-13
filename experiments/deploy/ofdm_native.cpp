// Native kernels of ofdm_phy (10 MHz PHY), for all packets of a burst in one call each:
// - ofdm_modulate (ofdm_phy.modulate): OFDM symbols (64-point IFFT + cyclic prefix), x2
//   interpolation with the taps of scipy.signal.resample_poly, one global gain and int8
//   quantization;
// - ofdm_rx_recover (ofdm_phy.recover): packet detection, timing, CFO, channel estimate,
//   equalization, pilot phase tracking and header Viterbi decoding of every candidate (the
//   CRC check and the per-packet report stay in Python).
//
// Built and loaded by ofdm_phy.native_kernel() through ctypes (plain C ABI, no Python headers):
//     c++ -O3 -ffp-contract=off -shared -fPIC -std=c++17 ofdm_native.cpp -o ofdm_native.so
// -ffp-contract=off keeps every multiply and add separate, as in numpy, so the samples agree
// with the numpy path to the last few bits.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace {

constexpr int NFFT = 64;

struct Twiddles {
    double re[NFFT / 2], im[NFFT / 2];
    Twiddles() {
        for (int k = 0; k < NFFT / 2; ++k) {  // exp(+2 pi i k / N): inverse transform
            const double angle = 2.0 * 3.14159265358979323846 * k / NFFT;
            re[k] = std::cos(angle);
            im[k] = std::sin(angle);
        }
    }
};

const Twiddles TW;

// In-place FFT (no normalization), iterative radix 2; sign +1 = inverse, -1 = forward.
void fft64(double* re, double* im, double sign) {
    for (int i = 1, j = 0; i < NFFT; ++i) {
        int bit = NFFT >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            std::swap(re[i], re[j]);
            std::swap(im[i], im[j]);
        }
    }
    for (int len = 2; len <= NFFT; len <<= 1) {
        const int half = len / 2, step = NFFT / len;
        for (int i = 0; i < NFFT; i += len) {
            for (int k = 0; k < half; ++k) {
                const double wr = TW.re[k * step], wi = sign * TW.im[k * step];
                const int a = i + k, b = a + half;
                const double vr = re[b] * wr - im[b] * wi, vi = re[b] * wi + im[b] * wr;
                re[b] = re[a] - vr;
                im[b] = im[a] - vi;
                re[a] = re[a] + vr;
                im[a] = im[a] + vi;
            }
        }
    }
}

void ifft64(double* re, double* im) { fft64(re, im, 1.0); }

}  // namespace

extern "C" {

// data:      [packets][symbols][48] complex (interleaved re, im): header + payload rows
// train:     [train_len] complex training samples that open every packet
// pilots:    [symbols][4] real pilot values
// data_idx, pilot_idx: FFT bins of the 48 data and 4 pilot carriers
// taps:      [ntaps][2] interpolation taps, y[2n + r] = sum_i taps[i][r] x[n + i - left]
// pad:       zero baseband samples kept on each side of a packet (filter context)
// out:       int8 I/Q pairs of the whole burst, zero-filled by the caller; the interpolated
//            segment of packet p starts at pair first + p * pitch
// stats:     gain, active RMS, peak (before gain), quantized peak magnitude
// Returns 0, 1 for non-finite samples or 2 when the quantized peak exceeds the limit.
int ofdm_modulate(const double* data, int packets, int symbols, const double* train, int train_len,
                  const double* pilots, const int64_t* data_idx, const int64_t* pilot_idx, int cp,
                  const double* taps, int ntaps, int left, int pad, double target_rms, double peak_limit,
                  int8_t* out, int64_t first, int64_t pitch, double* stats) {
    const int symlen = NFFT + cp;
    const int64_t block = train_len + int64_t(symbols) * symlen;
    const int64_t seg = block + 2 * pad;  // baseband samples per packet segment
    const int64_t up = 2 * seg;
    std::vector<double> xr(packets * seg, 0.0), xi(packets * seg, 0.0);
    double gr[NFFT], gi[NFFT];
    const double scale = 1.0 / std::sqrt(double(NFFT));  // norm="ortho"
    for (int p = 0; p < packets; ++p) {
        double* br = xr.data() + p * seg + pad;
        double* bi = xi.data() + p * seg + pad;
        for (int n = 0; n < train_len; ++n) {
            br[n] = train[2 * n];
            bi[n] = train[2 * n + 1];
        }
        for (int s = 0; s < symbols; ++s) {
            std::fill(gr, gr + NFFT, 0.0);
            std::fill(gi, gi + NFFT, 0.0);
            const double* row = data + (int64_t(p) * symbols + s) * 96;
            for (int k = 0; k < 48; ++k) {
                gr[data_idx[k]] = row[2 * k];
                gi[data_idx[k]] = row[2 * k + 1];
            }
            for (int k = 0; k < 4; ++k) gr[pilot_idx[k]] = pilots[s * 4 + k];
            ifft64(gr, gi);
            double* sr = br + train_len + int64_t(s) * symlen;
            double* si = bi + train_len + int64_t(s) * symlen;
            for (int n = 0; n < NFFT; ++n) {
                sr[cp + n] = gr[n] * scale;
                si[cp + n] = gi[n] * scale;
            }
            for (int n = 0; n < cp; ++n) {
                sr[n] = sr[NFFT + n];
                si[n] = si[NFFT + n];
            }
        }
    }

    // x2 interpolation in chunks of output samples that keep the four accumulators (I/Q x
    // output phase) in L1; the inner loop runs over output samples (vectorizable), and each
    // output sums its taps in index order. Zero taps are skipped (the caller zeroes the
    // ~1e-17 rounding residues at the sinc zeros of the half-band filter).
    constexpr int64_t CHUNK = 512;
    std::vector<double> yr(packets * up), yi(packets * up);
    double acc[4][CHUNK];
    for (int p = 0; p < packets; ++p) {
        const double* x[2] = {xr.data() + p * seg, xi.data() + p * seg};
        double* outr = yr.data() + p * up;
        double* outi = yi.data() + p * up;
        for (int64_t n0 = 0; n0 < seg; n0 += CHUNK) {
            const int64_t n1 = std::min(seg, n0 + CHUNK);
            for (auto& a : acc) std::fill(a, a + CHUNK, 0.0);
            for (int i = 0; i < ntaps; ++i) {
                const int64_t shift = i - left;  // output n reads x[n + shift]
                const int64_t lo = std::max<int64_t>(n0, -shift), hi = std::min<int64_t>(n1, seg - shift);
                for (int r = 0; r < 2; ++r) {
                    const double t = taps[2 * i + r];
                    if (t == 0.0) continue;
                    for (int c = 0; c < 2; ++c) {
                        double* a = acc[2 * c + r] - n0;
                        const double* src = x[c] + shift;
                        for (int64_t n = lo; n < hi; ++n) a[n] = a[n] + t * src[n];
                    }
                }
            }
            for (int64_t n = n0; n < n1; ++n) {
                outr[2 * n] = acc[0][n - n0];
                outr[2 * n + 1] = acc[1][n - n0];
                outi[2 * n] = acc[2][n - n0];
                outi[2 * n + 1] = acc[3][n - n0];
            }
        }
    }

    // one gain for the burst: active RMS to the target, capped by the peak
    double peak2 = 0.0, active = 0.0;
    bool finite = true;
    for (int p = 0; p < packets; ++p) {
        const double* r = yr.data() + p * up;
        const double* q = yi.data() + p * up;
        double sum = 0.0;
        for (int64_t n = 0; n < up; ++n) {
            const double power = r[n] * r[n] + q[n] * q[n];
            finite = finite && std::isfinite(power);
            peak2 = std::max(peak2, power);
            if (n >= 2 * pad && n < 2 * (pad + block)) sum += power;
        }
        active += sum;
    }
    if (!finite) return 1;
    const double rms = std::sqrt(active / double(packets * 2 * block));
    const double peak = std::sqrt(peak2);
    const double gain = std::min(target_rms / rms, (128.0 * peak_limit - 0.75) / peak);
    double limit2 = 0.0;
    for (int p = 0; p < packets; ++p) {
        const double* r = yr.data() + p * up;
        const double* q = yi.data() + p * up;
        int8_t* o = out + 2 * (first + p * pitch);
        for (int64_t n = 0; n < up; ++n) {
            const double i = std::nearbyint(r[n] * gain), v = std::nearbyint(q[n] * gain);  // as np.rint
            limit2 = std::max(limit2, i * i + v * v);
            o[2 * n] = int8_t(std::clamp(i, -128.0, 127.0));
            o[2 * n + 1] = int8_t(std::clamp(v, -128.0, 127.0));
        }
    }
    const double limit = std::sqrt(limit2);
    stats[0] = gain;
    stats[1] = rms;
    stats[2] = peak;
    stats[3] = limit;
    return (limit > 127.0 || limit / 128.0 > peak_limit + 1e-12) ? 2 : 0;
}

}  // extern "C"

// ============================================================================ receiver

namespace {

constexpr double PI = 3.14159265358979323846;
constexpr int N_ACTIVE = 52, N_DATA = 48, N_PILOT = 4;

struct RxTables {
    bool ready = false;
    std::vector<double> stf16;           // one 16-sample STF period, complex interleaved
    double stf_energy = 0;               // sum |STF|^2 over the whole STF
    std::vector<double> ltf;             // TRAIN[len(STF):], complex interleaved
    double ltf_active[2 * N_ACTIVE];     // LTF_GRID[ACTIVE_IDX]
    int active_k[N_ACTIVE], active_idx[N_ACTIVE], data_k[N_DATA], data_idx[N_DATA];
    int pilot_k[N_PILOT], pilot_idx[N_PILOT];
    std::vector<double> pilots;          // [symbols][4]
    std::vector<int> kind, param;        // smoother s: kind 0 = box of length param, 1 = projection of rank param
    std::vector<std::vector<double>> basis, pinv;  // projections: [52][r] and [r][52], complex interleaved
    std::vector<double> trace;
    std::vector<int> interleaver;        // air[i] = coded[interleaver[i]]
    double conv_sign[64][2][2];
    int symbols = 0, header_rows = 0, coded_bits = 0;
    double base_rate = 0, stf_threshold = 0, score_threshold = 0;
    int stf_len = 0, ltf_len = 0, cp = 0, advance = 0, train_len = 0, packet_len = 0, guard_far = 0,
        guard_near = 0, delay = 0, window = 0, distance = 0, before = 0, after = 0, dedupe = 0,
        end_margin = 0, ltf_search = 0, extra = 0;
};

RxTables RT;

inline double sq(double v) { return v * v; }

// dst[m] = (src[m] - dc) * exp(i * w * (m0 + m)), re-anchored with cos/sin every 64 samples.
void rotate(const double* src, int64_t len, double w, int64_t m0, double* dst, double dcr = 0, double dci = 0) {
    const double sr = std::cos(w), si = std::sin(w);
    for (int64_t a = 0; a < len; a += 64) {
        double cr = std::cos(w * double(m0 + a)), ci = std::sin(w * double(m0 + a));
        const int64_t b = std::min<int64_t>(len, a + 64);
        for (int64_t m = a; m < b; ++m) {
            const double xr = src[2 * m] - dcr, xi = src[2 * m + 1] - dci;
            dst[2 * m] = xr * cr - xi * ci;
            dst[2 * m + 1] = xr * ci + xi * cr;
            const double nr = cr * sr - ci * si;
            ci = cr * si + ci * sr;
            cr = nr;
        }
    }
}

// 64 samples starting at `at` of the packet, CFO-corrected, -> spectrum (norm="ortho").
void window_fft(const double* packet, int64_t at, double w, double* re, double* im, double dcr, double dci) {
    double buf[128];
    rotate(packet + 2 * at, 64, w, at, buf, dcr, dci);
    for (int n = 0; n < 64; ++n) {
        re[n] = buf[2 * n];
        im[n] = buf[2 * n + 1];
    }
    fft64(re, im, -1.0);
    for (int n = 0; n < 64; ++n) {
        re[n] *= 0.125;
        im[n] *= 0.125;
    }
}

// table[k + 32] = exp(i (c0 + c1 k)) for |k| <= 32, from two cos/sin pairs and recurrences.
void carrier_phases(double c0, double c1, double* table) {
    const double sr = std::cos(c1), si = std::sin(c1);
    double pr = std::cos(c0), pi = std::sin(c0), nr = pr, ni = pi;
    table[64] = pr;
    table[65] = pi;
    for (int k = 1; k <= 32; ++k) {
        const double ar = pr * sr - pi * si, ai = pr * si + pi * sr;  // * exp(+i c1)
        const double br = nr * sr + ni * si, bi = ni * sr - nr * si;  // * exp(-i c1)
        pr = ar; pi = ai; nr = br; ni = bi;
        table[2 * (32 + k)] = pr;
        table[2 * (32 + k) + 1] = pi;
        table[2 * (32 - k)] = nr;
        table[2 * (32 - k) + 1] = ni;
    }
}

// scipy.signal.find_peaks(x, height=hmin, distance=distance)
std::vector<int64_t> find_peaks(const std::vector<double>& x, double hmin, int64_t distance) {
    std::vector<int64_t> peaks;
    const int64_t n = int64_t(x.size());
    int64_t i = 1;
    while (i < n - 1) {
        if (x[i - 1] < x[i]) {
            int64_t ahead = i + 1;
            while (ahead < n - 1 && x[ahead] == x[i]) ++ahead;
            if (x[ahead] < x[i]) {
                peaks.push_back((i + ahead - 1) / 2);
                i = ahead;
            }
        }
        ++i;
    }
    std::vector<int64_t> high;
    for (int64_t p : peaks)
        if (x[p] >= hmin) high.push_back(p);
    const int64_t count = int64_t(high.size());
    std::vector<int64_t> order(count);
    for (int64_t k = 0; k < count; ++k) order[k] = k;
    std::stable_sort(order.begin(), order.end(), [&](int64_t a, int64_t b) { return x[high[a]] < x[high[b]]; });
    std::vector<char> keep(count, 1);
    for (int64_t r = count - 1; r >= 0; --r) {
        const int64_t j = order[r];
        if (!keep[j]) continue;
        for (int64_t k = j - 1; k >= 0 && high[j] - high[k] < distance; --k) keep[k] = 0;
        for (int64_t k = j + 1; k < count && high[k] - high[j] < distance; ++k) keep[k] = 0;
    }
    std::vector<int64_t> out;
    for (int64_t k = 0; k < count; ++k)
        if (keep[k]) out.push_back(high[k]);
    return out;
}

// Linear phase across the active carriers: ramp[k] = exp(i * slope * active_k[k]).
void channel_ramp(const double* h, double* ramp) {
    double sr = 0, si = 0;
    for (int k = 0; k + 1 < N_ACTIVE; ++k) {
        if (RT.active_k[k + 1] - RT.active_k[k] != 1) continue;
        const double ar = h[2 * k], ai = h[2 * k + 1], br = h[2 * k + 2], bi = h[2 * k + 3];
        sr += ar * br + ai * bi;  // conj(a) * b
        si += ar * bi - ai * br;
    }
    const double slope = std::atan2(si, sr);
    for (int k = 0; k < N_ACTIVE; ++k) {
        ramp[2 * k] = std::cos(slope * RT.active_k[k]);
        ramp[2 * k + 1] = std::sin(slope * RT.active_k[k]);
    }
}

// Moving average of g over carriers [lo, hi) for every carrier k, from prefix sums.
void box_average(const double* prefix, int lo_off, int hi_off, double* out) {
    for (int k = 0; k < N_ACTIVE; ++k) {
        const int lo = std::max(0, k + lo_off), hi = std::min(N_ACTIVE, k + hi_off);
        const double inv = 1.0 / double(hi - lo);
        out[2 * k] = (prefix[2 * hi] - prefix[2 * lo]) * inv;
        out[2 * k + 1] = (prefix[2 * hi + 1] - prefix[2 * lo + 1]) * inv;
    }
}

// Channel estimate from the LS values h (method 0 = SURE, 1 = moving average of `taps`
// carriers, 2 = LS). Returns the smoother id for SURE, -1 otherwise.
int estimate_channel(const double* h, double noise_var, int method, int taps, double* out) {
    if (method == 2 || (method == 1 && taps <= 1)) {
        std::copy(h, h + 2 * N_ACTIVE, out);
        return -1;
    }
    double ramp[2 * N_ACTIVE], g[2 * N_ACTIVE], prefix[2 * (N_ACTIVE + 1)];
    channel_ramp(h, ramp);
    for (int k = 0; k < N_ACTIVE; ++k) {  // g = h / ramp, |ramp| = 1
        g[2 * k] = h[2 * k] * ramp[2 * k] + h[2 * k + 1] * ramp[2 * k + 1];
        g[2 * k + 1] = h[2 * k + 1] * ramp[2 * k] - h[2 * k] * ramp[2 * k + 1];
    }
    prefix[0] = prefix[1] = 0;
    for (int k = 0; k < N_ACTIVE; ++k) {
        prefix[2 * k + 2] = prefix[2 * k] + g[2 * k];
        prefix[2 * k + 3] = prefix[2 * k + 1] + g[2 * k + 1];
    }
    double best[2 * N_ACTIVE], est[2 * N_ACTIVE];
    int chosen = -1;
    if (method == 1) {  // np.convolve(..., "same") alignment
        box_average(prefix, (taps - 1) / 2 - taps + 1, (taps - 1) / 2 + 1, best);
    } else {
        double best_risk = INFINITY;
        for (size_t s = 0; s < RT.kind.size(); ++s) {
            if (RT.kind[s] == 0) {
                const int length = RT.param[s];
                box_average(prefix, -(length / 2), length / 2 + 1, est);
            } else {
                const int rank = RT.param[s];
                const double* B = RT.basis[s].data();
                const double* P = RT.pinv[s].data();
                double c[64];
                for (int j = 0; j < rank; ++j) {
                    double cr = 0, ci = 0;
                    for (int k = 0; k < N_ACTIVE; ++k) {
                        const double pr = P[2 * (j * N_ACTIVE + k)], pi = P[2 * (j * N_ACTIVE + k) + 1];
                        cr += pr * g[2 * k] - pi * g[2 * k + 1];
                        ci += pr * g[2 * k + 1] + pi * g[2 * k];
                    }
                    c[2 * j] = cr;
                    c[2 * j + 1] = ci;
                }
                for (int k = 0; k < N_ACTIVE; ++k) {
                    double er = 0, ei = 0;
                    for (int j = 0; j < rank; ++j) {
                        const double br = B[2 * (k * rank + j)], bi = B[2 * (k * rank + j) + 1];
                        er += br * c[2 * j] - bi * c[2 * j + 1];
                        ei += br * c[2 * j + 1] + bi * c[2 * j];
                    }
                    est[2 * k] = er;
                    est[2 * k + 1] = ei;
                }
            }
            double risk = 0;
            for (int k = 0; k < N_ACTIVE; ++k) risk += sq(est[2 * k] - g[2 * k]) + sq(est[2 * k + 1] - g[2 * k + 1]);
            risk += 2 * noise_var * RT.trace[s];
            if (risk < best_risk) {
                best_risk = risk;
                chosen = int(s);
                std::copy(est, est + 2 * N_ACTIVE, best);
            }
        }
    }
    for (int k = 0; k < N_ACTIVE; ++k) {  // restore the ramp
        out[2 * k] = best[2 * k] * ramp[2 * k] - best[2 * k + 1] * ramp[2 * k + 1];
        out[2 * k + 1] = best[2 * k] * ramp[2 * k + 1] + best[2 * k + 1] * ramp[2 * k];
    }
    return chosen;
}

// Phase model per OFDM symbol (offset, slope) from the pilot residuals [symbols][4]
// (tracking 0 = hybrid, 1 = packet, 2 = symbol), as ofdm_phy._phase_track.
void phase_track(const double* res, int symbols, int tracking, double* coef) {
    const double spacing = double(RT.pilot_k[1] - RT.pilot_k[0]);
    if (tracking == 2) {
        double mean_k = 0;
        for (int j = 0; j < N_PILOT; ++j) mean_k += RT.pilot_k[j];
        mean_k /= N_PILOT;
        double sxx = 0;
        for (int j = 0; j < N_PILOT; ++j) sxx += sq(RT.pilot_k[j] - mean_k);
        for (int t = 0; t < symbols; ++t) {
            double raw[N_PILOT], ph[N_PILOT];
            for (int j = 0; j < N_PILOT; ++j) raw[j] = std::atan2(res[2 * (t * N_PILOT + j) + 1], res[2 * (t * N_PILOT + j)]);
            double correction = 0;  // np.unwrap
            ph[0] = raw[0];
            for (int j = 1; j < N_PILOT; ++j) {
                const double dd = raw[j] - raw[j - 1];
                double ddmod = std::fmod(dd + PI, 2 * PI);
                if (ddmod < 0) ddmod += 2 * PI;
                ddmod -= PI;
                if (ddmod == -PI && dd > 0) ddmod = PI;
                if (std::fabs(dd) >= PI) correction += ddmod - dd;
                ph[j] = raw[j] + correction;
            }
            double mean_p = 0;
            for (int j = 0; j < N_PILOT; ++j) mean_p += ph[j];
            mean_p /= N_PILOT;
            double sxy = 0;
            for (int j = 0; j < N_PILOT; ++j) sxy += (RT.pilot_k[j] - mean_k) * (ph[j] - mean_p);
            coef[2 * t + 1] = sxy / sxx;
            coef[2 * t] = mean_p - coef[2 * t + 1] * mean_k;
        }
        return;
    }
    double sr = 0, si = 0;
    for (int t = 0; t < symbols; ++t)
        for (int j = 0; j + 1 < N_PILOT; ++j) {
            const double* a = res + 2 * (t * N_PILOT + j);
            sr += a[0] * a[2] + a[1] * a[3];
            si += a[0] * a[3] - a[1] * a[2];
        }
    const double slope = std::atan2(si, sr) / spacing;
    std::vector<double> der(2 * symbols * N_PILOT);
    for (int t = 0; t < symbols; ++t)
        for (int j = 0; j < N_PILOT; ++j) {
            const double c = std::cos(slope * RT.pilot_k[j]), s = std::sin(slope * RT.pilot_k[j]);
            const double* a = res + 2 * (t * N_PILOT + j);
            der[2 * (t * N_PILOT + j)] = a[0] * c + a[1] * s;  // * exp(-i slope k)
            der[2 * (t * N_PILOT + j) + 1] = a[1] * c - a[0] * s;
        }
    if (tracking == 0) {
        for (int t = 0; t < symbols; ++t) {
            double cr = 0, ci = 0;
            for (int j = 0; j < N_PILOT; ++j) {
                cr += der[2 * (t * N_PILOT + j)];
                ci += der[2 * (t * N_PILOT + j) + 1];
            }
            coef[2 * t] = std::atan2(ci, cr);
            coef[2 * t + 1] = slope;
        }
        return;
    }
    double dr = 0, di = 0;  // packet: drift per symbol, then one offset
    for (int t = 0; t + 1 < symbols; ++t)
        for (int j = 0; j < N_PILOT; ++j) {
            const double* a = der.data() + 2 * (t * N_PILOT + j);
            const double* b = der.data() + 2 * ((t + 1) * N_PILOT + j);
            dr += a[0] * b[0] + a[1] * b[1];
            di += a[0] * b[1] - a[1] * b[0];
        }
    const double drift = std::atan2(di, dr);
    double orr = 0, oi = 0;
    for (int t = 0; t < symbols; ++t) {
        const double c = std::cos(drift * t), s = std::sin(drift * t);
        for (int j = 0; j < N_PILOT; ++j) {
            const double* a = der.data() + 2 * (t * N_PILOT + j);
            orr += a[0] * c + a[1] * s;
            oi += a[1] * c - a[0] * s;
        }
    }
    const double offset = std::atan2(oi, orr);
    for (int t = 0; t < symbols; ++t) {
        coef[2 * t] = offset + drift * t;
        coef[2 * t + 1] = slope;
    }
}

// Soft-decision Viterbi of the K = 7 header code, as ofdm_phy.viterbi.
void viterbi(const double* soft, int steps, uint8_t* bits) {
    double metric[64], next[64];
    std::vector<uint8_t> decision(size_t(steps) * 64);
    std::fill(metric, metric + 64, -INFINITY);
    metric[0] = 0.0;
    for (int t = 0; t < steps; ++t) {
        const double s0 = soft[2 * t], s1 = soft[2 * t + 1];
        for (int ns = 0; ns < 64; ++ns) {
            const int input = ns >> 5, j = ns & 31, pe = 2 * j, po = 2 * j + 1;
            const double even = metric[pe] + (s0 * RT.conv_sign[pe][input][0] + s1 * RT.conv_sign[pe][input][1]);
            const double odd = metric[po] + (s0 * RT.conv_sign[po][input][0] + s1 * RT.conv_sign[po][input][1]);
            decision[size_t(t) * 64 + ns] = odd > even;
            next[ns] = odd > even ? odd : even;
        }
        std::copy(next, next + 64, metric);
    }
    int state = 0;
    for (int t = steps - 1; t >= 0; --t) {
        bits[t] = uint8_t(state >> 5);
        state = ((state & 31) << 1) | decision[size_t(t) * 64 + state];
    }
}

}  // namespace

extern "C" {

// Tables and constants of ofdm_phy (see ofdm_phy._native_rx_setup). Returns 0 or a negative
// code for an unsupported layout.
int ofdm_rx_setup(const double* stf, int stf_len, const double* ltf, int ltf_len, const double* ltf_active,
                  const int64_t* active_k, const int64_t* data_k, const int64_t* pilot_k, const double* pilots,
                  int symbols, int header_rows, const int64_t* kinds, const int64_t* params, int n_smoothers,
                  const double* bases, const double* pinvs, const double* trace, const int64_t* interleaver,
                  int interleaver_len, int coded_bits, const double* conv_sign, const double* fparams,
                  const int64_t* iparams) {
    RxTables& t = RT;
    t = RxTables();
    if (stf_len % 16) return -1;
    for (int n = 0; n < stf_len; ++n)  // the STF must repeat a 16-sample period
        if (stf[2 * n] != stf[2 * (n % 16)] || stf[2 * n + 1] != stf[2 * (n % 16) + 1]) return -2;
    t.stf16.assign(stf, stf + 32);
    for (int n = 0; n < stf_len; ++n) t.stf_energy += sq(std::hypot(stf[2 * n], stf[2 * n + 1]));
    t.ltf.assign(ltf, ltf + 2 * ltf_len);
    std::copy(ltf_active, ltf_active + 2 * N_ACTIVE, t.ltf_active);
    for (int k = 0; k < N_ACTIVE; ++k) t.active_idx[k] = int(((t.active_k[k] = int(active_k[k])) % 64 + 64) % 64);
    for (int k = 0; k < N_DATA; ++k) t.data_idx[k] = int(((t.data_k[k] = int(data_k[k])) % 64 + 64) % 64);
    for (int k = 0; k < N_PILOT; ++k) t.pilot_idx[k] = int(((t.pilot_k[k] = int(pilot_k[k])) % 64 + 64) % 64);
    t.symbols = symbols;
    t.header_rows = header_rows;
    t.pilots.assign(pilots, pilots + symbols * N_PILOT);
    const double* b = bases;
    const double* p = pinvs;
    for (int s = 0; s < n_smoothers; ++s) {
        t.kind.push_back(int(kinds[s]));
        t.param.push_back(int(params[s]));
        t.trace.push_back(trace[s]);
        if (kinds[s] == 1) {
            if (params[s] > 32) return -3;
            const size_t size = size_t(2 * N_ACTIVE * params[s]);
            t.basis.emplace_back(b, b + size);
            t.pinv.emplace_back(p, p + size);
            b += size;
            p += size;
        } else {
            t.basis.emplace_back();
            t.pinv.emplace_back();
        }
    }
    t.interleaver.assign(interleaver, interleaver + interleaver_len);
    t.coded_bits = coded_bits;
    for (int s = 0; s < 64; ++s)
        for (int i = 0; i < 2; ++i)
            for (int k = 0; k < 2; ++k) t.conv_sign[s][i][k] = conv_sign[(s * 2 + i) * 2 + k];
    t.base_rate = fparams[0];
    t.stf_threshold = fparams[1];
    t.score_threshold = fparams[2];
    int* ints[] = {&t.cp, &t.advance, &t.train_len, &t.packet_len, &t.guard_far, &t.guard_near, &t.delay,
                   &t.window, &t.distance, &t.before, &t.after, &t.dedupe, &t.end_margin, &t.ltf_search, &t.extra};
    for (size_t i = 0; i < sizeof(ints) / sizeof(ints[0]); ++i) *ints[i] = int(iparams[i]);
    t.stf_len = stf_len;
    t.ltf_len = ltf_len;
    if (ltf_len + 2 * t.ltf_search > 256 || 2 * t.ltf_search + 1 > 64) return -4;  // LTF timing buffers
    t.ready = true;
    return 0;
}

// Blind recovery of the candidate packets in baseband z (complex, n samples), as the numpy
// ofdm_phy.recover up to (excluding) the header CRC check. Per candidate, in order of start:
// start, STF score, coarse and fine CFO, LTF noise, channel [52], smoother id, usable flag,
// equalized rows [symbols][48], decoded header bits [coded_bits / 2] and stats (guard noise
// or NaN, LTF signal power, pilot residual EVM %, min |H|, max |H|, data-carrier power spread dB).
// Returns the number of candidates (only the first `capacity` are written) or -1 before setup.
int64_t ofdm_rx_recover(const double* z_in, int64_t n, int method, int taps, int tracking, int64_t capacity,
                        double* dc_out, int64_t* rejected_end, int64_t* start_out, double* quality_out,
                        double* coarse_out, double* fine_out, double* noise_out, double* channel_out,
                        int32_t* smoother_out, uint8_t* usable_out, double* data_out, uint8_t* bits_out,
                        double* stats_out) {
    const RxTables& t = RT;
    if (!t.ready) return -1;
    // DC removal and the detection signal (z minus its 16-sample moving average)
    double dr = 0, di = 0;
    for (int64_t m = 0; m < n; ++m) {
        dr += z_in[2 * m];
        di += z_in[2 * m + 1];
    }
    dr /= double(n > 0 ? n : 1);
    di /= double(n > 0 ? n : 1);
    dc_out[0] = dr;
    dc_out[1] = di;
    // z (= z_in - dc) is not stored: readers subtract the DC on the fly
    std::vector<double> zd(2 * n);
    for (int64_t m = 0; m < n; ++m) {
        zd[2 * m] = z_in[2 * m] - dr;
        zd[2 * m + 1] = z_in[2 * m + 1] - di;
    }
    if (n >= 16) {
        double sr = 0, si = 0;
        for (int m = 0; m < 16; ++m) {
            sr += z_in[2 * m] - dr;
            si += z_in[2 * m + 1] - di;
        }
        for (int64_t k = 0; k + 16 <= n; ++k) {
            if (k) {
                sr += (z_in[2 * (k + 15)] - dr) - (z_in[2 * (k - 1)] - dr);
                si += (z_in[2 * (k + 15) + 1] - di) - (z_in[2 * (k - 1) + 1] - di);
            }
            zd[2 * (k + 8)] -= sr / 16;
            zd[2 * (k + 8) + 1] -= si / 16;
        }
    }
    // lag-16 autocorrelation metric over `window` products (ofdm_phy.STF_WINDOW)
    const int delay = t.delay, window = t.window;
    const int64_t lagged = n - delay, count = lagged - window + 1;
    std::vector<double> metric(count > 0 ? count : 0);
    if (count > 0) {
        double cr = 0, ci = 0, e1 = 0, e2 = 0;
        auto term = [&](int64_t m, double& r, double& i, double& a, double& b) {
            const double ar = zd[2 * m], ai = zd[2 * m + 1], br = zd[2 * (m + delay)], bi = zd[2 * (m + delay) + 1];
            r = ar * br + ai * bi;
            i = ar * bi - ai * br;
            a = ar * ar + ai * ai;
            b = br * br + bi * bi;
        };
        double r, i, a, b;
        for (int64_t m = 0; m < window; ++m) {
            term(m, r, i, a, b);
            cr += r; ci += i; e1 += a; e2 += b;
        }
        for (int64_t k = 0; k < count; ++k) {
            if (k) {
                term(k + window - 1, r, i, a, b);
                cr += r; ci += i; e1 += a; e2 += b;
                term(k - 1, r, i, a, b);
                cr -= r; ci -= i; e1 -= a; e2 -= b;
            }
            metric[k] = (cr * cr + ci * ci) / std::max(e1 * e2, 1e-20);
        }
    }
    const std::vector<int64_t> peaks = find_peaks(metric, t.stf_threshold, t.distance);

    // STF check of every peak: correlation with the STF over the window -before..+after
    struct Candidate { double score; int64_t start; double coarse; };
    std::vector<Candidate> candidates;
    std::vector<double> local, c16, energy, lr, li;
    for (int64_t peak : peaks) {
        double xr = 0, xi = 0;  // the lag-16 correlation at the peak
        for (int64_t m = peak; m < peak + window; ++m) {
            const double ar = zd[2 * m], ai = zd[2 * m + 1], br = zd[2 * (m + delay)], bi = zd[2 * (m + delay) + 1];
            xr += ar * br + ai * bi;
            xi += ar * bi - ai * br;
        }
        const double coarse = std::atan2(xi, xr) * t.base_rate / (2 * PI * delay);
        const int64_t lo = std::max<int64_t>(0, peak - t.before), hi = std::min<int64_t>(n, peak + t.after);
        const int64_t width = hi - lo;
        if (width < t.stf_len) continue;
        local.resize(2 * width);
        rotate(zd.data() + 2 * lo, width, -2 * PI * coarse / t.base_rate, 0, local.data());
        // correlation with one STF period, taps outer so the loop over lags vectorizes
        const int64_t lags = width - 15;
        lr.resize(width);
        li.resize(width);
        for (int64_t m = 0; m < width; ++m) {
            lr[m] = local[2 * m];
            li[m] = local[2 * m + 1];
        }
        c16.assign(2 * lags, 0.0);
        double* cr = c16.data();
        double* ci = c16.data() + lags;
        for (int j = 0; j < 16; ++j) {
            const double sr0 = t.stf16[2 * j], si0 = t.stf16[2 * j + 1];
            const double* xr = lr.data() + j;
            const double* xi = li.data() + j;
            for (int64_t m = 0; m < lags; ++m) {  // x * conj(stf)
                cr[m] = cr[m] + (xr[m] * sr0 + xi[m] * si0);
                ci[m] = ci[m] + (xi[m] * sr0 - xr[m] * si0);
            }
        }
        energy.assign(width + 1, 0.0);
        for (int64_t m = 0; m < width; ++m) energy[m + 1] = energy[m] + sq(local[2 * m]) + sq(local[2 * m + 1]);
        double best = -1;
        int64_t index = 0;
        const int periods = t.stf_len / 16;
        for (int64_t k = 0; k + t.stf_len <= width; ++k) {
            double sr = 0, si = 0;
            for (int r = 0; r < periods; ++r) {
                sr += cr[k + 16 * r];
                si += ci[k + 16 * r];
            }
            const double e = energy[k + t.stf_len] - energy[k];
            const double score = (sr * sr + si * si) / std::max(t.stf_energy * e, 1e-20);
            if (score > best) {
                best = score;
                index = k;
            }
        }
        if (best > t.score_threshold) candidates.push_back({best, lo + index, coarse});
    }
    std::sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) {
        if (a.score != b.score) return a.score > b.score;
        if (a.start != b.start) return a.start > b.start;
        return a.coarse > b.coarse;
    });
    std::vector<Candidate> chosen;
    for (const Candidate& c : candidates) {
        bool apart = true;
        for (const Candidate& o : chosen) apart = apart && std::llabs(c.start - o.start) > t.dedupe;
        if (apart) chosen.push_back(c);
    }
    std::stable_sort(chosen.begin(), chosen.end(), [](const Candidate& a, const Candidate& b) { return a.start < b.start; });
    *rejected_end = 0;

    // every candidate: LTF timing, fine CFO, channel, equalization, phase tracking, header
    const int symbols = t.symbols, symlen = 64 + t.cp;
    const int header_bits = t.coded_bits / 2;
    int64_t out = 0;
    std::vector<double> ltf_local(2 * (t.ltf_len + 2 * t.ltf_search));
    std::vector<double> full(2 * 64 * symbols), residual(2 * N_PILOT * symbols), coef(2 * symbols);
    std::vector<double> soft(t.interleaver.size()), coded(t.interleaver.size());
    for (const Candidate& c : chosen) {
        if (c.start + t.packet_len + t.end_margin >= n) {
            ++*rejected_end;
            continue;
        }
        const int64_t lo = c.start + t.stf_len - t.ltf_search;
        const int64_t width = t.ltf_len + 2 * t.ltf_search;
        rotate(z_in + 2 * lo, width, -2 * PI * c.coarse / t.base_rate, 0, ltf_local.data(), dr, di);
        double best = -1;
        int64_t index = 0;
        const int64_t lags = width - t.ltf_len + 1;
        double ar[64] = {0}, ai[64] = {0}, xr[256], xi[256];  // lags <= 64, width <= 256
        for (int64_t m = 0; m < width; ++m) {
            xr[m] = ltf_local[2 * m];
            xi[m] = ltf_local[2 * m + 1];
        }
        for (int j = 0; j < t.ltf_len; ++j) {  // taps outer: the loop over lags vectorizes
            const double lr = t.ltf[2 * j], li = t.ltf[2 * j + 1];
            for (int64_t k = 0; k < lags; ++k) {
                ar[k] = ar[k] + (xr[k + j] * lr + xi[k + j] * li);
                ai[k] = ai[k] + (xi[k + j] * lr - xr[k + j] * li);
            }
        }
        for (int64_t k = 0; k < lags; ++k) {
            const double magnitude = std::hypot(ar[k], ai[k]);
            if (magnitude > best) {
                best = magnitude;
                index = k;
            }
        }
        const int64_t start = lo + index - t.stf_len;
        if (start < 0) continue;
        const bool write = out < capacity;
        const double* packet = z_in + 2 * start;
        const int first = t.stf_len + 32 - t.advance;
        double l1[128], l2[128];
        rotate(packet + 2 * first, 64, -2 * PI * c.coarse / t.base_rate, first, l1, dr, di);
        rotate(packet + 2 * (first + 64), 64, -2 * PI * c.coarse / t.base_rate, first + 64, l2, dr, di);
        double fr = 0, fi = 0;
        for (int m = 0; m < 64; ++m) {
            fr += l1[2 * m] * l2[2 * m] + l1[2 * m + 1] * l2[2 * m + 1];
            fi += l1[2 * m] * l2[2 * m + 1] - l1[2 * m + 1] * l2[2 * m];
        }
        const double fine = std::atan2(fi, fr) * t.base_rate / (2 * PI * 64);
        const double cfo = c.coarse + fine, w = -2 * PI * cfo / t.base_rate;
        double t0r[64], t0i[64], t1r[64], t1i[64];
        window_fft(packet, first, w, t0r, t0i, dr, di);
        window_fft(packet, first + 64, w, t1r, t1i, dr, di);
        double ltf_noise = 0, ls[2 * N_ACTIVE];
        for (int k = 0; k < N_ACTIVE; ++k) {
            const int a = t.active_idx[k];
            ltf_noise += sq(t0r[a] - t1r[a]) + sq(t0i[a] - t1i[a]);
            const double mr = (t0r[a] + t1r[a]) / 2, mi = (t0i[a] + t1i[a]) / 2;
            const double gr = t.ltf_active[2 * k], gi = t.ltf_active[2 * k + 1], d = gr * gr + gi * gi;
            ls[2 * k] = (mr * gr + mi * gi) / d;
            ls[2 * k + 1] = (mi * gr - mr * gi) / d;
        }
        ltf_noise = ltf_noise / N_ACTIVE / 2;
        double h[2 * N_ACTIVE];
        const int smoother = estimate_channel(ls, ltf_noise / 2, method, taps, h);
        double hmin = INFINITY, hmax = 0, power = 0;
        for (int k = 0; k < N_ACTIVE; ++k) {
            const double magnitude = std::hypot(h[2 * k], h[2 * k + 1]);
            hmin = std::min(hmin, magnitude);
            hmax = std::max(hmax, magnitude);
            power += sq(magnitude);
        }
        const bool usable = hmin >= 1e-8;
        double* data = write ? data_out + size_t(out) * symbols * N_DATA * 2 : nullptr;
        uint8_t* bits = write ? bits_out + size_t(out) * header_bits : nullptr;
        double evm = NAN, spread = NAN, guard = NAN;
        if (write) std::fill(data, data + size_t(symbols) * N_DATA * 2, 0.0);
        if (write) std::fill(bits, bits + header_bits, 0);
        if (usable && write) {
            double hfull[128] = {0};
            for (int k = 0; k < N_ACTIVE; ++k) {
                hfull[2 * t.active_idx[k]] = h[2 * k];
                hfull[2 * t.active_idx[k] + 1] = h[2 * k + 1];
            }
            for (int s = 0; s < symbols; ++s) {
                double rr[64], ri[64];
                window_fft(packet, t.train_len + int64_t(s) * symlen + t.cp - t.advance, w, rr, ri, dr, di);
                double* f = full.data() + 2 * 64 * s;
                for (int k = 0; k < N_ACTIVE; ++k) {  // zero forcing
                    const int a = t.active_idx[k];
                    const double hr = hfull[2 * a], hi = hfull[2 * a + 1], d = hr * hr + hi * hi;
                    f[2 * a] = (rr[a] * hr + ri[a] * hi) / d;
                    f[2 * a + 1] = (ri[a] * hr - rr[a] * hi) / d;
                }
                for (int j = 0; j < N_PILOT; ++j) {
                    const double pv = t.pilots[s * N_PILOT + j];
                    residual[2 * (s * N_PILOT + j)] = f[2 * t.pilot_idx[j]] / pv;
                    residual[2 * (s * N_PILOT + j) + 1] = f[2 * t.pilot_idx[j] + 1] / pv;
                }
            }
            phase_track(residual.data(), symbols, tracking, coef.data());
            double fit = 0, phases[130];
            for (int s = 0; s < symbols; ++s) {
                const double* f = full.data() + 2 * 64 * s;
                carrier_phases(coef[2 * s], coef[2 * s + 1], phases);
                for (int k = 0; k < N_DATA; ++k) {
                    const double cr = phases[2 * (32 + t.data_k[k])], ci = phases[2 * (32 + t.data_k[k]) + 1];
                    const int a = t.data_idx[k];
                    data[2 * (s * N_DATA + k)] = f[2 * a] * cr + f[2 * a + 1] * ci;  // / exp(i theta)
                    data[2 * (s * N_DATA + k) + 1] = f[2 * a + 1] * cr - f[2 * a] * ci;
                }
                for (int j = 0; j < N_PILOT; ++j) {
                    fit += sq(residual[2 * (s * N_PILOT + j)] - phases[2 * (32 + t.pilot_k[j])])
                         + sq(residual[2 * (s * N_PILOT + j) + 1] - phases[2 * (32 + t.pilot_k[j]) + 1]);
                }
            }
            evm = 100 * std::sqrt(fit / (symbols * N_PILOT));
            double gains[N_DATA], gsum = 0, gmin = INFINITY, gmax = 0;
            for (int k = 0; k < N_DATA; ++k) {
                const int a = t.data_idx[k];
                gains[k] = sq(hfull[2 * a]) + sq(hfull[2 * a + 1]);
                gsum += gains[k];
                gmin = std::min(gmin, gains[k]);
                gmax = std::max(gmax, gains[k]);
            }
            spread = 10 * std::log10(gmax / gmin);
            const double gmean = gsum / N_DATA;
            for (size_t i = 0; i < t.interleaver.size(); ++i) {  // header rows, carrier-weighted
                const int row = int(i / N_DATA), k = int(i % N_DATA);
                soft[i] = data[2 * (row * N_DATA + k)] * (gains[k] / gmean);
            }
            for (size_t i = 0; i < t.interleaver.size(); ++i) coded[t.interleaver[i]] = soft[i];
            viterbi(coded.data(), header_bits, bits);
            // noise in the guard before the packet
            const int64_t g0 = std::max<int64_t>(0, start - t.guard_far), g1 = std::max<int64_t>(0, start - t.guard_near);
            const int64_t blocks = (g1 - g0) / 64;
            if (blocks > 0) {
                double sum = 0;
                for (int64_t q = 0; q < blocks; ++q) {
                    double gr[64], gi[64], buf[128];
                    rotate(z_in + 2 * (g0 + 64 * q), 64, w, 64 * q, buf, dr, di);
                    for (int m = 0; m < 64; ++m) {
                        gr[m] = buf[2 * m];
                        gi[m] = buf[2 * m + 1];
                    }
                    fft64(gr, gi, -1.0);
                    for (int k = 0; k < N_ACTIVE; ++k) {
                        const int a = t.active_idx[k];
                        sum += sq(gr[a] * 0.125) + sq(gi[a] * 0.125);
                    }
                }
                guard = sum / double(blocks * N_ACTIVE);
            }
        }
        if (write) {
            start_out[out] = start;
            quality_out[out] = c.score;
            coarse_out[out] = c.coarse;
            fine_out[out] = fine;
            noise_out[out] = ltf_noise;
            std::copy(h, h + 2 * N_ACTIVE, channel_out + size_t(out) * 2 * N_ACTIVE);
            smoother_out[out] = smoother;
            usable_out[out] = usable;
            double* st = stats_out + size_t(out) * 6;
            st[0] = guard;
            st[1] = power / N_ACTIVE - ltf_noise / 2;
            st[2] = evm;
            st[3] = hmin;
            st[4] = hmax;
            st[5] = spread;
        }
        ++out;
    }
    return out;
}

}  // extern "C"
