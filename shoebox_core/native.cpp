// Mixture of facet softmaxes over one live node's legal children.
// The caller supplies contiguous FP32 arrays. Source links select the key rows
// before this kernel runs. No expert matrix is read by the controller.
#include <algorithm>
#include <cmath>
#include <cstddef>

namespace {
constexpr int width = 8;
constexpr float inv_sqrt_width = 0.3535533905932738f;

float inner(const float* a, const float* b) {
    float sum = 0;
    for (int r = 0; r < width; ++r) sum += a[r] * b[r];
    return sum;
}
}

extern "C" void facet_forward(int parents, int facets, int children, int key_stride,
        const float* queries, const float* keys, const float* mixture,
        float* probabilities, float* output) {
    for (int p = 0; p < parents; ++p) {
        const float* local_keys = keys + size_t(p) * key_stride;
        float* combined = output + size_t(p) * children;
        std::fill_n(combined, children, 0.f);
        for (int f = 0; f < facets; ++f) {
            size_t row = size_t(p) * facets + f;
            const float* q = queries + row * width;
            float* prob = probabilities + row * children;
            for (int c = 0; c < children; ++c)
                prob[c] = inv_sqrt_width * inner(q, local_keys + c * width);
            float largest = *std::max_element(prob, prob + children);
            float sum = 0;
            for (int c = 0; c < children; ++c) {
                prob[c] = std::exp(prob[c] - largest);
                sum += prob[c];
            }
            for (int c = 0; c < children; ++c) {
                prob[c] /= sum;
                combined[c] += mixture[row] * prob[c];
            }
        }
    }
}

extern "C" void facet_backward(int parents, int facets, int children, int key_stride,
        const float* queries, const float* keys, const float* mixture,
        const float* probabilities, const float* incoming,
        float* grad_query, float* grad_key, float* grad_mix_logits) {
    // Python zeroes the accumulation buffers. A stride of zero shares child
    // keys across the parents and directly coalesces their derivatives.
    for (int p = 0; p < parents; ++p) {
        const float* local_keys = keys + size_t(p) * key_stride;
        float* dk = grad_key + size_t(p) * key_stride;
        const float* g = incoming + size_t(p) * children;
        float mix_mean = 0;
        for (int f = 0; f < facets; ++f) {
            size_t row = size_t(p) * facets + f;
            const float* q = queries + row * width;
            const float* prob = probabilities + row * children;
            float* dq = grad_query + row * width;
            float mean = 0;
            for (int c = 0; c < children; ++c) mean += prob[c] * g[c];
            grad_mix_logits[row] = mean;
            mix_mean += mixture[row] * mean;
            for (int c = 0; c < children; ++c) {
                float d = mixture[row] * prob[c] * (g[c] - mean) * inv_sqrt_width;
                for (int r = 0; r < width; ++r) {
                    dq[r] += d * local_keys[c * width + r];
                    dk[c * width + r] += d * q[r];
                }
            }
        }
        for (int f = 0; f < facets; ++f) {
            size_t row = size_t(p) * facets + f;
            grad_mix_logits[row] = mixture[row] * (grad_mix_logits[row] - mix_mean);
        }
    }
}
