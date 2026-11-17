#pragma once
// SHoEbox core: byte-level sparse experts, multiscale state and online learning.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>
#if defined(__AVX512F__) || defined(__AVX2__)
#include <immintrin.h>
#endif

namespace shoe {
constexpr int D=64, K=8, BLOCK=8, GROUP=4, NEG=16, VOCAB=256;
using Vec=std::array<float,D>;
using Mat=std::array<float,D*D>;
using Small=std::array<float,K>;
using Memory=std::array<float,K*K>;
inline float dot(const float* a,const float* b,int n=D){
    float s=0; for(int j=0;j<n;++j)s+=a[j]*b[j]; return s;
}
inline void routing_dot16(const float* keys,const float* cue,float* scores){
    // Each lane is a separate expert. No horizontal reduction, reassociation,
    // or FMA: arithmetic order matches dot() exactly, including tie behavior.
#if defined(__AVX512F__)
    __m512 sum=_mm512_setzero_ps();
    for(int j=0;j<D;++j)
        sum=_mm512_add_ps(sum,_mm512_mul_ps(_mm512_loadu_ps(keys+j*16),_mm512_set1_ps(cue[j])));
    _mm512_storeu_ps(scores,sum);
#elif defined(__AVX2__)
    __m256 lo=_mm256_setzero_ps(),hi=_mm256_setzero_ps();
    for(int j=0;j<D;++j){
        __m256 x=_mm256_set1_ps(cue[j]);
        lo=_mm256_add_ps(lo,_mm256_mul_ps(_mm256_loadu_ps(keys+j*16),x));
        hi=_mm256_add_ps(hi,_mm256_mul_ps(_mm256_loadu_ps(keys+j*16+8),x));
    }
    _mm256_storeu_ps(scores,lo);_mm256_storeu_ps(scores+8,hi);
#else
    std::fill(scores,scores+16,0.f);
    for(int j=0;j<D;++j)for(int lane=0;lane<16;++lane)scores[lane]+=keys[j*16+lane]*cue[j];
#endif
}
inline float energy(const Vec& a){return dot(a.data(),a.data());}
inline void normalize(Vec& a){float s=1/std::sqrt(energy(a)+1e-8f);for(auto& x:a)x*=s;}
inline float sigmoid(float x){return 1/(1+std::exp(-x));}
inline float softplus(float x){return std::max(x,0.f)+std::log1p(std::exp(-std::abs(x)));}
inline void mv(const Mat& a,const Vec& x,Vec& y){for(int i=0;i<D;++i)y[i]=dot(a.data()+i*D,x.data());}
inline void vjp(const Mat& a,const Vec& g,Vec& x){
    for(int i=0;i<D;++i)for(int j=0;j<D;++j)x[j]+=a[i*D+j]*g[i];
}
inline void outer(Mat& a,const Vec& g,const Vec& x){
    for(int i=0;i<D;++i)for(int j=0;j<D;++j)a[i*D+j]+=g[i]*x[j];
}
template<class T> void step(T& p,const T& g,float lr){
    for(size_t j=0;j<p.size();++j)p[j]-=lr*std::clamp(g[j],-2.f,2.f);
}

#include "address_memory.hpp"
#include "recall.hpp"

struct Expert {Mat w{};Vec bias{},key{};};
struct Shared {
    Mat recurrent{},gain{},bias{},predict{};
    std::array<float,D*K> read{};
    std::array<float,GROUP*D> role{};
    Vec length{},offset{};
};
struct Event {
    Vec input{},content{},residual{},state{},before{},gain_grad{},bias_grad{},task{},native{};
    std::array<Vec,GROUP> children{};
    std::array<float,GROUP> lengths{};
    Memory memory{};
    Small key{},value{},read{};
    Vec forecast{},unit{};
    float unit_norm=1,gate=1,gate_derivative=0,span=1;
    int expert=0,runner=0,byte=0,uses=0,has_target=0,reconstruction_role=0,content_id=-1;
};
// Content-only eligibility: no recurrent state, memory, readout or replay.
// Its weights stay frozen until this bounded publication epoch retires.
struct Constituent {
    Vec input{},content{},residual{},grad{};
    std::array<Vec,GROUP> children{};
    std::array<float,GROUP> lengths{};
    std::array<int,GROUP> child_ids{{-1,-1,-1,-1}};
    float gate=1,gate_derivative=0,span=1;
    int expert=0,runner=0,byte=0;
};
struct Stream {
    std::array<Event,BLOCK> events{};
    std::array<Vec,NEG> negatives{};
    std::array<Vec,GROUP> children{};
    std::array<float,GROUP> lengths{};
    std::array<int,GROUP> child_ids{{-1,-1,-1,-1}};
    Vec state{},content{},gain{},bias{};
    Memory memory{};
    // Kept outside Event: the byte loop does not clear a slow memory record.
    AddressStream addressed{};
    std::array<AddressEvent,BLOCK> address_events{};
    std::array<Vec,GROUP> stable_children{};
    Vec stable{};
    int count=0,children_count=0,negative_count=0,negative_cursor=0;
    uint64_t executions=0,updates=0,native_targets=0;
    double native_loss=0;
    uint64_t reconstruction_targets=0;
    double reconstruction_loss=0;
};
#include "routing.hpp"

struct Level {
    std::vector<Expert> experts;
    // Derived exact-search cache. Coordinates are contiguous across 16 experts
    // so SIMD preserves each dot product's original accumulation order.
    // Rebuilt after load; never serialized or counted as fitted parameters.
    std::vector<float> routing_keys;
    RoutingState routing{};
    Shared p{};
    AddressParameters address{};
    RecallParameters retrieval{};
    RecallStream retained{};
    RecallRead read_now{};
    Stream s{};
    std::vector<Constituent> constituents;
    uint64_t credit_edges=0,content_updates=0;
    double task_gradient_energy=0,child_gradient_energy=0;
};

struct Model {
    static bool summary(int scale){return scale>=2;}
    std::vector<Level> levels;
    std::array<Vec,VOCAB> embedding{};
    std::array<Vec,255> decoder{};
    std::array<float,255> decoder_bias{};
    Vec cached_gain{},cached_bias{},features{};
    bool training=true;
    uint64_t bytes=0,route_comparisons=0;
    float learning_rate=.008f,slow_rate=.08f,query_rate=.008f,head_rate=.16f;
    float aux_weight=.1f,route_rate=.002f,reconstruction_weight=.05f;
    int epoch_bytes=0,epoch_limit=BLOCK;
    explicit Model(int count=4,int target=1000000,uint32_t seed=7);
    void rebuild_index(Level& l);
    void update_index(Level& l,int expert);
    void routing_prepare(int scale,Event& e,int slot);
    void forward_event(int scale,const std::array<Vec,GROUP>& child,const std::array<float,GROUP>& lengths,int byte);
    void flush(int scale);
    void retire_constituents();
    void refresh();
    void observe(uint8_t byte);
    float teach(uint8_t target);
    void log_probs(float* out)const;
    double run(const uint8_t* x,const uint8_t* y,size_t n);
    void reset();
    void finish();
    void inference(bool cold);
    size_t parameter_count()const;
};

inline int coord(int a,int b){return a*8+b;}
inline float sign(int a,int b){return (a+b)%3==0?-1.f:1.f;}
inline void context_forward(Event& e,Memory& memory){
    e.memory=memory;float norm=1;
    for(int a=0;a<K;++a){float q=0,v=0;
        for(int b=0;b<K;++b){int j=coord(a,b);float s=sign(a,b)*.3535533905932738f;
            q+=s*e.content[j];v+=s*e.content[(j+19)%D];}
        e.key[a]=std::tanh(q);e.value[a]=std::tanh(v);norm+=e.key[a]*e.key[a];
    }
    for(int b=0;b<K;++b){float read=0;for(int a=0;a<K;++a)read+=memory[a*K+b]*e.key[a];
        e.read[b]=read;float err=e.value[b]-.98f*read;
        for(int a=0;a<K;++a)memory[a*K+b]=.98f*memory[a*K+b]+.5f*e.key[a]*err/norm;
    }
}
// Full local VJP through both the read and normalized delta write. The entering
// matrix is detached at the eight-event boundary, just like entering h.
inline void context_backward(const Event& e,const Small& read_grad,Memory& next,Vec& dc){
    Memory before{};Small dq{},dv{},dr=read_grad;float norm=1,dnorm=0;
    for(float q:e.key)norm+=q*q;
    for(int a=0;a<K;++a)for(int b=0;b<K;++b){
        float g=next[a*K+b],err=e.value[b]-.98f*e.read[b];
        before[a*K+b]+=.98f*g;dq[a]+=.5f*g*err/norm;
        dv[b]+=.5f*g*e.key[a]/norm;dnorm-=.5f*g*e.key[a]*err/(norm*norm);
    }
    for(int b=0;b<K;++b)dr[b]-=.98f*dv[b];
    for(int a=0;a<K;++a){dq[a]+=2*e.key[a]*dnorm;
        for(int b=0;b<K;++b){before[a*K+b]+=e.key[a]*dr[b];dq[a]+=e.memory[a*K+b]*dr[b];}
        dq[a]*=1-e.key[a]*e.key[a];dv[a]*=1-e.value[a]*e.value[a];
        for(int b=0;b<K;++b){int j=coord(a,b);float s=sign(a,b)*.3535533905932738f;
            dc[j]+=s*dq[a];dc[(j+19)%D]+=s*dv[a];}
    }next=before;
}

struct Writer {
    std::string out;
    template<class T> void pod(const T& v){static_assert(std::is_trivially_copyable<T>::value);out.append(reinterpret_cast<const char*>(&v),sizeof(v));}
    void pod(bool v){pod(uint8_t(v));}
    template<class... T> void fields(const T&... values){(pod(values),...);}
    template<class T> void vector(const std::vector<T>& v,size_t){uint64_t n=v.size();pod(n);for(const auto& a:v)pod(a);}
};
struct Reader {
    const char* p;size_t n;
    template<class T> void pod(T& v){static_assert(std::is_trivially_copyable<T>::value);if(n<sizeof(v))throw std::runtime_error("truncated checkpoint");std::memcpy(&v,p,sizeof(v));p+=sizeof(v);n-=sizeof(v);}
    void pod(bool& v){uint8_t x;pod(x);if(x>1)throw std::runtime_error("invalid boolean");v=bool(x);}
    template<class... T> void fields(T&... values){(pod(values),...);}
    template<class T> void vector(std::vector<T>& v,size_t limit){uint64_t size;pod(size);if(size>limit||size>n/sizeof(T))throw std::runtime_error("invalid checkpoint vector");v.resize(size);for(auto& a:v)pod(a);}
};
std::string save(const Model& m);
Model load(const char* p,size_t n);
} // namespace shoe
