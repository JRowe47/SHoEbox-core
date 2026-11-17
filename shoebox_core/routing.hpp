#pragma once
// Included inside namespace shoe. Centered exact routing and its local derivatives.
struct RoutingTrace {
    Vec cue{},expert_input{},key_difference{};
    float norm=1;
};
struct RoutingState {
    Vec mean{};
    std::vector<Vec> seed_cues;
    std::vector<float> bias;
    std::vector<uint64_t> selections,window;
    std::array<RoutingTrace,BLOCK> events{};
    std::vector<RoutingTrace> constituents;
    uint64_t seen=0;
    int seeded=0,seed_count=0,window_count=0;
    void initialize(size_t n) {
        bias.resize(n);selections.resize(n);window.resize(n);seed_cues.resize(4*n);
    }
    void reset_stream(){events={};constituents.clear();}
};

inline Vec routing_vjp(const RoutingTrace& r,const Vec& delta){
    Vec out{};float radial=dot(r.cue.data(),delta.data());
    for(int j=0;j<D;++j)out[j]=(delta[j]-radial*r.cue[j])/r.norm;
    return out;
}
