#include "native.hpp"
#include <memory>

namespace shoe {
Model::Model(int count,int target,uint32_t seed){
    if(count<1||count>4||target<50000||target>100000000)throw std::invalid_argument("levels must be 1..4 and parameters 50,000..100,000,000");
    levels.resize(count);
    for(int scale=1;scale<count;++scale)epoch_limit*=GROUP;
    std::mt19937 rng(seed);std::normal_distribution<float> normal(0,1);
    for(auto& v:embedding)for(auto& x:v)x=.35f*normal(rng);
    for(auto& v:decoder)for(auto& x:v)x=.02f*normal(rng);
    // Level zero fits recurrence/read maps, but no slow composer or projector.
    int fixed=int(sizeof(embedding)/sizeof(float)+sizeof(decoder)/sizeof(float)+decoder_bias.size());
    for(int scale=0;scale<count;++scale)
        fixed+=(scale?(summary(scale)?D*K+9*D:int(sizeof(Shared)/sizeof(float)))+int(sizeof(AddressParameters)/sizeof(float)):D*D+D*K)+int(sizeof(RecallParameters)/sizeof(float));
    int available=(target-fixed)/int(sizeof(Expert)/sizeof(float));
    int banks=std::min(count,2);
    if(available<banks*2)throw std::invalid_argument("parameter budget too small for this topology");
    const float shares[4]{.5f,.25f,.15f,.1f};float total=0;for(int l=0;l<banks;++l)total+=shares[l];
    int left=available;
    for(int scale=0;scale<count;++scale){
        auto& l=levels[scale];
        int span=1;for(int i=0;i<scale;++i)span*=GROUP;
        l.constituents.reserve(epoch_limit/span+1);
        int e=summary(scale)?0:(scale==banks-1?left:std::min(left-2*(banks-1-scale),std::max(2,int(available*shares[scale]/total))));
        left-=e;
        l.experts.resize(e);l.routing.initialize(e);
        for(auto& expert:l.experts){
            for(auto& x:expert.w)x=.025f*normal(rng);
            for(auto& x:expert.key)x=normal(rng);
            normalize(expert.key);
        }
        for(int i=0;i<D;++i){
            l.p.recurrent[i*D+i]=.45f;
            if(!summary(scale)){
                l.p.gain[i*D+i]=.05f/count;
                l.p.bias[i*D+i]=.35f/count;
                l.p.predict[i*D+i]=.15f;
            }
            for(int j=0;j<K;++j)l.p.read[i*K+j]=.025f*normal(rng);
            for(int k=0;k<GROUP;++k)l.p.role[k*D+i]=.5f;
            l.p.length[i]=.01f*normal(rng);
        }
        if(scale>0){
            for(auto& x:l.address.read)x=.025f*normal(rng);
            for(int i=0;i<TIME_D;++i)l.address.query_clock[i*TIME_D+i]=.25f;
            for(int i=0;i<D;++i)l.address.reconstruct[i*D+i]=.15f;
        }
        for(auto& x:l.retrieval.query)x=.02f*normal(rng);
        for(auto& x:l.retrieval.read)x=.02f*normal(rng);
        rebuild_index(l);
    }
}
size_t Model::parameter_count()const{
    size_t n=sizeof(embedding)/sizeof(float)+sizeof(decoder)/sizeof(float)+decoder_bias.size();
    for(size_t i=0;i<levels.size();++i)n+=(i?(summary(int(i))?D*K+9*D:sizeof(Shared)/sizeof(float))+sizeof(AddressParameters)/sizeof(float):D*D+D*K)+sizeof(RecallParameters)/sizeof(float)+levels[i].experts.size()*sizeof(Expert)/sizeof(float);
    return n;
}
void Model::rebuild_index(Level& l){
    l.routing_keys.clear();
    if(l.experts.size()>=256){
        l.routing_keys.resize(((l.experts.size()+15)/16)*D*16);
        for(size_t k=0;k<l.experts.size();++k)
            for(int j=0;j<D;++j)l.routing_keys[(k/16)*D*16+j*16+k%16]=l.experts[k].key[j];
    }
}
void Model::update_index(Level& l,int expert){
    if(!l.routing_keys.empty())
        for(int j=0;j<D;++j)l.routing_keys[(expert/16)*D*16+j*16+expert%16]=l.experts[expert].key[j];
}

#include "routing_impl.hpp"

void Model::forward_event(int scale,const std::array<Vec,GROUP>& child,const std::array<float,GROUP>& lengths,int byte){
    auto& l=levels[scale];auto& s=l.s;
    const bool compact=summary(scale);
    // Retirement occurs before the next event. No output-lifetime gradient can
    // arrive after this update. Cross-block native forecasts are truncated.
    if(training&&s.count==BLOCK)flush(scale);
    int slot=training?s.count:0;auto& e=s.events[slot];e=Event{};e.byte=byte;e.before=s.state;
    e.reconstruction_role=int(s.executions%GROUP);
    if(scale==0){e.input=embedding[byte];e.span=1;s.stable=stable_bytes()[byte];}
    else {
        s.stable=stable_group(s.stable_children);
        e.children=child;e.lengths=lengths;e.span=0;for(float n:lengths)e.span+=n;
        for(int i=0;i<D;++i){float z=l.p.offset[i];
            for(int k=0;k<GROUP;++k){int j=(i*(2*k+1)+13*k)%D;
                z+=l.p.role[k*D+i]*child[k][j];
                z+=l.p.length[i]*(k+1)*lengths[k]/std::max(e.span,1.f)*.25f;
            }e.input[i]=std::tanh(z);
        }
    }
    if(scale==0){
        // Canonical byte content is independent of context. All fast experts
        // now model a transition conditioned on both content and prior state.
        for(int i=0;i<D;++i){e.content[i]=std::tanh(embedding[byte][i]);
            e.input[i]=std::tanh(e.content[i]+.5f*e.before[i]);}
    }
    if(compact){e.content=e.input;e.expert=e.runner=-1;}
    else routing_prepare(scale,e,slot);
    if(!compact){const auto& expert=l.experts[e.expert];
        mv(expert.w,l.routing.events[slot].expert_input,e.residual);
        if(scale)for(int i=0;i<D;++i)e.content[i]=std::tanh(e.input[i]+expert.bias[i]+e.gate*e.residual[i]);
    }

    if(training){
        e.content_id=int(l.constituents.size());Constituent c;
        c.input=e.input;c.content=e.content;c.residual=e.residual;c.children=e.children;c.lengths=e.lengths;
        c.child_ids=s.child_ids;c.gate=e.gate;c.gate_derivative=e.gate_derivative;c.span=e.span;
        c.expert=e.expert;c.runner=e.runner;c.byte=byte;l.constituents.push_back(c);
        if(!compact)l.routing.constituents.push_back(l.routing.events[slot]);
    }
    context_forward(e,s.memory);
    if(scale>0)address_forward(s.address_events[slot],e.content,e.before,s.stable,s.addressed,l.address);
    Vec recurrent{};
    if(compact)for(int i=0;i<D;++i)recurrent[i]=l.p.recurrent[i*D+i]*e.before[i];
    else mv(l.p.recurrent,e.before,recurrent);
    for(int i=0;i<D;++i){float a=scale?dot(l.address.read.data()+i*ADDRESS_V,s.address_events[slot].read.data(),ADDRESS_V):0;
        float transition=scale?0:l.experts[e.expert].bias[i]+e.gate*e.residual[i];
        e.state[i]=std::tanh(e.content[i]+recurrent[i]+dot(l.p.read.data()+i*K,e.read.data(),K)+a+transition);}
    if(training&&scale>0&&!compact){
        // Same-scale contrastive next-unit objective. Only older published
        // contents enter negatives. Neither target nor negatives receive VJPs.
        if(s.count>0&&s.negative_count>0){
            auto& prev=s.events[s.count-1];Vec target=e.content;normalize(target);
            std::array<float,NEG+1> scores{},probs{};std::array<Vec,NEG+1> candidates{};
            candidates[0]=target;int n=1;
            for(int k=0;k<s.negative_count;++k)if(dot(target.data(),s.negatives[k].data())<.9999f)candidates[n++]=s.negatives[k];
            if(n>1){
                float mx=-1e30f;for(int k=0;k<n;++k){scores[k]=dot(prev.unit.data(),candidates[k].data())/.2f;mx=std::max(mx,scores[k]);}
                float denom=0;for(int k=0;k<n;++k){probs[k]=std::exp(scores[k]-mx);denom+=probs[k];}
                s.native_loss+=std::log(denom)+mx-scores[0];++s.native_targets;
                Vec dp{};for(int k=0;k<n;++k){float g=(probs[k]/denom-(k==0))/.2f;for(int j=0;j<D;++j)dp[j]+=g*candidates[k][j];}
                float radial=dot(dp.data(),prev.unit.data());
                for(int j=0;j<D;++j)prev.native[j]=(dp[j]-radial*prev.unit[j])/prev.unit_norm;
                prev.has_target=1;
            }
        }
        mv(l.p.predict,e.state,e.forecast);e.unit=e.forecast;
        e.unit_norm=std::sqrt(energy(e.forecast)+1e-8f);for(auto& x:e.unit)x/=e.unit_norm;
        Vec detached=e.content;normalize(detached);s.negatives[s.negative_cursor]=detached;
        s.negative_cursor=(s.negative_cursor+1)%NEG;s.negative_count=std::min(NEG,s.negative_count+1);
    }
    s.state=e.state;s.content=e.content;++s.executions;
    l.retained.admit(s.stable,e.content);
    if(training)++s.count;
    if(compact){std::copy_n(l.p.gain.begin(),D,s.gain.begin());std::copy_n(l.p.bias.begin(),D,s.bias.begin());}
    else if(scale>0){mv(l.p.gain,s.state,s.gain);mv(l.p.bias,s.state,s.bias);}
    if(scale+1<int(levels.size())){
        auto& parent=levels[scale+1].s;int k=parent.children_count++;
        parent.children[k]=e.content;parent.lengths[k]=e.span;
        parent.child_ids[k]=e.content_id;
        parent.stable_children[k]=s.stable;
        if(parent.children_count==GROUP){parent.children_count=0;forward_event(scale+1,parent.children,parent.lengths,0);}
    }
}

void Model::flush(int scale){
    auto& l=levels[scale];auto& s=l.s;if(!s.count)return;
    const bool compact=summary(scale);
    Shared g{};AddressParameters ag{};std::array<Expert,BLOCK> eg{};std::array<int,BLOCK> selected{};int unique=0;
    std::array<Vec,BLOCK> key_delta{},runner_delta{};
    int uses=0,targets=0;for(int t=0;t<s.count;++t){uses+=s.events[t].uses;targets+=s.events[t].has_target;}
    float task_scale=1.f/std::max(1,uses),native_scale=aux_weight/std::max(1,targets);
    Vec future{};Memory memory_grad{};AddressMatrix address_grad{};
    for(int t=s.count-1;t>=0;--t){
        const auto& e=s.events[t];
        Vec dh=future,dgain{},dbias{},dp{},task{};
        for(int i=0;i<D;++i){task[i]=task_scale*e.task[i];dgain[i]=task_scale*e.gain_grad[i];dbias[i]=task_scale*e.bias_grad[i];dp[i]=native_scale*e.native[i];}
        if(compact){for(int i=0;i<D;++i){g.gain[i]+=dgain[i];g.bias[i]+=dbias[i];}}
        else if(scale>0){vjp(l.p.gain,dgain,task);vjp(l.p.bias,dbias,task);outer(g.gain,dgain,e.state);outer(g.bias,dbias,e.state);
            vjp(l.p.predict,dp,dh);outer(g.predict,dp,e.state);}

        l.task_gradient_energy+=energy(task);
        for(int i=0;i<D;++i)dh[i]+=task[i];
        Vec dz{},dc{};Small dr{};AddressValue ar{};future={};
        for(int i=0;i<D;++i){dz[i]=dh[i]*(1-e.state[i]*e.state[i]);dc[i]=dz[i];
            for(int j=0;j<K;++j){g.read[i*K+j]+=dz[i]*e.read[j];dr[j]+=l.p.read[i*K+j]*dz[i];}
            if(scale>0)for(int j=0;j<ADDRESS_V;++j){ag.read[i*ADDRESS_V+j]+=dz[i]*s.address_events[t].read[j];ar[j]+=l.address.read[i*ADDRESS_V+j]*dz[i];}}
        if(compact)for(int i=0;i<D;++i){g.recurrent[i*D+i]+=dz[i]*e.before[i];future[i]+=l.p.recurrent[i*D+i]*dz[i];}
        else {outer(g.recurrent,dz,e.before);vjp(l.p.recurrent,dz,future);}
        context_backward(e,dr,memory_grad,dc);
        if(scale>0){
            address_backward(s.address_events[t],e.content,e.before,l.address,ar,address_grad,dc,future,ag);
            if(reconstruction_weight>0){s.reconstruction_loss+=reconstruct(e.content,e.children[e.reconstruction_role],e.reconstruction_role,
                    l.address,reconstruction_weight/s.count,dc,ag);++s.reconstruction_targets;}
        }else{
            int k=0;for(;k<unique;++k)if(selected[k]==e.expert)break;
            if(k==unique)selected[unique++]=e.expert;
            const auto& expert=l.experts[e.expert];Vec du{},gated=dz;float dgate=dot(dz.data(),e.residual.data());
            for(int i=0;i<D;++i){eg[k].bias[i]+=dz[i];gated[i]*=e.gate;}
            const auto& input=l.routing.events[t].expert_input;
            outer(eg[k].w,gated,input);vjp(expert.w,gated,du);
            float margin=dgate*e.gate_derivative;
            Vec route_grad{};for(int i=0;i<D;++i)route_grad[i]=margin*l.routing.events[t].key_difference[i];
            route_grad=routing_vjp(l.routing.events[t],route_grad);
            const auto& cue=l.routing.events[t].cue;
            for(int i=0;i<D;++i){key_delta[t][i]=margin*cue[i];runner_delta[t][i]=-margin*cue[i];
                du[i]+=route_grad[i];
                float input_grad=du[i]*(1-e.input[i]*e.input[i]);dc[i]+=input_grad;future[i]+=.5f*input_grad;}
        }
        // Keep the exact publication-time content VJP until all parent credit
        // has arrived. Context weights still retire at eight native events.
        auto& c=l.constituents.at(e.content_id);for(int i=0;i<D;++i)c.grad[i]+=dc[i];
    }
    float lr=(scale?slow_rate:learning_rate);
    if(compact)for(int i=0;i<D;++i)l.p.recurrent[i*D+i]-=lr*std::clamp(g.recurrent[i*D+i],-2.f,2.f);
    else step(l.p.recurrent,g.recurrent,lr);
    step(l.p.read,g.read,lr);
    if(scale>0){
        if(compact)for(int i=0;i<D;++i){l.p.gain[i]-=lr*std::clamp(g.gain[i],-2.f,2.f);l.p.bias[i]-=lr*std::clamp(g.bias[i],-2.f,2.f);}
        else {step(l.p.gain,g.gain,lr);step(l.p.bias,g.bias,lr);step(l.p.predict,g.predict,lr);}
        step(l.address.query_content,ag.query_content,lr);step(l.address.query_state,ag.query_state,lr);
        step(l.address.query_clock,ag.query_clock,lr);step(l.address.read,ag.read,lr);
        step(l.address.reconstruct,ag.reconstruct,lr);step(l.address.reconstruct_bias,ag.reconstruct_bias,lr);}
    else{
        for(int k=0;k<unique;++k){float rate=learning_rate;
            step(l.experts[selected[k]].w,eg[k].w,rate);step(l.experts[selected[k]].bias,eg[k].bias,rate);}
        for(int t=0;t<s.count;++t){const auto& e=s.events[t];auto& key=l.experts[e.expert].key;auto& runner=l.experts[e.runner].key;
            float factor=1.f;
            step(key,key_delta[t],learning_rate*factor);if(e.runner!=e.expert)step(runner,runner_delta[t],learning_rate);
            Vec cue=l.routing.events[t].cue;normalize(cue);for(int i=0;i<D;++i)key[i]+=(route_rate*factor)*(cue[i]-key[i]);
            normalize(key);if(e.runner!=e.expert)normalize(runner);}
        for(int t=0;t<s.count;++t){update_index(l,s.events[t].expert);update_index(l,s.events[t].runner);}
    }
    s.count=0;++s.updates;
}

void Model::retire_constituents(){
    // A reverse pass through a bounded tree of observed constituents only.
    // No recurrent graph, stale-weight replay, or unbounded ancestry is kept.
    for(int scale=int(levels.size())-1;scale>=0;--scale){
        auto& l=levels[scale];if(l.constituents.empty()){l.s.child_ids.fill(-1);continue;}
        if(scale==0){
            std::array<Vec,VOCAB> gradient{};
            for(const auto& c:l.constituents)for(int i=0;i<D;++i)
                gradient[c.byte][i]+=c.grad[i]*(1-c.content[i]*c.content[i]);
            for(int b=0;b<VOCAB;++b)step(embedding[b],gradient[b],learning_rate);
        }else if(summary(scale)){
            // Ordered content composition remains trainable; no expert,
            // router derivative, forecast head or refinement tape exists here.
            Shared g{};
            for(const auto& c:l.constituents){
                std::array<Vec,GROUP> child_grad{};
                for(int i=0;i<D;++i){float a=c.grad[i]*(1-c.input[i]*c.input[i]);g.offset[i]+=a;
                    for(int pos=0;pos<GROUP;++pos){int j=(i*(2*pos+1)+13*pos)%D;
                        g.role[pos*D+i]+=a*c.children[pos][j];
                        g.length[i]+=a*(pos+1)*c.lengths[pos]/std::max(1.f,c.span)*.25f;
                        child_grad[pos][j]+=a*l.p.role[pos*D+i];}}
                auto& child=levels[scale-1];
                for(int pos=0;pos<GROUP;++pos)if(c.child_ids[pos]>=0){
                    auto& destination=child.constituents.at(c.child_ids[pos]).grad;
                    for(int i=0;i<D;++i)destination[i]+=child_grad[pos][i];
                    child.child_gradient_energy+=energy(child_grad[pos]);++child.credit_edges;}
            }
            step(l.p.role,g.role,slow_rate);step(l.p.length,g.length,slow_rate);step(l.p.offset,g.offset,slow_rate);
        }else{
            Shared g{};std::vector<Expert> eg;std::vector<int> selected;
            eg.reserve(l.constituents.size());selected.reserve(l.constituents.size());
            std::vector<Vec> keys(l.experts.size());std::vector<uint8_t> touched(l.experts.size());
            size_t ci=0;
            for(const auto& c:l.constituents){
                const RoutingTrace* rt=&l.routing.constituents[ci];++ci;
                int k=0;for(;k<int(selected.size());++k)if(selected[k]==c.expert)break;
                if(k==int(selected.size())){selected.push_back(c.expert);eg.emplace_back();}
                const auto& expert=l.experts[c.expert];Vec pre{},du{},dc=c.grad;float dgate=0;
                const Vec* base=&c.content;

                for(int i=0;i<D;++i){pre[i]=dc[i]*(1-(*base)[i]*(*base)[i]);eg[k].bias[i]+=pre[i];du[i]=pre[i];dgate+=pre[i]*c.residual[i];}
                Vec gated=pre;for(auto& x:gated)x*=c.gate;outer(eg[k].w,gated,rt->expert_input);vjp(expert.w,gated,du);
                float margin=dgate*c.gate_derivative;
                touched[c.expert]=touched[c.runner]=1;
                Vec route_grad{};for(int i=0;i<D;++i)route_grad[i]=margin*rt->key_difference[i];
                route_grad=routing_vjp(*rt,route_grad);
                const auto& cue=rt->cue;
                for(int i=0;i<D;++i){keys[c.expert][i]+=margin*cue[i];keys[c.runner][i]-=margin*cue[i];du[i]+=route_grad[i];}
                std::array<Vec,GROUP> child_grad{};
                for(int i=0;i<D;++i){float a=du[i]*(1-c.input[i]*c.input[i]);g.offset[i]+=a;
                    for(int pos=0;pos<GROUP;++pos){int j=(i*(2*pos+1)+13*pos)%D;
                        g.role[pos*D+i]+=a*c.children[pos][j];g.length[i]+=a*(pos+1)*c.lengths[pos]/std::max(1.f,c.span)*.25f;
                        child_grad[pos][j]+=a*l.p.role[pos*D+i];}}
                auto& child=levels[scale-1];
                for(int pos=0;pos<GROUP;++pos)if(c.child_ids[pos]>=0){
                    auto& destination=child.constituents.at(c.child_ids[pos]).grad;
                    for(int i=0;i<D;++i)destination[i]+=child_grad[pos][i];
                    child.child_gradient_energy+=energy(child_grad[pos]);++child.credit_edges;}
            }
            float shared_rate=slow_rate;
            step(l.p.role,g.role,shared_rate);step(l.p.length,g.length,shared_rate);step(l.p.offset,g.offset,shared_rate);
            for(size_t k=0;k<selected.size();++k){float rate=slow_rate;
                step(l.experts[selected[k]].w,eg[k].w,rate);step(l.experts[selected[k]].bias,eg[k].bias,rate);}
            for(size_t k=0;k<keys.size();++k)if(touched[k])step(l.experts[k].key,keys[k],slow_rate);
            ci=0;
            for(const auto& c:l.constituents){auto& key=l.experts[c.expert].key;Vec cue=l.routing.constituents[ci].cue;++ci;normalize(cue);
                float rate=route_rate;
                for(int i=0;i<D;++i)key[i]+=rate*(cue[i]-key[i]);}
            for(size_t k=0;k<keys.size();++k)if(touched[k])normalize(l.experts[k].key);
            for(size_t k=0;k<keys.size();++k)if(touched[k])update_index(l,int(k));
        }
        l.constituents.clear();l.routing.constituents.clear();l.s.child_ids.fill(-1);++l.content_updates;
    }
    epoch_bytes=0;
}
void Model::refresh(){
    cached_gain={};cached_bias={};
    for(size_t l=1;l<levels.size();++l){
        const auto& s=levels[l].s;
        for(int i=0;i<D;++i){int j=i;cached_gain[i]+=s.gain[j];cached_bias[i]+=s.bias[j];}
    }
}
void Model::observe(uint8_t byte){
    if(training&&epoch_bytes==epoch_limit)finish();
    forward_event(0,{}, {},byte);++bytes;if(training)++epoch_bytes;
    // Projections themselves are cached per slow event. Summation is 3x64.
    refresh();
    // Retained pairs contain observed input only. Queries execute now, including
    // when no slow level completes on this byte; they never consume the target.
    for(size_t l=0;l<levels.size();++l){
        auto& level=levels[l];
        recall_forward(level.retained,level.retrieval,levels[0].s.state,byte,level.read_now);
    }
    for(int i=0;i<D;++i)features[i]=std::tanh(levels[0].s.state[i]*(1+cached_gain[i])+cached_bias[i]);
}
float Model::teach(uint8_t target){
    int node=0;float loss=0;Vec feature_grad{};std::array<RecallValue,4> recall_grad{};
    const float effective_head=head_rate,effective_query=query_rate;
    for(int depth=0;depth<8;++depth){
        int bit=(target>>(7-depth))&1;float z=decoder_bias[node]+dot(decoder[node].data(),features.data());
        for(size_t l=0;l<levels.size();++l)z+=dot(levels[l].retrieval.read.data()+node*RECALL_V,levels[l].read_now.value.data(),RECALL_V);
        loss+=softplus(z)-bit*z;
        if(training){float error=sigmoid(z)-bit;
            for(size_t l=0;l<levels.size();++l){auto& p=levels[l].retrieval;auto& read=levels[l].read_now;
                float lr=effective_head/(1+dot(read.value.data(),read.value.data(),RECALL_V));
                for(int j=0;j<RECALL_V;++j){recall_grad[l][j]+=error*p.read[node*RECALL_V+j];p.read[node*RECALL_V+j]-=lr*error*read.value[j];}}
            // Read the exact weights which produced z before changing them.
            for(int j=0;j<D;++j)feature_grad[j]+=error*decoder[node][j];
            float lr=effective_head/(1+energy(features));
            for(int j=0;j<D;++j)decoder[node][j]-=lr*error*features[j];
            decoder_bias[node]-=effective_head*.05f*error;
        }
        node=2*node+1+bit;
    }
    if(training){
        auto& fast=levels[0].s.events[levels[0].s.count-1];
        for(size_t l=0;l<levels.size();++l){RecallGradient g{};Vec di{};recall_backward(levels[l].retained,levels[l].retrieval,levels[l].read_now,recall_grad[l],g,di);

            step(levels[l].retrieval.query,g.query,effective_query);
            for(int j=0;j<RECALL_K;++j)
                levels[l].retrieval.query_byte[levels[l].read_now.byte*RECALL_K+j]-=effective_query*std::clamp(g.query_byte[j],-2.f,2.f);
            for(int i=0;i<D;++i)fast.task[i]+=di[i];}
        for(int i=0;i<D;++i){float g=feature_grad[i]*(1-features[i]*features[i]);fast.task[i]+=g*(1+cached_gain[i]);
            for(size_t l=1;l<levels.size();++l){auto& s=levels[l].s;if(!s.count)continue;
                s.events[s.count-1].gain_grad[i]+=g*levels[0].s.state[i];s.events[s.count-1].bias_grad[i]+=g;}
        }
        for(auto& l:levels)if(l.s.count)++l.s.events[l.s.count-1].uses;
    }
    return loss;
}
void Model::log_probs(float* out)const{
    std::array<float,511> p{};
    for(int n=0;n<255;++n){float z=decoder_bias[n]+dot(decoder[n].data(),features.data());
        for(size_t l=0;l<levels.size();++l)z+=dot(levels[l].retrieval.read.data()+n*RECALL_V,levels[l].read_now.value.data(),RECALL_V);
        p[2*n+1]=p[n]-softplus(z);p[2*n+2]=p[n]-softplus(-z);}
    std::copy(p.begin()+255,p.end(),out);
}
double Model::run(const uint8_t* x,const uint8_t* y,size_t n){
    if(!n)throw std::invalid_argument("at least one prediction is required");
    double loss=0;for(size_t t=0;t<n;++t){observe(x[t]);loss+=teach(y[t]);}return loss/n;
}
void Model::finish(){if(training){for(size_t l=0;l<levels.size();++l)flush(int(l));retire_constituents();}}
void Model::reset(){
    finish();
    for(auto& l:levels){
        l.routing.reset_stream();l.s=Stream{};l.retained=RecallStream{};l.read_now=RecallRead{};
        l.credit_edges=l.content_updates=0;l.task_gradient_energy=l.child_gradient_energy=0;
    }
    cached_gain={};cached_bias={};features={};bytes=0;route_comparisons=0;
}
void Model::inference(bool cold){
    // Freeze the current weights without applying or draining pending credit.
    training=false;epoch_bytes=0;
    for(auto& l:levels){
        l.routing.reset_stream();l.constituents.clear();l.s.child_ids.fill(-1);
        l.s.count=0;l.s.events={};l.s.address_events={};l.s.negatives={};
        l.s.negative_count=0;l.s.negative_cursor=0;
    }
    if(cold)reset();
}

// The core owns a separate checkpoint format; no historical architecture
// dispatch or research payloads are included. Serialize members, not padding.
template<class IO,class S> void stream_state(IO& io,S& s){
    io.fields(s.events,s.negatives,s.children,s.lengths,s.child_ids,s.state,s.content,s.gain,s.bias,s.memory);
    io.fields(s.addressed.memory,s.addressed.previous_stable,s.addressed.clock.phase,s.addressed.clock.ticks);
    io.fields(s.address_events,s.stable_children,s.stable,s.count,s.children_count,s.negative_count,s.negative_cursor);
    io.fields(s.executions,s.updates,s.native_targets,s.native_loss,s.reconstruction_targets,s.reconstruction_loss);
}
template<class IO,class M> void model_state(IO& io,M& m){
    io.fields(m.training,m.embedding,m.decoder,m.decoder_bias,m.cached_gain,m.cached_bias,m.features);
    io.fields(m.bytes,m.route_comparisons,m.learning_rate,m.slow_rate,m.query_rate,m.head_rate,
              m.aux_weight,m.route_rate,m.reconstruction_weight,m.epoch_bytes);
    for(size_t scale=0;scale<m.levels.size();++scale){auto& l=m.levels[scale];
        const size_t experts=l.experts.size();io.vector(l.experts,experts);
        if(l.experts.size()!=experts)throw std::runtime_error("checkpoint expert count mismatch");
        io.fields(l.p,l.address,l.retrieval);
        stream_state(io,l.s);
        io.fields(l.retained.units,l.retained.previous,l.retained.previous_content,l.retained.last,
                  l.retained.have_previous,l.retained.clock,l.retained.count);
        io.fields(l.read_now.input,l.read_now.byte,l.read_now.query,l.read_now.value,l.read_now.weights);
        size_t span=size_t(1)<<(2*scale);
        io.vector(l.constituents,m.epoch_limit/span+1);
        io.fields(l.credit_edges,l.content_updates,l.task_gradient_energy,l.child_gradient_energy);
        auto& q=l.routing;
        io.fields(q.mean);io.vector(q.seed_cues,4*experts);io.vector(q.bias,experts);
        io.vector(q.selections,experts);io.vector(q.window,experts);io.fields(q.events);
        io.vector(q.constituents,m.epoch_limit/span+1);
        io.fields(q.seen,q.seeded,q.seed_count,q.window_count);
        if(q.seed_cues.size()!=4*experts||q.bias.size()!=experts||q.selections.size()!=experts||q.window.size()!=experts)
            throw std::runtime_error("checkpoint router size mismatch");
    }
}
std::string save(const Model& m){
    Writer w;
    const uint64_t magic=0x3145524f43454853ULL;
    const uint32_t format=1,endian=0x01020304;
    // Array layouts contain only homogeneous primitives. Structs used as
    // vectors are additionally guarded by these sizes.
    const std::array<uint32_t,6> abi{{sizeof(Expert),sizeof(Shared),sizeof(Event),
        sizeof(Constituent),sizeof(RoutingTrace),sizeof(AddressEvent)}};
    const int count=int(m.levels.size()),target=int(m.parameter_count());
    w.fields(magic,format,endian,abi,count,target);model_state(w,m);
    return w.out;
}
Model load(const char* p,size_t n){
    Reader r{p,n};uint64_t magic;uint32_t format,endian;std::array<uint32_t,6> abi;int count,target;
    r.fields(magic,format,endian,abi,count,target);
    const std::array<uint32_t,6> expected{{sizeof(Expert),sizeof(Shared),sizeof(Event),
        sizeof(Constituent),sizeof(RoutingTrace),sizeof(AddressEvent)}};
    if(magic!=0x3145524f43454853ULL||format!=1||endian!=0x01020304||abi!=expected)
        throw std::runtime_error("incompatible SHoEbox-core checkpoint");
    Model m(count,target,7);model_state(r,m);
    if(r.n)throw std::runtime_error("unexpected trailing checkpoint data");
    for(float rate:{m.learning_rate,m.slow_rate,m.query_rate,m.head_rate,m.aux_weight,m.route_rate,m.reconstruction_weight})
        if(!std::isfinite(rate)||rate<0)throw std::runtime_error("invalid learning rate");
    if(m.epoch_bytes<0||m.epoch_bytes>m.epoch_limit||(!m.training&&m.epoch_bytes))
        throw std::runtime_error("invalid constituent epoch");
    for(size_t scale=0;scale<m.levels.size();++scale){auto& l=m.levels[scale];auto& s=l.s;auto& q=l.routing;
        auto valid_route=[&](int expert,int runner){return m.summary(int(scale))?
            expert==-1&&runner==-1:expert>=0&&expert<int(l.experts.size())&&runner>=0&&runner<int(l.experts.size());};
        auto valid_child=[&](int id){return id==-1||(scale&&id>=0&&id<int(m.levels[scale-1].constituents.size()));};
        if(s.count<0||s.count>BLOCK||s.children_count<0||s.children_count>=GROUP||s.negative_count<0||s.negative_count>NEG||s.negative_cursor<0||s.negative_cursor>=NEG)
            throw std::runtime_error("invalid event counters");
        if((!m.training&&(!l.constituents.empty()||s.count))||q.constituents.size()!=(m.summary(int(scale))?0:l.constituents.size()))
            throw std::runtime_error("invalid pending credit");
        for(const auto& c:l.constituents){
            if(!valid_route(c.expert,c.runner)||c.byte<0||c.byte>=VOCAB)throw std::runtime_error("invalid constituent");
            for(int id:c.child_ids)if(!valid_child(id))throw std::runtime_error("invalid constituent child");
        }
        for(int id:s.child_ids)if(!valid_child(id))throw std::runtime_error("invalid pending child");
        for(int i=0;i<s.count;++i){const auto& e=s.events[i];
            if(!valid_route(e.expert,e.runner)||e.byte<0||e.byte>=VOCAB||e.content_id<0||e.content_id>=int(l.constituents.size())||e.reconstruction_role<0||e.reconstruction_role>=GROUP)
                throw std::runtime_error("invalid event");
        }
        if(l.retained.count<0||l.retained.count>RECALL_SLOTS||l.read_now.byte<0||l.read_now.byte>=VOCAB)
            throw std::runtime_error("invalid recall counters");
        for(int i=0;i<l.retained.count;++i)if(!l.retained.units[i].born||l.retained.units[i].born>l.retained.clock)
            throw std::runtime_error("invalid retained occurrence");
        if(scale&&s.addressed.clock.ticks!=s.executions)throw std::runtime_error("invalid native clock");
        if(q.seeded<0||q.seeded>1||q.seed_count<0||q.seed_count>int(q.seed_cues.size())||q.window_count<0||q.window_count>=256)
            throw std::runtime_error("invalid routing counters");
        for(float v:q.mean)if(!std::isfinite(v))throw std::runtime_error("invalid routing mean");
        for(float v:q.bias)if(!std::isfinite(v))throw std::runtime_error("invalid routing bias");
        m.rebuild_index(l);
    }
    return m;
}
} // namespace shoe

namespace {
thread_local std::string error;
template<class F> int guard(F f){try{f();error.clear();return 0;}catch(const std::exception& e){error=e.what();return -1;}}
}
extern "C" {
const char* ms_error(){return error.c_str();}
void* ms_create(int levels,int target,uint32_t seed){void* p=nullptr;guard([&]{p=new shoe::Model(levels,target,seed);});return p;}
void ms_destroy(void* p){delete static_cast<shoe::Model*>(p);}
void* ms_clone(void* p,int inference,int cold){void* out=nullptr;guard([&]{auto q=std::make_unique<shoe::Model>(*static_cast<shoe::Model*>(p));if(inference)q->inference(cold);out=q.release();});return out;}
int ms_run(void* p,const uint8_t* x,const uint8_t* y,size_t n,double* loss){return guard([&]{*loss=static_cast<shoe::Model*>(p)->run(x,y,n);});}
int ms_observe(void* p,uint8_t byte,float* out){return guard([&]{auto& m=*static_cast<shoe::Model*>(p);if(m.training)throw std::logic_error("observe requires inference; use run for training");m.observe(byte);m.log_probs(out);});}
int ms_logits(void* p,float* out){return guard([&]{static_cast<shoe::Model*>(p)->log_probs(out);});}
int ms_reset(void* p){return guard([&]{static_cast<shoe::Model*>(p)->reset();});}
int ms_finish(void* p){return guard([&]{static_cast<shoe::Model*>(p)->finish();});}
int ms_configure(void* p,const float* rates){return guard([&]{
    for(int i=0;i<7;++i)if(!std::isfinite(rates[i])||rates[i]<0)throw std::invalid_argument("rates must be finite and nonnegative");
    auto& m=*static_cast<shoe::Model*>(p);
    m.learning_rate=rates[0];m.slow_rate=rates[1];m.query_rate=rates[2];m.head_rate=rates[3];
    m.aux_weight=rates[4];m.route_rate=rates[5];m.reconstruction_weight=rates[6];
});}
int ms_stats(void* p,double* out){return guard([&]{
    const auto& m=*static_cast<shoe::Model*>(p);std::fill_n(out,80,0.);
    out[0]=m.parameter_count();out[1]=m.bytes;out[2]=m.training;out[3]=m.levels.size();
    out[4]=m.epoch_bytes;out[5]=m.epoch_limit;out[6]=m.route_comparisons;
    out[8]=m.learning_rate;out[9]=m.slow_rate;out[10]=m.query_rate;out[11]=m.head_rate;
    out[12]=m.aux_weight;out[13]=m.route_rate;out[14]=m.reconstruction_weight;
    for(size_t i=0;i<m.levels.size();++i){const auto& l=m.levels[i];auto a=out+16+16*i;
        a[0]=l.experts.size();a[1]=l.s.executions;a[2]=l.s.updates;a[3]=l.s.count;
        a[4]=l.constituents.size();a[5]=l.s.children_count;a[6]=l.retained.count;
        a[7]=l.s.native_targets;a[8]=l.s.native_loss;a[9]=l.s.reconstruction_targets;
        a[10]=l.s.reconstruction_loss;a[11]=shoe::energy(l.s.state);a[12]=l.routing.seen;
        a[13]=l.routing.seeded;a[14]=l.credit_edges;a[15]=l.content_updates;
    }
});}
size_t ms_save_size(void* p){size_t n=0;guard([&]{n=shoe::save(*static_cast<shoe::Model*>(p)).size();});return n;}
int ms_save(void* p,char* out,size_t n){return guard([&]{auto data=shoe::save(*static_cast<shoe::Model*>(p));if(n!=data.size())throw std::runtime_error("checkpoint buffer size mismatch");std::memcpy(out,data.data(),n);});}
void* ms_load(const char* p,size_t n){void* out=nullptr;guard([&]{out=new shoe::Model(shoe::load(p,n));});return out;}
}
