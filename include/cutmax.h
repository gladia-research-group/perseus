#pragma once

#include "inference.h"
#include "nonlinear.h"   // CutMaxCalib (the configs.json "cutmax" section)

#include <vector>

struct CutMaxConfig {
    struct Iter {
        int    p;        
        double c;        
        double m;        
        double s2_hi;    
        int    passes;   
        int    ex2; 
        double ca = 0.0;
        double cb = 0.0;
        int casc_iters = 0;
    };
    std::vector<Iter> iters;
    int    newton_per_pass = 4;
    int    newton_polish   = 0;
    double sum_lo = 0.0;          
    double sum_hi = 0.0;
    int    gs_sum_iters = 4;
    double entry_scale = 1.0 / 256.0;
};

CutMaxConfig default_gpt2_cutmax_config();

CutMaxConfig cutmax_config_from_calib(const CutMaxCalib& c);

std::vector<PackedCtx> cutmax_argmax(Inference& inf,
                                     const std::vector<PackedCtx>& tiles,
                                     int vocab, const CutMaxConfig& cfg);

inline int cutmax_tile_col_of_slot(int m, int d, int W_tile) {
    const int a = W_tile / d;
    return (m / a + (m % a) * d) % W_tile;
}
