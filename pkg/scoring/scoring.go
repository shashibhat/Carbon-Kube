package scoring

import (
    "math"
)

type CarbonScores struct {
    CarbonScore float64
    CostScore float64
    SLARisk float64
    DataGravityPenalty float64
}

type PolicyWeights struct {
    CarbonWeight float64
    CostWeight float64
    SLARiskWeight float64
    DataGravityWeight float64
}

type PolicyConstraints struct {
    HighCarbonLimit int
    ExtremeCarbonLimit int
    MaxSLAIncreasePercent int
}

func Normalize(values []float64) []float64 {
    sum := 0.0
    for _, v := range values {
        sum += math.Abs(v)
    }
    if sum == 0 {
        return make([]float64, len(values))
    }
    out := make([]float64, len(values))
    for i, v := range values {
        out[i] = v / sum
    }
    return out
}

func ComputeScore(scores CarbonScores, weights PolicyWeights, constraints PolicyConstraints, regionCarbonIntensity int, slaIncreasePercent int) float64 {
    normalized := Normalize([]float64{weights.CarbonWeight, weights.CostWeight, weights.SLARiskWeight, weights.DataGravityWeight})
    cw := normalized[0]
    kw := normalized[1]
    sw := normalized[2]
    dw := normalized[3]
    base := cw*scores.CarbonScore + kw*scores.CostScore + sw*scores.SLARisk + dw*scores.DataGravityPenalty
    penalty := 0.0
    if regionCarbonIntensity > constraints.HighCarbonLimit {
        penalty += 0.1 * float64(regionCarbonIntensity-constraints.HighCarbonLimit)
    }
    if regionCarbonIntensity > constraints.ExtremeCarbonLimit {
        penalty += 0.2 * float64(regionCarbonIntensity-constraints.ExtremeCarbonLimit)
    }
    if slaIncreasePercent > constraints.MaxSLAIncreasePercent {
        penalty += 10.0
    }
    score := base - penalty
    if score < 0 {
        score = 0
    }
    if score > 100 {
        score = 100
    }
    return score
}

