package controllers

import (
    "encoding/json"
    "errors"
    "net/http"
    "sort"
)

type RegionScore struct {
    Region string
    CarbonIntensity int
}

type RegionSelectorInput struct {
    AllowedRegions []string
    AvoidRegions []string
    MaxExtraLatencyMs int
    MobilityLevel string
    CarbonAPI string
}

type RegionSelectorOutput struct {
    SelectedRegion string
}

func fetchCarbon(api string, regions []string) ([]RegionScore, error) {
    if api == "" {
        out := make([]RegionScore, 0, len(regions))
        for i, r := range regions {
            out = append(out, RegionScore{Region: r, CarbonIntensity: 300 + i*10})
        }
        return out, nil
    }
    resp, err := http.Get(api)
    if err != nil {
        return nil, err
    }
    defer resp.Body.Close()
    var data []RegionScore
    dec := json.NewDecoder(resp.Body)
    if err := dec.Decode(&data); err != nil {
        return nil, err
    }
    return data, nil
}

func SelectRegion(input RegionSelectorInput, latency map[string]int) (RegionSelectorOutput, error) {
    allowed := map[string]bool{}
    for _, r := range input.AllowedRegions {
        allowed[r] = true
    }
    scores, err := fetchCarbon(input.CarbonAPI, input.AllowedRegions)
    if err != nil {
        return RegionSelectorOutput{}, err
    }
    sort.Slice(scores, func(i, j int) bool { return scores[i].CarbonIntensity < scores[j].CarbonIntensity })
    for _, s := range scores {
        if !allowed[s.Region] {
            continue
        }
        skip := false
        for _, a := range input.AvoidRegions {
            if a == s.Region {
                skip = true
                break
            }
        }
        if skip {
            continue
        }
        l := latency[s.Region]
        if l > input.MaxExtraLatencyMs {
            continue
        }
        if input.MobilityLevel == "pinned" {
            if l > 0 {
                continue
            }
        }
        return RegionSelectorOutput{SelectedRegion: s.Region}, nil
    }
    return RegionSelectorOutput{}, errors.New("no region satisfies constraints")
}

