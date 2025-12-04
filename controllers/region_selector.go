package controllers

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"sort"
)

type RegionScore struct {
	Region          string
	CarbonIntensity int
}

type RegionSelectorInput struct {
	AllowedRegions    []string
	AvoidRegions      []string
	MaxExtraLatencyMs int
	MobilityLevel     string
	CarbonAPI         string
}

type RegionSelectorOutput struct {
	SelectedRegion string
	Score          float64
	DebugInfo      map[string]string
}

func fetchCarbon(api string, regions []string) ([]RegionScore, error) {
	if api == "" {
		return nil, errors.New("carbon API endpoint required")
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
	best := RegionSelectorOutput{}
	best.Score = -1
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
		dgPenalty := 0.0
		if l > 0 {
			dgPenalty += float64(l) * 0.01
		}
		score := 100.0 - float64(s.CarbonIntensity) - dgPenalty
		if score > best.Score {
			best = RegionSelectorOutput{SelectedRegion: s.Region, Score: score, DebugInfo: map[string]string{"latencyMs": fmt.Sprintf("%d", l), "carbon": fmt.Sprintf("%d", s.CarbonIntensity)}}
		}
	}
	if best.Score < 0 {
		return RegionSelectorOutput{}, errors.New("no region satisfies constraints")
	}
	return best, nil
}
