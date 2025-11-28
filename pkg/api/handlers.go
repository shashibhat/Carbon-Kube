package api

import (
    "encoding/json"
    "net/http"
)

type PolicyResolved struct {
    NormalizedWeights map[string]float64 `json:"normalizedWeights"`
}

type JobAnalysis struct {
    Mobility string `json:"mobility"`
    RegionScores map[string]float64 `json:"regionScores"`
}

type RLUpdateRequest struct {
    LearningRate float64 `json:"learningRate"`
    ExplorationRate float64 `json:"explorationRate"`
}

func writeJSON(w http.ResponseWriter, v interface{}) {
    w.Header().Set("Content-Type", "application/json")
    enc := json.NewEncoder(w)
    _ = enc.Encode(v)
}

func HandlePolicyResolved(w http.ResponseWriter, r *http.Request) {
    resp := PolicyResolved{NormalizedWeights: map[string]float64{"carbon": 0.4, "cost": 0.2, "sla": 0.2, "dataGravity": 0.2}}
    writeJSON(w, resp)
}

func HandleJobAnalysis(w http.ResponseWriter, r *http.Request) {
    resp := JobAnalysis{Mobility: "constrained", RegionScores: map[string]float64{"us-east-1": 80, "eu-west-1": 70}}
    writeJSON(w, resp)
}

func HandleRLUpdate(w http.ResponseWriter, r *http.Request) {
    var req RLUpdateRequest
    _ = json.NewDecoder(r.Body).Decode(&req)
    writeJSON(w, map[string]string{"status": "ok"})
}

