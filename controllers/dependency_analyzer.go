package controllers

import (
    "math"
)

type DependencyInputs struct {
    EgressCost float64
    LatencyToKafkaMs int
    LatencyToDBMs int
    S3ReplicationAvailable bool
}

func DataGravityPenalty(inputs DependencyInputs) float64 {
    costTerm := inputs.EgressCost
    kafkaTerm := float64(inputs.LatencyToKafkaMs) / 10.0
    dbTerm := float64(inputs.LatencyToDBMs) / 10.0
    replicationTerm := 0.0
    if !inputs.S3ReplicationAvailable {
        replicationTerm = 20.0
    }
    penalty := costTerm + kafkaTerm + dbTerm + replicationTerm
    if penalty < 0 {
        penalty = 0
    }
    return math.Min(100.0, penalty)
}

