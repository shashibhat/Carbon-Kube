package controllers

import (
	"testing"
	"time"

	"github.com/shashibhat/Carbon-Kube/pkg/providers"
)

func TestTemporalWindow(t *testing.T) {
	t0 := time.Date(2025, 12, 3, 10, 0, 0, 0, time.UTC)
	T := 30 * time.Minute
	dmax := 2 * time.Hour
	asap := t0
	deadline := t0.Add(6 * time.Hour)
	tMin := t0
	tMax := minTime(t0.Add(dmax), deadline.Add(-T), asap.Add(T/2))
	if !tMax.After(tMin) {
		t.Fatalf("expected tMax > tMin")
	}
}

func TestCarbonCostSeries(t *testing.T) {
	start := time.Date(2025, 12, 3, 1, 0, 0, 0, time.UTC)
	T := 1 * time.Hour
	slot := 15 * time.Minute
	points := []providers.ForecastPoint{
		{Timestamp: start, Value: 100},
		{Timestamp: start.Add(30 * time.Minute), Value: 200},
		{Timestamp: start.Add(12 * time.Hour), Value: 300},
	}
	c1 := carbonCostSeries(start, T, slot, points)
	c2 := carbonCostSeries(start.Add(12*time.Hour), T, slot, points)
	if c1 == c2 {
		t.Fatalf("expected different carbon costs across time windows")
	}
}
