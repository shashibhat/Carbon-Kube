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

func TestParseDeadlineVariants(t *testing.T) {
	t0 := time.Date(2025, 12, 3, 10, 0, 0, 0, time.UTC)
	// explicit absolute
	abs := parseDeadline("2025-12-03T12:00:00Z", t0, map[string]string{})
	if !abs.Equal(time.Date(2025, 12, 3, 12, 0, 0, 0, time.UTC)) {
		t.Fatalf("absolute parse failed")
	}
	// relative via annotations
	anns := map[string]string{"carbonkube.io/deadlineMode": "Relative", "carbonkube.io/defaultRelativeDeadlineSeconds": "7200"}
	rel := parseDeadline("", t0, anns)
	if !rel.Equal(t0.Add(2 * time.Hour)) {
		t.Fatalf("relative deadline failed")
	}
	// fallback
	fb := parseDeadline("", t0, map[string]string{})
	if !fb.Equal(t0.Add(24 * time.Hour)) {
		t.Fatalf("fallback 24h failed")
	}
}

func TestScheduleWindowFromPolicy(t *testing.T) {
	t0 := time.Now().UTC()
	T := 30 * time.Minute
	tDelay := t0.Add(2 * time.Hour)
	deadline := t0.Add(6 * time.Hour)
	tSlow := t0.Add(T / 2)
	tMin, tMax := computeScheduleWindow(t0, T, tDelay, deadline, tSlow)
	if !tMax.After(tMin) {
		t.Fatalf("tMax should be after tMin")
	}
}

func TestCriticalBypass(t *testing.T) {
	t0 := time.Now().UTC()
	T := 1 * time.Hour
	slot := 15 * time.Minute
	tMin := t0
	tMax := t0.Add(2 * time.Hour)
	points := []providers.ForecastPoint{{Timestamp: t0.Add(1 * time.Hour), Value: 100}}
	final := computeFinalStartTime(t0, tMin, tMax, slot, T, points, 1.0, true)
	if !final.Equal(t0) {
		t.Fatalf("critical should be ASAP")
	}
}

func TestAggressivenessEffect(t *testing.T) {
	t0 := time.Now().UTC()
	T := 1 * time.Hour
	slot := 15 * time.Minute
	tMin := t0
	tMax := t0.Add(2 * time.Hour)
	points := []providers.ForecastPoint{
		{Timestamp: t0, Value: 300},
		{Timestamp: t0.Add(1 * time.Hour), Value: 100},
	}
	high := computeFinalStartTime(t0, tMin, tMax, slot, T, points, 1.0, false)
	low := computeFinalStartTime(t0, tMin, tMax, slot, T, points, 0.1, false)
	if !(high.After(low)) {
		t.Fatalf("high aggressiveness should move further from ASAP")
	}
}

func TestNoForecastPointsSticksToASAP(t *testing.T) {
	t0 := time.Now().UTC()
	T := 1 * time.Hour
	slot := 15 * time.Minute
	tMin := t0
	tMax := t0.Add(2 * time.Hour)
	final := computeFinalStartTime(t0, tMin, tMax, slot, T, []providers.ForecastPoint{}, 1.0, false)
	if !final.Equal(t0) {
		t.Fatalf("no points should keep ASAP")
	}
}
