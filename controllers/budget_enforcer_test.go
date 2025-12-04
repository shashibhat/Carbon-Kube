package controllers

import "testing"

func TestSafeInt(t *testing.T) {
    if safeInt(5) != 5 {
        t.Fatalf("safeInt failed for int")
    }
    if safeInt(int64(7)) != 7 {
        t.Fatalf("safeInt failed for int64")
    }
    if safeInt(3.0) != 3 {
        t.Fatalf("safeInt failed for float64")
    }
}

func TestFmtInt(t *testing.T) {
    if fmtInt(10) != "10" {
        t.Fatalf("fmtInt failed")
    }
}
