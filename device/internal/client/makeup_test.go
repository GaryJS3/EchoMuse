package client

import (
	"encoding/binary"
	"testing"
)

// The headroom taken before the canceller comes back exactly, and only a
// sample that would overflow is clamped and counted.
func TestApplyMakeupGivesTheHeadroomBack(t *testing.T) {
	in := []int16{0, 1, -1, 1000, -1000, 16383, -16384, 16384, -16385, 32767}
	want := []int16{0, 2, -2, 2000, -2000, 32766, -32768, 32767, -32768, 32767}
	b := make([]byte, len(in)*2)
	for i, v := range in {
		binary.LittleEndian.PutUint16(b[i*2:], uint16(v))
	}
	if got := applyMakeup(b, aecHeadroom); got != 3 {
		t.Errorf("clamped %d samples, want 3", got)
	}
	for i := range want {
		if got := int16(binary.LittleEndian.Uint16(b[i*2:])); got != want[i] {
			t.Errorf("sample %d: %d, want %d", i, got, want[i])
		}
	}
}
