package board

// Voicing is how a board's speaker is voiced (#229): a correction for its
// driver, three compressor bands above the bass guard, a filter on the voice
// plane, and how far full volume drives the chain. Every number is measured
// on that board's own driver and chosen by ear against it, so a board nobody
// has measured has none and plays through the plain chain.
//
// The Python reference is em_eq / em_mbc / em_volume, held to biscuit's
// numbers by the output chain's vectors.
type Voicing struct {
	Correction []Peak // after the user's EQ
	Shelf      Shelf
	// HighpassHz (LR4) and Notch sit under the correction, where its peaks
	// would otherwise boost what the driver cannot play.
	HighpassHz float64
	Notch      Peak

	Crossovers [2]float64 // Hz, between the three upper bands
	Upper      [3]BandLaw // low to high, above the bass guard's crossover
	// UpperFloorDb is the most any upper band takes off; UpperHoldMs holds a
	// reduction before it releases, at UpperReleaseMs per 10dB.
	UpperFloorDb, UpperHoldMs, UpperReleaseMs float64
	// The bass guard under the voicing: the deeper of BassFloorDb and the
	// user's bassGuardDb, held BassHoldMs. Without the holds the gain
	// recovers between a waveform's peaks and is cut on the next.
	BassFloorDb, BassHoldMs float64

	Voice []Peak // the voice plane's own filter, before the mix

	// CeilingDb is full volume's gain into the chain: the plain law
	// (0.5dB per level, unity at 127) lifted by this much, so the top of the
	// range drives the compressors rather than the DAC.
	CeilingDb float64
}

// Peak is a peaking biquad.
type Peak struct{ Hz, Db, Q float64 }

// Shelf is a high shelf, slope 1.
type Shelf struct{ Hz, Db float64 }

// BandLaw is a compressor band's static law and attack.
type BandLaw struct{ Ratio, ThresholdDb, AttackMs float64 }

// biscuitVoicing: fitted to VVV's driver (#247 sweep, 2026-10-16) and picked
// by ear on 2026-10-18; the volume ceiling chosen on 2026-10-19.
var biscuitVoicing = &Voicing{
	Correction: []Peak{
		{174.9, 26.13, 0.388},
		{3277.5, -10.94, 0.753},
		{576.3, -10.93, 0.735},
		{8707.8, 5.52, 2.985},
	},
	Shelf:      Shelf{11874.4, 12.0},
	HighpassHz: 80,
	Notch:      Peak{65, -25, 4},
	Crossovers: [2]float64{500, 7500},
	// The first band is harder than stock's 2:1 from -10, because this
	// driver's 2nd harmonic at 150-300Hz rises with level.
	Upper:          [3]BandLaw{{3, -16, 20}, {2, -10, 10}, {2, -10, 10}},
	UpperFloorDb:   -40,
	UpperHoldMs:    30,
	UpperReleaseMs: 100,
	BassFloorDb:    -40,
	BassHoldMs:     20,
	Voice: []Peak{
		{228.0, 3.0, 0.76},
		{2105.5, -2.8, 1.25},
		{8583.1, -5.17, 5.0},
	},
	CeilingDb: 10,
}

// BiscuitVoicing is biscuit's voicing, for the output chain's tests and its
// default chain.
func BiscuitVoicing() *Voicing { return biscuitVoicing }
