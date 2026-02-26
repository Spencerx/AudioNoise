static inline void pll_describe(float pot[4])
{
	fprintf(stderr, " lpf=%.0f Hz", linear(pot[0], 40.0, 1000.0));
	fprintf(stderr, " tightness=%g", pot[1]);
	fprintf(stderr, " blend=%g", pot[2]);
}

struct {
	// Track the envelope of the incoming signal
	float amplitude, decay, blend;
	struct lfo_state output;

	// Zero-crossing detector
	int samples_since_cross;
	int is_high;

	// "PLL" tracking state
	struct lfo_state tracking;
	struct biquad lpf;
	float tightness;
	float smoothed_freq;
	float phase_error_smoothed;
} pll;

static inline void pll_init(float pot[4])
{
	// Amplitude decay - 40Hz halftime
	pll.decay = (float) pow(0.5, 40.0 / SAMPLES_PER_SEC);

	biquad_lpf(&pll.lpf, linear(pot[0], 40.0, 1000.0), 0.707);
	pll.tightness = pot[1];
	pll.blend = pot[2];
}

static inline float pll_amplitude(float in)
{
	float a = fabsf(in);
	if (a < pll.amplitude)
		a = linear(pll.decay, a, pll.amplitude);
	pll.amplitude = a;
	return a;
}

static inline float pll_step(float in)
{
	float amplitude = pll_amplitude(in);
	float clean_in = biquad_step(&pll.lpf, in);

	// Zero-crossing tracker
	pll.samples_since_cross++;
	float threshold = amplitude * 0.1;

	// What should we do about silence?
	if (threshold < 0.0001f)
		threshold = 0.0001f;

	if (!pll.is_high && clean_in > threshold) {
		pll.is_high = 1;

		float current_freq = SAMPLES_PER_SEC / pll.samples_since_cross;
		if (current_freq > 20.0f && current_freq < 2000.0f)
			pll.smoothed_freq = linear(0.1, pll.smoothed_freq, current_freq);

		pll.samples_since_cross = 0;
	} else if (pll.is_high && clean_in < -threshold) {
		pll.is_high = 0;
	}

	// Phase error tracking (clean_in * quadrature sine wave)
	// Yes, this is sine, not "quadrature sine" (aka cos).
	// I'm not convinced the pll part even works, and the
	// zero-crossing probably does everything. Whatever.
	float cos_out = lfo_step(&pll.tracking, lfo_sinewave);

	float phase_error = clean_in * cos_out;

	// Loop filter. The tightness is controlled by pot[1]
	float loop_filter_decay = pll.tightness * pll.tightness * 0.05f;
	pll.phase_error_smoothed = linear(loop_filter_decay, pll.phase_error_smoothed, phase_error);

	// The VCO frequency is the smoothed base frequency + phase error nudging
	float vco_freq =
	    pll.smoothed_freq +
	    (pll.phase_error_smoothed * pll.smoothed_freq * 0.5f);

	if (vco_freq < 20.0f)
		vco_freq = 20.0f;
	if (vco_freq > 2000.0f)
		vco_freq = 2000.0f;

	// Update the tracking LFO and the output
	// LFO with the same target frequency.
	//
	// Well - two octaves higher for that "Alvin
	// and the chipmunks" accompaniment we are
	// after.
	set_lfo_freq(&pll.tracking, vco_freq);
	set_lfo_freq(&pll.output, vco_freq*4);

	float out = amplitude * lfo_step(&pll.output, lfo_triangle);

	return linear(pll.blend, in, out);
}
