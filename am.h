//
// Silly amplitude modulation signal generator "effect"
// It doesn't actually care about the input, it's useful
// mainly for testing the LFO (and generating signals
// for testing other effects)
//
static struct {
	struct lfo_state base_lfo, mod_lfo;
	float depth, volume;
} am;

static inline void am_init(float pot1, float pot2, float pot3, float pot4)
{
	float freq, lfo;

	am.volume = pot1;
	freq = fastpow(8000, pot2)+100;
	set_lfo_freq(&am.base_lfo, freq);

	am.depth = pot3;
	lfo = 1 + 10*pot4;			// 1..11 Hz
	set_lfo_freq(&am.mod_lfo, lfo);

	fprintf(stderr, "am:");
	fprintf(stderr, " volume=%g", am.volume);
	fprintf(stderr, " freq=%g Hz", freq);
	fprintf(stderr, " depth=%g", am.depth);
	fprintf(stderr, " lfo=%g Hz\n", lfo);
}

static inline float am_step(float in)
{
	float val = lfo_step(&am.base_lfo, lfo_sinewave);
	float mod = lfo_step(&am.mod_lfo, lfo_sinewave);
	float multiplier = 1 + mod * am.depth;

	return val * multiplier * am.volume;
}
