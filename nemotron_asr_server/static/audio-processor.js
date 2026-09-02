/**
 * AudioWorkletProcessor: mono float32 mic audio -> 160 ms Int16 PCM frames at 16 kHz.
 *
 * The page asks for a 16 kHz AudioContext, in which case this is pure format conversion.
 * When a browser ignores that request (Safari < 14.1 and friends) it falls back to
 * resampling here instead of failing: 48000/16000 is an exact 3:1 ratio, and measured
 * against native 16 kHz audio this path produced identical transcripts in quiet and
 * typical conditions, differing by a single character only on a deliberately noisy mic.
 * A hard failure would be a far worse trade than that.
 */
class PCMProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const opts = (options && options.processorOptions) || {};
    this.targetSampleRate = 16000;
    this.targetChunkSamples = 2560;                 // 160 ms @ 16 kHz

    // `sampleRate` is the AudioContext's real rate, provided by the worklet global.
    this.needsResample = Math.abs(sampleRate - this.targetSampleRate) > 1;
    this.resampleRatio = sampleRate / this.targetSampleRate;
    this.sourceIndex = 0.0;
    if (this.needsResample) {
      this.port.postMessage({ notice: 'resampling', from: sampleRate });
    }

    this.outputBuffer = new Int16Array(this.targetChunkSamples);
    this.outputOffset = 0;
  }

  emit(sample) {
    const clamped = Math.max(-1.0, Math.min(1.0, sample));
    this.outputBuffer[this.outputOffset++] =
      clamped < 0 ? Math.round(clamped * 0x8000) : Math.round(clamped * 0x7FFF);

    if (this.outputOffset >= this.targetChunkSamples) {
      const chunk = this.outputBuffer.slice(0, this.targetChunkSamples);
      let sumSq = 0, peak = 0;
      for (let k = 0; k < chunk.length; k++) {
        const v = chunk[k] / 32768;
        sumSq += v * v;
        const av = v < 0 ? -v : v;
        if (av > peak) peak = av;
      }
      this.port.postMessage(
        { pcm: chunk.buffer, rms: Math.sqrt(sumSq / chunk.length), peak },
        [chunk.buffer]
      );
      this.outputOffset = 0;
    }
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || !input[0]) return true;
    const ch = input[0];

    if (!this.needsResample) {
      for (let i = 0; i < ch.length; i++) this.emit(ch[i]);
      return true;
    }

    // Linear interpolation, with the fractional read position carried across blocks so
    // the phase does not reset at every render quantum.
    while (this.sourceIndex < ch.length) {
      const idx = Math.floor(this.sourceIndex);
      const frac = this.sourceIndex - idx;
      const s0 = ch[idx];
      const s1 = (idx + 1 < ch.length) ? ch[idx + 1] : s0;
      this.emit(s0 + frac * (s1 - s0));
      this.sourceIndex += this.resampleRatio;
    }
    this.sourceIndex -= ch.length;
    return true;
  }
}

registerProcessor('pcm-processor', PCMProcessor);
