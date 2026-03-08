from parler_tts import ParlerTTSForConditionalGeneration
from transformers.generation.streamers import BaseStreamer
from typing import Optional, Union
import os
import torch
import numpy as np
import math
from scipy.io import wavfile
from ragged_parler_utils import *


class ParlerTTSStreamer(BaseStreamer):
    def __init__(
        self,
        model: ParlerTTSForConditionalGeneration,
        device: Optional[str] = None,
        play_steps: Optional[int] = 10,
        stride: Optional[int] = None,
    ):
        """
        Streamer that stores playback-ready audio chunks in a list. Use chunk_list_size() to check
        how many chunks are available and pop_latest_chunk() to consume them.
        Parameters:
            model (`ParlerTTSForConditionalGeneration`):
                The Parler-TTS model used to generate the audio waveform.
            device (`str`, *optional*):
                The torch device on which to run the computation. If `None`, will default to the device of the model.
            play_steps (`int`, *optional*, defaults to 10):
                The number of generation steps with which to return the generated audio array. Using fewer steps will
                mean the first chunk is ready faster, but will require more codec decoding steps overall. This value
                should be tuned to your device and latency requirements.
            stride (`int`, *optional*):
                The window (stride) between adjacent audio samples. Using a stride between adjacent audio samples reduces
                the hard boundary between them, giving smoother playback. If `None`, will default to a value equivalent to
                play_steps // 6 in the audio space.
        """
        self.decoder = model.decoder
        self.audio_encoder = model.audio_encoder
        self.generation_config = model.generation_config
        self.device = device if device is not None else model.device
        self.use_audio_scales = model.use_audio_scales
        self.use_4dim_audio_codes = model.use_4dim_audio_codes
        self.audio_kwargs = {}
        if self.use_audio_scales:
            self.audio_kwargs["audio_scales"] = [None]

        # variables used in the streaming process
        self.play_steps = play_steps
        if stride is not None:
            self.stride = stride
        else:
            hop_length = math.floor(self.audio_encoder.config.sampling_rate / self.audio_encoder.config.frame_rate)
            self.stride = hop_length * (play_steps - self.decoder.num_codebooks) // 6
        self.token_cache = None
        self.to_yield = 0

        # list of finalized audio chunks (use chunk_list_size() and pop_latest_chunk() to consume)
        self.audio_chunks_list: list[np.ndarray] = []
        self.stramer_eos_flag = False
        self.streamer_bos = True
    def apply_delay_pattern_mask(self, input_ids):
        # build the delay pattern mask for offsetting each codebook prediction by 1 (this behaviour is specific to Parler)
        _, delay_pattern_mask = self.decoder.build_delay_pattern_mask(
            input_ids[:, :1],
            bos_token_id=self.generation_config.bos_token_id,
            pad_token_id=self.generation_config.decoder_start_token_id,
            max_length=input_ids.shape[-1],
        )
        # apply the pattern mask to the input ids
        input_ids = self.decoder.apply_delay_pattern_mask(input_ids, delay_pattern_mask)

        # revert the pattern delay mask by filtering the pad token id
        mask = (delay_pattern_mask != self.generation_config.bos_token_id) & (delay_pattern_mask != self.generation_config.pad_token_id)
        input_ids = input_ids[mask].reshape(1, self.decoder.num_codebooks, -1)

        if self.use_4dim_audio_codes:
            # append the frame dimension back to the audio codes
            input_ids = input_ids[None, ...]

        # send the input_ids to the correct device
        input_ids = input_ids.to(self.audio_encoder.device)

        decode_sequentially = (
            self.generation_config.bos_token_id in input_ids
            or self.generation_config.pad_token_id in input_ids
            or self.generation_config.eos_token_id in input_ids
        )
        if not decode_sequentially:
            sample = self.audio_encoder.decode(
                audio_codes=input_ids,
                **self.audio_kwargs,
            ).audio_values
            output_values = sample if sample.ndim == 3 else sample.unsqueeze(0)
        else:
            sample = input_ids[:, 0] if self.use_4dim_audio_codes else input_ids[0]
            sample_mask = ((sample >= self.audio_encoder.config.codebook_size).sum(dim=(0, 1)) == 0) if self.use_4dim_audio_codes else ((sample >= self.audio_encoder.config.codebook_size).sum(dim=0) == 0)
            sample = sample[:, :, sample_mask] if self.use_4dim_audio_codes else sample[:, sample_mask]
            sample = self.audio_encoder.decode(audio_codes=sample[None, ...], **self.audio_kwargs).audio_values
            output_values = sample if sample.ndim == 3 else sample.unsqueeze(0)

        audio_values = output_values[0, 0]
        return audio_values.cpu().float().detach().numpy()

    def put(self, value):
        if self.streamer_bos:
            #print("Streamer BOS flag set to False")
            self.streamer_bos = False
            
        if not self.stramer_eos_flag:
            batch_size = value.shape[0] // self.decoder.num_codebooks
            if batch_size > 1:
                raise ValueError("ParlerTTSStreamer only supports batch size 1")

            if self.token_cache is None:
                self.token_cache = value
            else:
                self.token_cache = torch.concatenate([self.token_cache, value[:, None]], dim=-1)

            #if self.token_cache.shape[-1] % self.play_steps == 0:
            #    audio_values = self.apply_delay_pattern_mask(self.token_cache)
            #    chunk = audio_values[self.to_yield : -self.stride]
            #    self.audio_chunks_list.append(chunk.copy())
            #    self.to_yield += len(audio_values) - self.to_yield - self.stride

    def end(self):
        if not self.stramer_eos_flag:
            """Flushes any remaining cache into the chunk list."""
            if self.token_cache is not None:
                audio_values = self.apply_delay_pattern_mask(self.token_cache)
            else:
                audio_values = np.zeros(self.to_yield)
            final_chunk = audio_values[self.to_yield :]
            self.audio_chunks_list.append(final_chunk.copy())
            #print("Streamer EOS flag set to True")
            self.stramer_eos_flag = True

    def chunk_list_size(self) -> int:
        """Return the number of audio chunks in the list."""
        return len(self.audio_chunks_list)

    def get_chunk(self) -> Optional[np.ndarray]:
        """Remove and return the most recently added audio chunk, or None if the list is empty."""
        if not self.audio_chunks_list:
            return None
        return self.audio_chunks_list.pop()

    def clear_chunks(self):
        self.audio_chunks_list.clear()

    def save_chunks_to_file(
        self,
        path: Union[str, os.PathLike[str]],
        sampling_rate: Optional[int] = None,
    ):
        if not self.audio_chunks_list:
            raise ValueError("No audio chunks in the list to save")
        rate = sampling_rate if sampling_rate is not None else self.audio_encoder.config.sampling_rate
        audio = np.concatenate(self.audio_chunks_list, axis=0)
        if np.issubdtype(audio.dtype, np.floating):
            audio = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
        wavfile.write(path, rate, audio)
