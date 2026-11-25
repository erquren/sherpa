nomo-pvad
=========

.. caution::

   The code of `nomo-pvad`_ is licensed under Apache License 2.0. Its weights
   are a derivative of the ERes2NetV2 backbone from
   `3D-Speaker <https://github.com/modelscope/3D-Speaker>`_, which is also
   licensed under Apache License 2.0.

   Please read the license before you use it ``commercially``.

   如果你需要把它用于商业目的，请先阅读它的 `协议 <https://huggingface.co/TuyaNomo/nomo-pvad/blob/main/LICENSE>`_ 。

Enrolling a voiceprint requires the informed consent of the speaker. Please
do not use voiceprints to monitor people who have not agreed to it.

Introduction
------------

`silero-vad`_ and `ten-vad`_ answer the question *"is anybody talking?"*.
`nomo-pvad`_ answers a different question: *"is the enrolled speaker
talking?"*. Models of this kind are also called target-speaker VAD or
personalized VAD.

Since the answer depends on who is talking, the model needs two inputs: the
audio stream to analyze, and a d-vector that describes the target speaker. The
d-vector is computed only once, from a short enrollment clip, with one of the
speaker embedding models documented at :ref:`onnx-speaker-identification`. For
`nomo-pvad`_ you must use the 3d-speaker ERes2NetV2 model listed below, because
that is the model `nomo-pvad`_ was trained with: it produces a 192-dimensional
vector, which is exactly what `nomo-pvad`_ expects.

Typical use cases are speaker-based gating (react only to the owner of the
device), and dropping other people's speech before running an expensive ASR
model.

Download models files
---------------------

.. list-table::

 * -
   - Model size
   - Download URL
   - Comment
 * - ``nomo-pvad.onnx``
   - 72.7 MB
   - `Download <https://github.com/erquren/sherpa-onnx/releases/download/vad-models/nomo-pvad.onnx>`_
   - | It supports only 16kHz
     | It is exported with the script described in
     | :ref:`nomo_pvad_export` below
 * - ``3dspeaker_speech_eres2netv2_sv_zh-cn_16k-common.onnx``
   - 71.4 MB
   - `Download <https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_eres2netv2_sv_zh-cn_16k-common.onnx>`_
   - | The enrollment model. It is used once, on the
     | enrollment clip, to compute the d-vector of
     | the target speaker. It is needed even if you
     | already use other speaker embedding models

.. note::

   ``nomo-pvad.onnx`` is currently hosted in a fork of `sherpa-onnx`_ so that
   the model can be reviewed together with the code that uses it. The URL will
   be switched to the official release once the model has been reviewed.

You need both files: ``nomo-pvad.onnx`` for detecting the target speaker, and
the ERes2NetV2 model for computing the d-vector of that speaker. The second
file is used only once per speaker, so if you embed the d-vector into your
application you can ship just ``nomo-pvad.onnx``.

We have added some meta data to the original checkpoint released at
`https://huggingface.co/TuyaNomo/nomo-pvad <https://huggingface.co/TuyaNomo/nomo-pvad>`_,
so please use ``nomo-pvad.onnx`` from the table above. If you use the
``weights/nomo_pvad.pt`` file directly, or export it yourself without the meta
data, `sherpa-onnx`_ will refuse to load it.

We do not provide an ``int8`` version on purpose. Dynamic 8-bit quantization
makes this model about 4x **slower** on CPU while only shrinking it from 72 MB
to 19 MB, so the trade-off is not worth it here.

Note that `nomo-pvad`_ supports only 16kHz samples.

How it works
------------

The model is a single network that does VAD, target-speaker discrimination and
temporal modeling at the same time:

.. list-table::

 * - Stage
   - Detail
 * - Acoustic features
   - 80-dim fbank, 25 ms window / 10 ms hop, ``dither=0``, with the mean
     over time subtracted
 * - Speaker backbone
   - ERes2NetV2 with the final pooling removed. It downsamples time by 8,
     giving 80 ms per frame, then averages over frequency and outputs 1024
     dimensions
 * - Speaker conditioning
   - A FiLM layer turns the 192-dim d-vector into a channel-wise affine
     transform, i.e., ``gamma(z) * (x + beta(z))``, applied to every frame
 * - Temporal modeling
   - Two causal 1-D convolution blocks with GLU residuals, followed by one
     more such block, all with ``d_model=256``
 * - Output
   - Every 2 backbone frames are averaged into a 160 ms chunk, and a small
     classifier emits ``P(target speaker)`` for that chunk

The total number of parameters is 18.2 M, of which 13.9 M belong to the
ERes2NetV2 backbone.

Two details of the feature extraction matter if you want to reproduce the
numbers from the reference implementation. The model was trained with
``torchaudio.compliance.kaldi.fbank``, whose defaults differ from the ones in
`sherpa-onnx`_ in two places: ``snip_edges`` is ``True`` instead of ``False``,
and ``high_freq`` is the Nyquist frequency instead of ``Nyquist - 400``.
`sherpa-onnx`_ already uses the right values for this model, so you only need
to be careful if you re-export it.

The model is also **not** strictly causal: the convolutions in the backbone use
symmetric padding, so the score of a chunk depends on a little of the future.
That is why the reference implementation does not run the network frame by
frame. Instead, it keeps the last 2.4 s of audio and re-scores the whole window
for every new 160 ms chunk, using only the probability of the newest chunk.
`sherpa-onnx`_ does the same, so ``window_size`` and ``window_shift`` of this
model are both 2560 samples (160 ms), while the model internally holds 38400
samples (2.4 s) of history.

Note that this means each 160 ms chunk costs one forward pass over 2.4 s of
audio. On an x86-64 CPU with 4 threads that takes roughly 35 ms, which still
leaves plenty of headroom for real-time use, but it is much heavier than
`silero-vad`_ and `ten-vad`_. If you only need to know whether *anybody* is
talking, use one of those two.

Python API examples
-------------------

.. hint::

   We use ``soundfile`` to read the wave files in the following examples. You
   can also use ``sherpa_onnx.read_wave()``.

Please first download the two models from the table above, plus two wave files:
one with the target speaker to enroll, and one to analyze. The example audio
files used by `nomo-pvad`_ are available at
`https://huggingface.co/TuyaNomo/nomo-pvad/tree/main/examples
<https://huggingface.co/TuyaNomo/nomo-pvad/tree/main/examples>`_.

.. code-block:: python

   import numpy as np
   import sherpa_onnx
   import soundfile as sf

   # ------------------------------------------------------------------
   # Step 1: compute the d-vector of the target speaker. Do it only once
   # and cache the result if you run the detector many times.
   # ------------------------------------------------------------------
   extractor = sherpa_onnx.SpeakerEmbeddingExtractor(
       sherpa_onnx.SpeakerEmbeddingExtractorConfig(
           model="./3dspeaker_speech_eres2netv2_sv_zh-cn_16k-common.onnx"
       )
   )

   samples, sample_rate = sf.read("./enroll.wav", dtype="float32")

   stream = extractor.create_stream()
   stream.accept_waveform(sample_rate, samples)
   stream.input_finished()

   assert extractor.is_ready(stream), "please use a longer enrollment clip"
   embedding = extractor.compute(stream)

   # ------------------------------------------------------------------
   # Step 2: build the detector
   # ------------------------------------------------------------------
   config = sherpa_onnx.VadModelConfig()
   config.nomo_pvad = sherpa_onnx.NomoPvadModelConfig(
       model="./nomo-pvad.onnx",
       speaker_embedding=embedding,
       threshold=0.7,
   )
   config.sample_rate = 16000

   vad = sherpa_onnx.VoiceActivityDetector(config, buffer_size_in_seconds=60)

   # ------------------------------------------------------------------
   # Step 3: feed audio and collect segments. Only chunks in which the
   # enrolled speaker is talking end up in a segment.
   # ------------------------------------------------------------------
   samples, sample_rate = sf.read("./test.wav", dtype="float32")

   window_size = 512  # 32 ms, any size works
   for i in range(0, len(samples), window_size):
       vad.accept_waveform(samples[i : i + window_size])

       while not vad.empty():
           segment = vad.front
           start = segment.start / sample_rate
           duration = len(segment.samples) / sample_rate
           print(f"{start:.2f}s + {duration:.2f}s")
           vad.pop()

   vad.flush()

   while not vad.empty():
       segment = vad.front
       start = segment.start / sample_rate
       duration = len(segment.samples) / sample_rate
       print(f"{start:.2f}s + {duration:.2f}s")
       vad.pop()

For the example files shipped with the model, the first 2.4 s belong to the
target speaker and the rest belongs to somebody else. The script above prints

.. code-block:: text

   0.23s + 2.31s

Choosing a threshold
--------------------

`P(target speaker)` is a probability, so the threshold depends on what a wrong
decision costs you:

.. list-table::

 * - Scenario
   - Threshold
   - Trade-off
 * - Speaker-based gating, where a false acceptance is expensive
   - 0.7 - 0.85
   - Fewer false accepts, slightly more misses
 * - Balanced
   - 0.6
   - A compromise between recall and false accepts
 * - Multi-speaker turn taking, where recall matters more
   - 0.5 - 0.6
   - More false accepts

The default is 0.7.

To avoid flickering at segment boundaries, `sherpa-onnx`_ uses two thresholds:
a chunk starts a segment when its probability is at least ``threshold``, but an
ongoing segment only ends after a chunk drops below ``neg_threshold``, which
defaults to ``max(threshold - 0.15, 0.01)``. A segment is also discarded if it
is shorter than ``min_speech_duration``.

C API examples
--------------

.. list-table::

 * - Filename
   - Comment
 * - `target-speaker-vad-c-api.c <https://github.com/erquren/sherpa-onnx/blob/master/c-api-examples/target-speaker-vad-c-api.c>`_
   - | | Enroll a target speaker with the speaker embedding
       | extractor, then find the segments in which that
       | speaker is talking

The example is self-contained; it needs no models other than the two listed
above:

.. code-block:: bash

   ./target-speaker-vad-c-api \
       ./3dspeaker_speech_eres2netv2_sv_zh-cn_16k-common.onnx \
       ./nomo-pvad.onnx \
       ./enroll.wav \
       ./test.wav

Command-line options
--------------------

If you use `nomo-pvad`_ from a C++ or Python program that parses command-line
flags, the following options are available:

.. list-table::

 * - Option
   - Default
   - Comment
 * - ``--nomo-pvad-model``
   - empty
   - Path to ``nomo-pvad.onnx``. It must be non-empty to select this model
 * - ``--nomo-pvad-speaker-embedding-file``
   - empty
   - A text file with the 192 floats of the d-vector, separated by whitespace
     or commas. It is only used when the embedding is not set
     programmatically
 * - ``--nomo-pvad-threshold``
   - 0.7
   - Probability above which a chunk counts as the target speaker talking
 * - ``--nomo-pvad-min-silence-duration``
   - 0.5
   - In seconds. How long the target speaker has to be silent before a segment
     is closed
 * - ``--nomo-pvad-min-speech-duration``
   - 0.25
   - In seconds. Shorter segments are discarded
 * - ``--nomo-pvad-max-speech-duration``
   - 20
   - In seconds. When a segment gets longer, the threshold is temporarily
     raised to encourage a split
 * - ``--nomo-pvad-neg-threshold``
   - -1
   - Threshold for speech to silence. Negative means
     ``max(threshold - 0.15, 0.01)``

.. _nomo_pvad_export:

How to export the model
-----------------------

If you want to export ``nomo-pvad.onnx`` yourself, or if you want to check how
the meta data is added, please see

`<https://github.com/erquren/sherpa-onnx/blob/master/scripts/nomo-pvad/export-onnx.py>`_

The script downloads the checkpoint from Hugging Face and the ERes2NetV2
backbone from ModelScope, exports the network with ``opset_version=17``,
attaches the meta data that `sherpa-onnx`_ reads at runtime, and finally
verifies that the ONNX model agrees with PyTorch.

.. code-block:: bash

   pip install torch onnx onnxruntime modelscope huggingface_hub
   ./export-onnx.py --output ./nomo-pvad.onnx
