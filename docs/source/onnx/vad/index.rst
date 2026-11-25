.. _sherpa_onnx_vad:

VAD
===

We support `silero-vad`_, `ten-vad`_ and `nomo-pvad`_ for voice activity
detection.

`silero-vad`_ uses `MIT license <https://github.com/snakers4/silero-vad/blob/master/LICENSE>`_ while `ten-vad`_ uses a
`modified version <https://github.com/TEN-framework/ten-vad/blob/main/LICENSE>`_
of Apache License 2.0

`nomo-pvad`_ is a different kind of model: instead of detecting whether
*anybody* is talking, it detects whether the *enrolled* speaker is talking. It
is licensed under Apache License 2.0, both the code and the weights.

Please read the license of the model before you use it.

.. toctree::
   :maxdepth: 7

   ./silero-vad.rst
   ./ten-vad.rst
   ./nomo-pvad.rst
