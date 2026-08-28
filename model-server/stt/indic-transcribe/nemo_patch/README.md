# nemo_patch — the one file upstream NeMo is missing

`canary_bhili_ft.nemo` declares:

    tokenizer.custom_tokenizer._target_:
      nemo.collections.common.tokenizers.canary_multilingual_tokenizer.CanaryMultilingualTokenizer

That module exists only in AI4Bharat's NeMo fork (`canary-nemo.tar.gz`). It is the sole
reason the model card says "the stock release ... will fail to load the checkpoint".

Everything else in the checkpoint is stock upstream:
  * `nemo_version: 2.8.0rc0`  (NOT 2.3.0 as the card claims)
  * `target: nemo.collections.asr.models.aed_multitask_models.EncDecMultiTaskModel`
  * `prompt_format: canary2`, standard `ConformerEncoder`, standard preprocessor

`canary_multilingual_tokenizer.py` here is a **verbatim copy** from the fork. It is safe to
drop into upstream because its only imports are `AggregateTokenizer`, `SentencePieceTokenizer`
and `create_spt_model`, and the fork's `canary_tokenizer.py` differs from upstream 3.0.0 by
exactly two lines (the copyright year) -- i.e. those base APIs have not drifted.

Install (done in the Dockerfile, and by spike/install_patch.sh for the venv):

    cp canary_multilingual_tokenizer.py \
       <site-packages>/nemo/collections/common/tokenizers/

Upstream provenance: AI4Bharat NeMo fork, nemo/collections/common/tokenizers/,
from https://indicwhisper.objectstore.e2enetworks.net/indic-canary/canary-nemo.tar.gz
