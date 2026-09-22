# Text Embeddings Inference (CPU) with Qwen3-Embedding-0.6B baked in at the commit the
# engine pins (kyb.documents.embedding_models.QWEN3_EMBEDDING_0_6B). Used instead of a
# Tinfoil `models:` artifact: the measured tinfoil-config.yml pins this image's digest, so
# the weights are part of what attestation proves. The safetensors checksum is verified
# against the value Hugging Face publishes for that commit, so a build never silently ships
# different bytes.

FROM python:3.12-slim@sha256:2f17fc044b579bab302c2e8054d3a686e2cb9a83de48e70534b94cd8ebbe06a9 AS weights

ARG MODEL_REPO=Qwen/Qwen3-Embedding-0.6B
ARG MODEL_COMMIT=97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3
ARG SAFETENSORS_SHA256=0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd

ENV HF_HUB_DISABLE_TELEMETRY=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN pip install "huggingface_hub==1.28.0"

# Only the files TEI needs: weights, model config, tokenizer and pooling config. The
# sentence_bert_config.json is ours: TEI reads max_seq_length from it, which pins the
# server's input bound to the engine's 2,048-token chunk budget (the model's own 32K window
# would otherwise force a 32K-token warm-up and allow inputs the engine never produces).
RUN hf download "$MODEL_REPO" \
      model.safetensors config.json tokenizer.json tokenizer_config.json \
      1_Pooling/config.json modules.json config_sentence_transformers.json \
      --revision "$MODEL_COMMIT" --local-dir /models/qwen3-embedding-0.6b \
    && rm -rf /models/qwen3-embedding-0.6b/.cache \
    && echo "$SAFETENSORS_SHA256  /models/qwen3-embedding-0.6b/model.safetensors" | sha256sum -c - \
    && echo '{"max_seq_length": 2048}' > /models/qwen3-embedding-0.6b/sentence_bert_config.json

FROM ghcr.io/huggingface/text-embeddings-inference:cpu-1.9.4@sha256:2538ea1c9640d3763b15af668039d24172d063b42337b0c27796fc2be180c78d

COPY --from=weights /models /models

# Weights are already present; never reach for the Hub at runtime.
ENV HF_HUB_OFFLINE=1 \
    HF_HOME=/tmp/hf

# text-embeddings-router is the base image's entrypoint; arguments come from tinfoil-config.yml.
