from pathlib import Path


def test_modal_defaults_to_sglang_and_documents_rollback_image() -> None:
    env_example = Path(".env.example").read_text()

    assert "EOSIN_MODAL_INFERENCE_BACKEND=sglang" in env_example
    assert "EOSIN_MODAL_VLLM_IMAGE=vllm/vllm-openai:v0.19.0-ubuntu2404" in env_example
    assert "EOSIN_MODAL_SGLANG_IMAGE=lmsysorg/sglang:v0.5.10" in env_example


def test_modal_supports_isolated_sglang_benchmarks() -> None:
    modal_app = Path("modal_app.py").read_text()

    assert 'INFERENCE_BACKEND = os.getenv("EOSIN_MODAL_INFERENCE_BACKEND", "sglang")' in modal_app
    assert 'SGLANG_IMAGE = os.getenv("EOSIN_MODAL_SGLANG_IMAGE", "lmsysorg/sglang:v0.5.10")' in modal_app
    assert '"--speculative-algorithm",' in modal_app
    assert '"NEXTN",' in modal_app
    assert '"SGLANG_ENABLE_SPEC_V2": "1"' in modal_app
    assert '"EOSIN_MODAL_INFERENCE_BACKEND": INFERENCE_BACKEND' in modal_app
