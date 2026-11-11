"""NVIDIA's hosted Riva speech models (build.nvidia.com) over gRPC, with the existing NVIDIA key.

The key is only ever sent to ``grpc.nvcf.nvidia.com``. Function ids come from the model
pages on build.nvidia.com ("API" tab); the ones below were verified with the owner's key
on 2026-11-11.
"""

from __future__ import annotations

NVCF_SERVER = "grpc.nvcf.nvidia.com:443"

# name -> NVCF function id
ASR_FUNCTIONS = {
    "parakeet-ctc-1.1b": "1598d209-5e27-4d3c-8079-4751568b1081",
    "parakeet-tdt-0.6b-v2": "d3fe9151-442b-4204-a70d-5fcc597fd610",
}
TTS_FUNCTIONS = {
    "magpie-tts-multilingual": "877104f7-e885-42b9-8de8-f6e4c6303969",
}
DEFAULT_ASR_MODEL = "parakeet-ctc-1.1b"
DEFAULT_TTS_MODEL = "magpie-tts-multilingual"


def function_id(name_or_id: str, table: dict[str, str], default: str) -> str:
    """A known model name, or a raw function id (a UUID) pasted from build.nvidia.com."""
    wanted = (name_or_id or default).strip()
    return table.get(wanted.lower(), wanted)


def riva_auth(api_key: str, function: str):
    import riva.client

    return riva.client.Auth(None, True, NVCF_SERVER, [["function-id", function], ["authorization", f"Bearer {api_key}"]])
