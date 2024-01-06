import pytest

from text_generation_server.models.types import Generation, Tokens


def test_generation_to_ob():
    prefill_tokens = Tokens(
        token_ids=[10367, 4040, 632, 101190, 530, 473],
        logprobs=[float("nan"), -5.98, -0.12, -8.11, -3.66, -0.45],
        texts=["My", "name", "is", "Oliver", "and", "I"],
        is_special=[]
    )
    tokens = Tokens(token_ids=[912], logprobs=[0.], texts=[' am'], is_special=[False])
    top_tokens = Tokens(token_ids=[912, 1542, 20152, 2909, 1620],
                        logprobs=[-0.55, -2.61, -2.91, -2.96, -3.79],
                        texts=[' am', ' have', ' live', ' work', ' was'],
                        is_special=[False, False, False, False, False])

    try:
        Generation(request_id=1, prefill_tokens=prefill_tokens, tokens=tokens, generated_text=None,
                   top_tokens=top_tokens).to_pb()
    except TypeError as e:
        pytest.fail(f"Unexpected type error when converting 'Generation' to protobuf: {e}")
