import torch

from winnow import calibration
from winnow.calibration import calibration_batches


class _FakeStream:
    def __init__(self, rows):
        self.rows = rows

    def select_columns(self, _columns):
        return self

    def shuffle(self, seed, buffer_size):
        return self

    def __iter__(self):
        return iter(self.rows)


class _FakeTokenizer:
    """Tokenize each 'word' to one id; chat rendering joins message contents."""

    eos_token_id = 0

    def __call__(self, texts, add_special_tokens):
        assert add_special_tokens is False
        return {"input_ids": [[hash(word) % 61 + 1 for word in text.split()] for text in texts]}

    def apply_chat_template(self, conversations, tokenize):
        assert tokenize is False
        return [
            " ".join(message["content"] for message in conversation) + " <eos>"
            for conversation in conversations
        ]


def _install_stream(monkeypatch, rows):
    monkeypatch.setattr(calibration, "load_dataset", lambda **kwargs: _FakeStream(rows))


def test_plain_text_batches(monkeypatch):
    _install_stream(monkeypatch, [{"text": "a b c d"} for _ in range(8)])
    batches = list(calibration_batches(_FakeTokenizer(), "fake", sequences=3, sequence_length=4))
    stacked = torch.cat(batches)
    assert stacked.shape == (3, 4)
    # Documents are EOS-separated: every fifth token is the separator.
    assert (stacked.flatten() == 0).sum() == 2


def test_chat_template_renders_messages(monkeypatch):
    rows = [
        {
            "messages": [
                {"role": "user", "content": "hi there"},
                {"role": "assistant", "content": "hello"},
            ]
        }
        for _ in range(8)
    ]
    _install_stream(monkeypatch, rows)
    tokenizer = _FakeTokenizer()
    eos_word_id = hash("<eos>") % 61 + 1
    batches = list(
        calibration_batches(
            tokenizer,
            "fake",
            text_field="messages",
            chat_template=True,
            sequences=2,
            sequence_length=4,
        )
    )
    stacked = torch.cat(batches)
    assert stacked.shape == (2, 4)
    # The rendered template's own terminator survives; since it does not equal
    # eos_token_id, the packer still adds exactly one separator after it.
    flat = stacked.flatten().tolist()
    assert eos_word_id in flat
    assert 0 in flat


def test_chat_template_does_not_double_eos(monkeypatch):
    class _EosTokenizer(_FakeTokenizer):
        def __call__(self, texts, add_special_tokens):
            # Every rendered conversation tokenizes to ids ending in EOS.
            return {"input_ids": [[7, 8, self.eos_token_id] for _ in texts]}

    _install_stream(monkeypatch, [{"messages": []} for _ in range(8)])
    batches = list(
        calibration_batches(
            _EosTokenizer(),
            "fake",
            text_field="messages",
            chat_template=True,
            sequences=2,
            sequence_length=3,
        )
    )
    flat = torch.cat(batches).flatten().tolist()
    assert flat == [7, 8, 0, 7, 8, 0]
