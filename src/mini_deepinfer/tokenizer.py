"""UTF-8 bytes 0..255, BOS=256, EOS=257; no downloads or vocabulary files."""


class ByteTokenizer:
    bos_id = 256
    eos_id = 257

    def encode(self, text):
        if not isinstance(text, str):
            raise TypeError("prompt must be a string")
        return [self.bos_id, *text.encode("utf-8")]

    def decode(self, ids):
        return bytes(i for i in ids if 0 <= i < 256).decode("utf-8", errors="replace")
