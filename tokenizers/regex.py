from tokenizers import BasicTokenizer
import regex

GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

class RegexTokenizer(BasicTokenizer):
    def __init__(self, regex_pattern=GPT4_SPLIT_PATTERN) -> None:
        super().__init__()
        self.regex_pattern = regex_pattern
        self.pat = regex.compile(self.regex_pattern)


    def _text_to_simple_tokens(self, text: str):
        chunks: str = self.pat.findall(text)
        tokens = []
        for chunk in chunks:
            tokens_chunk = list(map(int, chunk.encode("utf-8")))
            tokens.extend(tokens_chunk)
        return tokens


