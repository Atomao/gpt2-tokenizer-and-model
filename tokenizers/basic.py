class BasicTokenizer:
    def __init__(self) -> None:
        self.base_vocab_size = 256
        self.merges = {idx: idx for idx in range(self.base_vocab_size)}
        self.vocab = None

    def _text_to_simple_tokens(self, text: str):
        tokens = text.encode("utf-8")
        tokens = list(map(int, tokens))
        return tokens
    
    def _simple_tokens_to_text(self, tokens):
        return b"".join([bytes([token]) for token in tokens]).decode("utf-8", errors="allow")

    def _get_stats(self, tokens: list[int]) -> dict[tuple[int, int], int]:
        stats = {}
        for pair in zip(tokens, tokens[1:]):
            stats[pair] = stats.get(pair, 0) + 1
        return stats
    

    def _merge(self, tokens, max_pair, max_idx):
        idx = 0
        new_tokens = []
        while idx + 1 < len(tokens):
            pair = tokens[idx: idx + 2]
            if tuple(pair) == max_pair:
                new_tokens.append(max_idx)
                idx += 1
            else:
                new_tokens.append(pair[0])
            idx += 1
        return new_tokens

    def train(self, text: str, vocab_size: int, verbose: bool = False):
        tokens = self._text_to_simple_tokens(text)
        idx = len(self.merges)
        while idx < vocab_size and len(text) > 2:
            stats = self._get_stats(tokens)
            max_pair = max(stats, key=stats.get)
            if verbose:
                print(f"Merging pair ({max_pair}) into an index {idx}")
            self.merges[idx] = tuple(max_pair)
            tokens = self._merge(tokens, max_pair, idx)
            idx += 1
        self.vocab = {idx: bytes([idx]) for idx in range(self.base_vocab_size)}
        for i in range(self.base_vocab_size, len(self.merges)):
            first, second = self.merges[i]
            self.vocab[i] = self.vocab[first] + self.vocab[second]
        return tokens
        
    def encode(self, text: str):
        tokens = self._text_to_simple_tokens(text)
        for token in range(self.base_vocab_size, len(self.merges)):
            tokens = self._merge(tokens, self.merges[token], token)
        return tokens
            

    def mydecode(self, tokens: list[int]):
        idx = len(self.merges) - 1
        while idx > self.base_vocab_size - 1:
            first, second = self.merges[idx]
            for token_idx, token in enumerate(tokens):
                if token == idx:
                    tokens[token_idx] = first
                    tokens.insert(token_idx + 1, second)
            idx -= 1
        text = self._simple_tokens_to_text(tokens)
        return text
    
    def decode(self, tokens: list[int]):
        return b"".join(map(self.vocab.get, tokens)).decode("utf-8", errors="allow")
        
