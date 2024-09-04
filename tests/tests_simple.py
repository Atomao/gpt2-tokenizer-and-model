from tokenizers.basic import BasicTokenizer


def test_hello_world():
    text = "hello world, my name is Danylo"
    tokenizer = BasicTokenizer()
    tokenizer.train(text, vocab_size=257)
    assert tokenizer.decode(tokenizer.encode(text)) == text