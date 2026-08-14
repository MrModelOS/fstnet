from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors
import os, time

CORPUS = "data/local/corpus.txt"
VOCAB_SIZE = 32768
OUT = "tokenizer/fst_bpe.json"

t0 = time.time()
print("Loading corpus...")
files = [os.path.join(dp, f)
         for dp, _, fns in os.walk("data/local")
         for f in fns if f.endswith((".txt",))]

tok = Tokenizer(models.BPE(unk_token="<unk>"))
tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True, use_regex=True)
tok.decoder = decoders.ByteLevel()

trainer = trainers.BpeTrainer(
    vocab_size=VOCAB_SIZE,
    special_tokens=["<pad>", "<unk>", "<bos>", "<eos>", "<s>", "</s>"],
    min_frequency=2,
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    show_progress=True,
)

print("Training BPE...")
tok.train(files, trainer)
tok.save(OUT)

# Add post-processor (bos/eos)
tok = Tokenizer.from_file(OUT)
tok.post_processor = processors.TemplateProcessing(
    single="<bos> $A <eos>",
    special_tokens=[("<bos>", tok.token_to_id("<bos>")), ("<eos>", tok.token_to_id("<eos>"))],
)
tok.save(OUT)

print(f"Vocab: {tok.get_vocab_size()} tokens")
print(f"Saved: {OUT} ({time.time()-t0:.1f}s)")

# Test
test = "def binary_search(arr, target):\n    low, high = 0, len(arr) - 1\n    while low <= high:"
ids = tok.encode(test).ids
print(f"Test encode: {len(ids)} tokens for {len(test)} chars")
print("Sample:", tok.decode(ids[:30]))
