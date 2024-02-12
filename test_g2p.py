# from time import time
# from transformers import T5ForConditionalGeneration, AutoTokenizer

# model = T5ForConditionalGeneration.from_pretrained(
#     "charsiu/g2p_multilingual_byT5_tiny_8_layers_100"
# )
# model.to_bettertransformer()
# model = model.to("cpu")
# tokenizer = AutoTokenizer.from_pretrained("google/byt5-small")

# # tokenized English words
# words = ["Char", "siu", "is", "a", "Cantonese", "style", "of", "barbecued", "pork"]
# words = ["<eng-us>: " + i.lower() for i in words]

# start = time()
# out = tokenizer(words, padding=True, add_special_tokens=False, return_tensors="pt")

# preds = model.generate(
#     **out, num_beams=1, max_length=50
# )  # We do not find beam search helpful. Greedy decoding is enough.
# phones = tokenizer.batch_decode(preds.tolist(), skip_special_tokens=True)
# print(time() - start)

# start = time()
# out = tokenizer(words, padding=True, add_special_tokens=False, return_tensors="pt")

# preds = model.generate(
#     **out, num_beams=1, max_length=50
# )  # We do not find beam search helpful. Greedy decoding is enough.
# phones = tokenizer.batch_decode(preds.tolist(), skip_special_tokens=True)
# print(time() - start)
# print(phones)
