import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from text2musiclang.decoder import DecoderTokenizer

hub_model_path = "musiclang/text2musiclang"
encoder_tokenizer_path = "bert-base-cased"
decoder_tokenizer_path = "musiclang/text2musiclang"

# Test loading the model and tokenizers
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model from Hub
loaded_model = AutoModelForSeq2SeqLM.from_pretrained(hub_model_path)
# Load encoder tokenizer
encoder_tokenizer = AutoTokenizer.from_pretrained(encoder_tokenizer_path)
decoder_tokenizer = AutoTokenizer.from_pretrained(decoder_tokenizer_path)
loaded_model.to(device)

# Your prompt here
test_input = "A sad and angry piano and violin piece with a jazzy rhythm. In 4/4 time signature and E minor."
input_ids = encoder_tokenizer.encode(test_input, return_tensors="pt").to(device)

output_ids = loaded_model.generate(
    input_ids,
    max_length=1024,
    do_sample=True,
    temperature=1.0,
    top_k=80,
    top_p=0.99,
    num_return_sequences=1,
)
output_text = decoder_tokenizer.decode(output_ids[0].tolist())
musiclang_tokenizer = DecoderTokenizer()
midi = musiclang_tokenizer.decode(output_text)
midi.dump_midi("generated_music.mid")