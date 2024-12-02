!git clone https://github.com/aub-mind/arabert
!pip install pyarabic
!pip install farasapy

from transformers import AutoTokenizer, AutoModel
#from arabert.preprocess_arabert import never_split_tokens, preprocess
#from arabert.preprocess import ArabertPreprocessor, never_split_tokens
from arabert.preprocess import ArabertPreprocessor

from farasa.segmenter import FarasaSegmenter
import torch

arabert_tokenizer = AutoTokenizer.from_pretrained(
    "aubmindlab/bert-base-arabert",
    do_lower_case=False,
    do_basic_tokenize=True,
    never_split=True)
arabert_model = AutoModel.from_pretrained("aubmindlab/bert-base-arabert").to("cuda") #you can replace the path here with the folder containing the the pytorch model

farasa_segmenter = FarasaSegmenter(interactive=True)

from arabert.preprocess import ArabertPreprocessor

model_name = "bert-base-arabertv2"
arabert_prep = ArabertPreprocessor(model_name=model_name, keep_emojis=False)

text = "الغذاء هو أي مادة يتم تناولها أو شربها من قبل الكائنات الحية للحصول على الطاقة والعناصر الغذائية اللازمة للبقاء على قيد الحياة والنمو والتطور.  يمكن أن يكون الغذاء من أصل نباتي أو حيواني، ويمكن أن يكون معالجًا أو غير معالج. يكون الغذاء متنوعًا ويشمل العديد من العناصر الغذائية التي تشمل الكربوهيدرات والبروتينات والدهون والفيتامينات والمعادن. ما يلي بعض الأمثلة على الأطعمة التي تحتوي على هذه العناصر الغذائية: الطعام ، الطحين، الخبز، الأرز، المعكرونة، البطاطس، الفواكه، الخضروات، السبانخ، الفجل، الجزر، الجرجير، البندورة، الملفوف، البطاطا، اللحوم، الشاورما، الدواجن، الأسماك، البيض، البقوليات، المكسرات، البذور، شعير، القمح ، العدس، الزيوت، منتجات الألبان، الحليب، اللبن، الجبن، الشوربة، المرق، السلطات ، الذرة، الشراب ، العصير، الشاي، القهوة، الحلويات، الكعكة ، العسل، المربى."
text_preprocessed = arabert_prep.preprocess(text)
text_preprocessed

arabert_input = arabert_tokenizer.encode(text_preprocessed, add_special_tokens=True)
with open('output.txt', 'w', encoding='utf-8') as f:
    f.write(f"Arabert Input Tokens: {arabert_input}\n")
    f.write(f"Arabert Tokenized: {arabert_tokenizer.convert_ids_to_tokens(arabert_input)}\n")
    
tensor_input_ids = torch.tensor(arabert_input).unsqueeze(0).to("cuda")
f.write(f"Tensor Input IDs: {tensor_input_ids}\n")

output = arabert_model(tensor_input_ids)
f.write(f"Output Keys: {output.keys()}\n")

f.write(f"Output Shape: {output[0].shape}\n")

embeddings = output[0][0][1:-1]
f.write(f"Embeddings Shape: {embeddings.shape}\n")
f.write(f"Embeddings: {embeddings}\n")

# Calculate the average of all token vectors.
sum_vec = torch.mean(output[0][0][1:-1], dim=0)
f.write(f"Our final sentence embedding vector of shape: {sum_vec.size()}\n")
f.write(f"Final Embedding: {sum_vec}\n")

def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

with open("/content/FoodArabic.txt") as f_text:
    text = f_text.read()

text11 = text.split('\n\n')

for sent in text11:
    text_preprocessed11 = arabert_prep.preprocess(sent)
    arabert_input11 = arabert_tokenizer.encode(text_preprocessed11, add_special_tokens=True)
    tensor_input_ids11 = torch.tensor(arabert_input11).unsqueeze(0).to("cuda")
    output11 = arabert_model(tensor_input_ids11)
    sum_vec11 = torch.mean(output11[0][0][1:-1], dim=0)
    sim = cosine(sum_vec.detach().to("cpu").numpy(), sum_vec11.detach().to("cpu").numpy())
    
    if sim > 0.775:
        with open('output.txt', 'a', encoding='utf-8') as f:
            f.write(f"Sentence: {sent}; similarity = {sim}\n")
