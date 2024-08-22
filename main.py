#log dosyasını biçimlendirerek yeni bir log dosyası oluşturuldu


import re

# Dosya isimlerini buraya yazın
input_file = 'log_dosyasi.txt'
output_file = 'islenmis_loglar.txt'

def extract_info(line):
    # IP adresi ayıklama
    ip_match = re.match(r'^(\S+)', line)
    ip = ip_match.group(1) if ip_match else 'Unknown IP'

    # Tarih ve saat ayıklama
    date_match = re.search(r'\[([^]]+)\]', line)
    if date_match:
        date = date_match.group(1).split(' ')[0]
    else:
        date = 'Unknown Date'

    # Dosya adını ve uzantısı ayıklama
    file_name_match = re.search(r'GET /([^ ]+)', line)
    if file_name_match:
        file_path = file_name_match.group(1)
        file_extension_match = re.search(r'([^/]+\.[a-zA-Z0-9]+)$', file_path)
        file_extension = file_extension_match.group(1) if file_extension_match else 'Unknown File'
    else:
        file_extension = 'Unknown File'

    # Durum koduayıklama
    status_match = re.search(r' (\d{3}) ', line)
    status = status_match.group(1) if status_match else 'Unknown Status'

    # Dosya boyutu ayıklama
    size_match = re.search(r' (\d+)(?: "|\s|$)', line)
    size = size_match.group(1) if size_match else 'Unknown Size'

    # OS bilgisi ayıklama
    os_match = re.search(r'\"Mozilla\/[^"]* \(([^)]*)\)', line)
    if os_match:
        os_info = os_match.group(1)
        os_info = re.sub(r'\s*; \s*', '; ', os_info)  # Gereksiz boşlukları kaldır
        os_info = re.sub(r'\s+', ' ', os_info)  # Çoklu boşlukları tek bir boşlukla değiştir
        os = f"OS: {os_info}"
    else:
        os = 'OS: Unknown OS'

    # Satırı biçimlendir
    return f"IP: {ip} | Date: {date} | File Extension: {file_extension} | Status: {status} | Size: {size} | {os}"

def process_logs(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            if "GET" in line:
                processed_line = extract_info(line)
                outfile.write(processed_line + '\n')


process_logs(input_file, output_file)


##########################################################################################################
#VERİLERİ PINECONEA YÜKLEME İŞLEMİ YAPILDI


from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from pinecone import Pinecone, ServerlessSpec

# Log verilerini okuma fonksiyonu
def read_logs(file_path):
    with open(file_path, 'r') as file:
        data = file.read()
    return data

# Metni parçalara ayırma fonksiyonu
def split_into_chunks(text, chunk_size):
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    return chunks

# Vektörlere dönüştürme fonksiyonu
def text_to_vector(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze()
    # float32'ye dönüştür 
    return np.array(embeddings.float()).astype(np.float32).tolist()

# Soru metnini vektöre dönüştürme
def question_to_vector(question):
    return text_to_vector(question)

# Pinecone sorgulama fonksiyonu
def query_pinecone(query_vector, top_k=5):
    index = pinecone.Index(index_name)
    results = index.query(vector=query_vector, top_k=top_k)
    return results

# En iyi cevapları alma fonksiyonu
def get_best_answers(query):
    query_vector = question_to_vector(query)
    results = query_pinecone(query_vector)
    
    best_matches = results['matches']
    answers = []
    for match in best_matches:
        index_id = match['id']
        score = match['score']
        answers.append({
            'index_id': index_id,
            'score': score
        })
    
    return answers

# Log verilerini oku ve parçalara ayır
data = read_logs('islenmis_loglar.txt')
chunk_size = 200  # Her parçanın yaklaşık boyutu
chunks = split_into_chunks(data, chunk_size)

# Model ve tokenizerı yükle
model_name = 'bert-base-uncased' 
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Metinleri vektörlere dönüştür
vectors = [text_to_vector(chunk) for chunk in chunks]

# Pinecone API anahtarınızı ayarlayın
api_key = 'e0e15fc3-d15e-4270-b195-5be03b3bde02'
pinecone = Pinecone(api_key=api_key)

# İndeks adı ve boyutu
index_name = 'app'
dimension = len(vectors[0])  # Vektör boyutunu ayarlayın

# İndeksi oluşturun (eğer mevcut değilse)
if index_name not in pinecone.list_indexes().names():
    pinecone.create_index(
        name=index_name,
        dimension=dimension,
        metric='euclidean',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )

# Vektörleri Pinecone'a yükle
index = pinecone.Index(index_name)
for i, vector in enumerate(vectors):
    index.upsert(vectors=[(str(i), vector, {"chunk": chunks[i]})])

print("Vektörler başarıyla yüklendi.")



#####################################################################################3
#gpt2 dil modeli kullanarak verileri anlamlastırma

from transformers import GPT2Tokenizer, GPT2LMHeadModel


gpt_model_name = 'gpt2'
gpt_tokenizer = GPT2Tokenizer.from_pretrained(gpt_model_name)
gpt_model = GPT2LMHeadModel.from_pretrained(gpt_model_name)

def generate_answer_from_gpt(question, context):
    input_text = f"\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
    
    inputs = gpt_tokenizer.encode(input_text, return_tensors='pt')
    
    with torch.no_grad():
        outputs = gpt_model.generate(
            inputs,
            max_new_tokens=150,
            num_return_sequences=1,
            pad_token_id=gpt_tokenizer.eos_token_id,
            temperature=0.5,  # Daha az rastgelelik ve daha belirgin yanıtlar için
            top_p=0.9,        # Daha anlamlı yanıtlar için top-p sampling
            no_repeat_ngram_size=2,  # Yanıtların tekrarlanmaması için
        )
    
    answer = gpt_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

def generate_text_from_pinecone_results_with_gpt(answers, query):
    # En iyi 5 cevabı üretmesi sağlanır
    best_chunks = [chunks[int(answer['index_id'])] for answer in answers]
    context = " ".join(best_chunks)
    answer = generate_answer_from_gpt(query, context)
    return answer

query = "show me last used IP adress"  # kullanıcının sorusu

answers = get_best_answers(query)

# Pinecone'dan dönen sonuçları GPT ile işleyerek cevap üret
generated_answer = generate_text_from_pinecone_results_with_gpt(answers, query)

print("Üretilen Cevap:")
print(generated_answer)

