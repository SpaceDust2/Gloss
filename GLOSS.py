from llama_cpp import Llama
import nltk
import csv
import string
nltk.download('punkt')

def split_text(text, max_tokens=1000):
    sentences = nltk.sent_tokenize(text)
    fragments = []
    current_fragment = ""
    
    for sentence in sentences:
        tokens = nltk.word_tokenize(sentence)
        if len(current_fragment) + len(tokens) <= max_tokens:
            current_fragment += sentence + " "
        else:
            fragments.append(current_fragment.strip())
            current_fragment = sentence + " "
    
    if current_fragment:
        fragments.append(current_fragment.strip())
    
    return fragments

def clean_terms(terms):
    cleaned_terms = set()
    for term in terms:
        # Исключаем пустые строки, строки с цифрами, символами и одной буквой
        if not term or any(char.isdigit() for char in term) or len(term) == 1 or any(char in string.punctuation for char in term):
            continue
        cleaned_terms.add(term)
    return cleaned_terms

# Открываем файл для чтения
with open('result6.txt', 'r', encoding='utf-8') as f:
    text = f.read()

fragments = split_text(text)

# Загружаем фрагменты по очереди в нейронную сеть
llm = Llama(model_path="speechless-llama2-hermes-orca-platypus-wizardlm-13b.Q2_K.gguf", n_ctx=4000, n_batch=256)

all_terms = set()

for i, current_fragment in enumerate(fragments, start=1):
    prompt = f"Извлеки и выведи перечень специальных терминов, употребленных в тексте. Необходимо предоставить только список терминов, без их определений. Текст: {current_fragment}"
    
    output = llm(
        prompt,
        max_tokens=1000,
        temperature=0.1,
        top_p=0.5,
        echo=False,
        stop=["#"],
    )
    output_text = output["choices"][0]["text"].strip()
    
    terms = output_text.split()
    cleaned_terms = clean_terms(terms)
    all_terms.update(cleaned_terms)

# Сохраняем все термины в файлы txt и csv
with open('all_terms.txt', 'w', encoding='utf-8') as terms_file:
    cleaned_terms = clean_terms(all_terms)
    terms_file.write("\n".join(cleaned_terms))

with open('terms.csv', 'w', encoding='utf-8', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['File', 'Term'])  # Заголовок файла CSV
    cleaned_terms = clean_terms(all_terms)
    for term in cleaned_terms:
        csv_writer.writerow(['audio6.mp3', term])
