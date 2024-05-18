import torch
import pandas as pd
import re
from transformers import BertModel, BertTokenizer
from torch.utils.data import Dataset
from torch import nn
import numpy as np
from collections import Counter
from kobert_tokenizer import KoBERTTokenizer

# GPU든 CPU 사용
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# BERT 모델을 불러오기.
model = BertModel.from_pretrained('skt/kobert-base-v1')
# BERT 모델을 위한 토크나이저를 불러오기
tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')

# 미리 학습된 감정 분류기 모델을 불러오기
class BERTClassifier(nn.Module):
    def __init__(self, bert, num_classes=6):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.classifier = nn.Linear(bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = output.pooler_output
        logits = self.classifier(pooled_output)
        return logits

# 모델 불러오기
model_path = './model_state_dict.pt'
classifier_model = BERTClassifier(model).to(device)
classifier_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
classifier_model.eval()

# 데이터셋을 불러오고 학습모델에 맞게 클래스를 정의.
class BERTDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.sentences = [i[1] for i in dataset]
        self.labels = [np.int32(i[0]) for i in dataset]

    def __getitem__(self, i):
        sentence = self.sentences[i]
        label = self.labels[i]
        encoding = self.tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

    def __len__(self):
        return len(self.labels)
      
max_len = 512

# 예측 함수
def testModel(a, s, ly, top_k=2):
    cate = ['기쁨', '분노', '상처', '슬픔', '불안', '당황']
    ly = re.sub('[a-zA-z]', '', ly) # 영문 제거
    ly = re.sub("\n", ' ', ly) #빈칸 제거
    ly = re.sub('[\{\}\[\]\/?.,;:|\)*~`!^\-_+<>@\#$%&\\\=\(\'\"]', '', ly) #특수문자 제거

    # 토큰화
    encoding = tokenizer.encode_plus(
        ly,
        add_special_tokens=True,
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    model.eval()
    with torch.no_grad():
        result = model(encoding['input_ids'].to(device), encoding['attention_mask'].to(device))
        probabilities = nn.functional.softmax(result, dim=1)[0]
        top_k_prob, top_k_idx = torch.topk(probabilities, k=top_k)
        print(f"{a}의 {s} 감정:")
        for idx in top_k_idx:
            emotion = cate[idx]
            print(f"    - {emotion}")

testData = pd.read_csv('song2022.csv')


for artist, songname, lyric in zip(testData['artist'], testData['song_name'], testData['lyric']):
    testModel(artist, songname, lyric, top_k=2)