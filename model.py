#import
import torch
import pandas as pd
import re
import gluonnlp as nlp
from kobert_tokenizer import KoBERTTokenizer
from transformers import BertModel
from torch.utils.data import Dataset
from torch import nn
import numpy as np
from collections import Counter


#GPU든 CPU 사용
device = "cuda" if torch.cuda.is_available() else "cpu"
# BERT 모델을 불러오기.
bertmodel = BertModel.from_pretrained('skt/kobert-base-v1')
# BERT 모델을 위한 토크나이저를 불러오기
tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
# BERT 분류기 모델을 정의
class BERTClassifier(nn.Module):
    def __init__(self, bert, hidden_size=768, num_classes=6, dr_rate=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate
        self.classifier = nn.Linear(hidden_size, num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = output['pooler_output']
        if self.dr_rate:
            pooled_output = self.dropout(pooled_output)
        return self.classifier(pooled_output)
      
# 모델 불러오기
model = BERTClassifier(bertmodel).to(device)
model.load_state_dict(torch.load('/model_state_dict.pt'))
model.to(device)

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
      

#평가 모델 parameter
max_len = 512
batch_size = 32
warmup_ratio = 0.1
num_epochs = 5
max_grad_norm = 1
log_interval = 100
learning_rate = 5e-5


# 평가 함수
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
        probabilities = F.softmax(result, dim=1)[0]
        top_k_prob, top_k_idx = torch.topk(probabilities, k=top_k)
        
        # 감정 카운트
        emotion_counts = Counter([cate[idx.item()] for idx in top_k_idx])

        return emotion_counts


#테스트 데이터 불러오기
testData = pd.read_csv('/content/drive/MyDrive/lg/song2022.csv')

# 모든 노래의 감정을 합산하여 가장 많이 등장한 감정 2개 선택
all_emotion_counts = Counter()
for artist, songname, lyric in zip(testData['artist'], testData['song_name'], testData['lyric']):
    emotion_counts = testModel(artist, songname, lyric, top_k=2)
    all_emotion_counts += emotion_counts

# 가장 많이 등장한 감정 2개 선택
most_common_emotions = all_emotion_counts.most_common(2)
most_common_emotions_str = ', '.join([emotion for emotion, _ in most_common_emotions])

print("현재의 감정:", most_common_emotions_str)
