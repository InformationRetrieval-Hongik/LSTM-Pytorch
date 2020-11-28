import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self,hidden_size,bidirectional = False,score_function = 'dot'):
        super(Attention, self).__init__()

        self.hidden_size = hidden_size
        self.softmax = nn.Softmax()
        self.score_function = score_function

        #if score_function == 'general':

        if bidirectional: # Is lstm bidirectional or not
          self.direction = 2
        else:
          self.direction = 1

    def forward(self,lstm_out,hidden):
        #lstm output shape [batch, seq_length, num_direction * hidden_size]
        #lstm hidden shape [num_direction, batch, hidden_size]
        seq_length = lstm_out.shape[1]
        hidden = hidden.view(-1,self.direction*self.hidden_size) #make hidden shape : [batch, num_direction * hidden_size]

        #calculate Attention score
        #score(st,hi) = st.T * hi

        if self.score_function == 'dot':
            score = torch.bmm(lstm_out,hidden.unsqueeze(2))
        elif self.score_function == 'scaled_dot':
            score = torch.bmm(lstm_out,hidden.unsqueeze(2))/seq_length
        

        at = self.softmax(score) # et : [st*h1,....,st*hN], at : softmax(et) [batch, seq_length,1]

        attention = torch.bmm(at.transpose(1,2),lstm_out) # attention : [batch,1,num_direction * hidden_size]
        return attention.squeeze(1) #return : [batch,num_direction * hidden_size]

        #return torch.cat((hidden,attention.squeeze(1)),1)

#Attention 코드는 아래 주소의 수식과 설명을 보고 구현하였으며
#decoder 용 끝단을 classification에 맞추기 위해 concat이 아닌 attention만 반환합니다.
#Attention 메커니즘 출처 https://wikidocs.net/22893


class BILSTM(nn.Module):
    def __init__(self, vocab_size=1002, embedding_dim=10, vector_len=80, unit_num=128):
        super(BILSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)  # output shape: [batch size, vector len, embedding_dim]
        self.lstm1 = nn.LSTM(embedding_dim, unit_num,bidirectional=True, batch_first=True,dropout = 0.35)  # output shape: [batch size, vector len, unit num]

        self.flatten = nn.Flatten()
        self.linear = nn.Linear(unit_num*2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embedding_vec = self.embedding(x)
        out1, (h_n, c_n) = self.lstm1(embedding_vec)
        out = self.sigmoid(self.linear(out1[:,-1,:]))

        return out

class BILSTM_withAttention(nn.Module):
    def __init__(self, vocab_size=1002, embedding_dim=10, vector_len=80, unit_num=128):
        super(BILSTM_withAttention, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)  # output shape: [batch size, vector len, embedding_dim]
        self.lstm1 = nn.LSTM(embedding_dim, unit_num,bidirectional=True, batch_first=True,dropout = 0.35)  # output shape: [batch size, vector len, unit num]

        self.atten = Attention(unit_num,bidirectional=True,score_function = 'scaled_dot')

        self.flatten = nn.Flatten()
        self.linear = nn.Linear(unit_num*2, 1)
        self.sigmoid = nn.Sigmoid()
        

    def forward(self, x):
        embedding_vec = self.embedding(x)
        out1, (h_n, c_n) = self.lstm1(embedding_vec)
        
        att = self.atten(out1,h_n).unsqueeze(1)

        out = self.sigmoid(self.linear(out1[:,-1,:]))

        return out

class BILSTM_withAttention2layer(nn.Module):
    def __init__(self, vocab_size=1002, embedding_dim=10, vector_len=80, unit_num=128):
        super(BILSTM_withAttention2layer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)  # output shape: [batch size, vector len, embedding_dim]
        self.lstm1 = nn.LSTM(embedding_dim, unit_num,bidirectional=True, batch_first=True,dropout = 0.35)  # output shape: [batch size, vector len, unit num]
        
        self.lstm2 = nn.LSTM(256, unit_num,bidirectional=True, batch_first=True,dropout = 0.35)
        self.atten1 = Attention(unit_num,bidirectional=True,score_function = 'scaled_dot')
        self.atten2 = Attention(unit_num,bidirectional=True,score_function = 'scaled_dot')

        self.flatten = nn.Flatten()
        self.linear = nn.Linear(unit_num*2, 1)
        self.sigmoid = nn.Sigmoid()
        

    def forward(self, x):
        embedding_vec = self.embedding(x)
        out1, (h_n, c_n) = self.lstm1(embedding_vec)
        
        att = self.atten1(out1,h_n).unsqueeze(1)

        for i in range(out1.shape[1]-2,-1,-1):
          att = torch.cat((self.atten1(out1,out1[:,i,:]).unsqueeze(1),att),1)
        
        out2, (h_n,c_n) = self.lstm2(att)

        att = self.atten2(out2,h_n)
        out = self.sigmoid(self.linear(att))

        return out

if __name__ == "__main__":
    
