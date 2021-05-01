import torch.nn as nn


class RNNClassifier(nn.Module):

    def __init__(
        self,
        input_size,
        word_vec_size,
        hidden_size,
        n_classes,
        n_layers=4,
        dropout_p=.3,
    ):
        self.input_size = input_size  # vocabulary_size, 단어의 개수, corpus에 따라 달라짐, torchtext가 알아서 읽어옴, data_loader에 정의
        self.word_vec_size = word_vec_size  # 워드 임베딩 벡터 (임베딩 후 벡터의 크기)
        self.hidden_size = hidden_size # bidirection RNN의 hidden state와 cell state의 사이즈
        self.n_classes = n_classes  # output class의 개수
        self.n_layers = n_layers  # layer의 수는 보통 4개, gradient vanishing 고려
        self.dropout_p = dropout_p  # dropout 비율

        super().__init__()

        # 단순히 Linear Layer
        self.emb = nn.Embedding(input_size, word_vec_size)
        # RNN
        self.rnn = nn.LSTM(
            input_size=word_vec_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            dropout=dropout_p,
            batch_first=True,  # batch_size를 맨 처음에 넣어줌
            bidirectional=True, # non auto regressive
        )
        self.generator = nn.Linear(
            hidden_size * 2, n_classes)  # bidirection이니까 hidden_size * 2, classification전에 차원축소
        # We use LogSoftmax + NLLLoss instead of Softmax + CrossEntropy
        self.activation = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        # |x| = (batch_size, length, 1) # 1은 one-hot vector의 index
        x = self.emb(x)
        # |x| = (batch_size, length, word_vec_size)
        # LSTM은 마지막 timestep의 hiddin state(x), cell state(_)를 반환
        x, _ = self.rnn(x)
        # |x| = (batch_size, length, hidden_size * 2)
        # 슬라이싱, |x[:, -1]| = (batch_size, hidden_size * 2)
        y = self.activation(self.generator(x[:, -1]))
        # |y| = (batch_size, n_classes)

        return y
