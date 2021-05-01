import torch
import torch.nn as nn


class CNNClassifier(nn.Module):

    def __init__(
        self,
        input_size,
        word_vec_size,
        n_classes,
        use_batch_norm=False,
        dropout_p=.5,
        window_sizes=[3, 4, 5],
        n_filters=[100, 100, 100],
    ):
        # rnn은 batch_norm을 못쓰지만 cnn은 쓸수있음
        # window_sizes는 몇단어 짜리 classifier인지
        # n_filters는 몇개의 패턴인지

        self.input_size = input_size  # vocabulary_size, 단어의 개수, corpus에 따라 달라짐, torchtext가 알아서 읽어옴, data_loader에 정의
        self.word_vec_size = word_vec_size # 워드 임베딩 벡터 (임베딩 후 벡터의 크기)
        self.n_classes = n_classes # output class의 개수
        self.use_batch_norm = use_batch_norm # dropout을 쓸지 batch_norm을 쓸지
        self.dropout_p = dropout_p
        # window_size means that how many words a pattern covers.
        self.window_sizes = window_sizes # 몇개 단어의 패던을 detect할지
        # n_filters means that how many patterns to cover.
        self.n_filters = n_filters # 단어 패턴을 몇개를 detect할지

        super().__init__()

        self.emb = nn.Embedding(input_size, word_vec_size)
        # Use nn.ModuleList to register each sub-simple_ntc.
        self.feature_extractors = nn.ModuleList()  # cnn을 만들어서 집어 넣는거, 필터 수 별로, 모듈을 담는 리스트
        # cnn 모듈을 만들어서 넣는 것
        for window_size, n_filter in zip(window_sizes, n_filters): # window_sizes와 n_filters로 sub_module을 나눠서 집어넣음
            self.feature_extractors.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=1,  # input channel, We only use one embedding layer. 단어당 한가지의 word embedding vector
                        # 저자의 논문에서는 여러개의 워드임베딩을 넣어주면 좋다고 나와있지만, 현재는 단일 레이어를 통화시키는 것이 대세
                        # 그리고 보통 vision task에서는 rgb 3개의 channel을 사용함
                        out_channels=n_filter,
                        kernel_size=(window_size, word_vec_size),
                    ),
                    nn.ReLU(),
                    nn.BatchNorm2d(
                        n_filter) if use_batch_norm else nn.Dropout(dropout_p),
                )
            )

        # max-polling은 forward에서 동적으로 시행

        # An input of generator layer is max values from each filter.
        self.generator = nn.Linear(sum(n_filters), n_classes)
        # We use LogSoftmax + NLLLoss instead of Softmax + CrossEntropy
        self.activation = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        # |x| = (batch_size, length)
        x = self.emb(x)
        # |x| = (batch_size, length, word_vec_size)
        min_length = max(self.window_sizes) # input되는 문장의 길이가 window_size보다 작은 경우는 padding을 추가
        if min_length > x.size(1):
            # Because some input does not long enough for maximum length of window size,
            # we add zero tensor for padding.
            pad = x.new(x.size(0), min_length - x.size(1),
                        self.word_vec_size).zero_()
            # |pad| = (batch_size, min_length - length, word_vec_size)
            x = torch.cat([x, pad], dim=1)
            # |x| = (batch_size, min_length, word_vec_size)

        # In ordinary case of vision task, you may have 3 channels on tensor, (rgb class)
        # but in this case, you would have just 1 channel,
        # which is added by 'unsqueeze' method in below:
        x = x.unsqueeze(1)
        # |x| = (batch_size, 1, length, word_vec_size)

        cnn_outs = []
        for block in self.feature_extractors:
            cnn_out = block(x)
            # |cnn_out| = (batch_size, n_filter, length - window_size + 1, 1)

            # In case of max pooling, we does not know the pooling size,
            # because it depends on the length of the sentence.
            # Therefore, we use instant function using 'nn.functional' package.
            # This is the beauty of PyTorch. :)
            # max polling은 학습이 아님, 따라서 함수로 선언해서 사용.   
            cnn_out = nn.functional.max_pool1d(
                input=cnn_out.squeeze(-1),
                kernel_size=cnn_out.size(-2)
            ).squeeze(-1)
            # |cnn_out| = (batch_size, n_filter)
            cnn_outs += [cnn_out]
        # Merge output tensors from each convolution layer.
        cnn_outs = torch.cat(cnn_outs, dim=-1) # 하나의 vector로 만들어 줌
        # |cnn_outs| = (batch_size, sum(n_filters))
        y = self.activation(self.generator(cnn_outs))
        # |y| = (batch_size, n_classes)

        return y
