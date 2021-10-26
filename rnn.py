import torch
import torch.nn as nn
import torch.optim as optim

from gen_gbu import gen_gbu


def prepare_data(X, Y):
    vocabulary = sorted(set([word for sequence in X for word in sequence]))
    one_hot_len = len(vocabulary)
    one_hot_vectors = dict()
    for i in range(one_hot_len):
        one_hot_vector = [0 for _ in range(one_hot_len)]
        one_hot_vector[i] = 1
        one_hot_vectors[vocabulary[i]] = one_hot_vector
    X = [
        torch.Tensor([
            one_hot_vectors[word]
            for word in sequence
        ])
        for sequence in X
    ]
    Y = torch.Tensor(Y)
    return X, Y


class LinearActivation(nn.Module):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input


class MyRNN(nn.Module):
    """Simple RNN."""
    def __init__(self, input_size, message_size, output_size=1, **kwargs):
        """Initialize."""
        super().__init__(**kwargs)
        self.to_message = nn.Sequential(
            nn.Linear(input_size + message_size, message_size, bias=False),
            LinearActivation(),
        )
        self.to_output = nn.Sequential(
            nn.Linear(input_size + message_size, output_size, bias=False),
            LinearActivation(),
        )
        # self.to_message.weight.data = torch.Tensor([[-1, 1, 0, 1]])

    def forward(self, sequence):
        """Run forward."""
        message = torch.zeros(self.to_message[0].out_features)
        outputs = []
        for token in sequence:
            _in = torch.cat((token, message), axis=0)
            message = self.to_message(_in)
            outputs.append(self.to_output(_in))
        return message, outputs


def main():
    """Run experiment."""
    N = 200
    X_train, Y_train = gen_gbu(N)
    X_train, Y_train = prepare_data(X_train, Y_train)

    # initialize the network
    message_size = 1
    input_size = len(X_train[0][0])
    net = MyRNN(input_size, message_size, 1)

    # Create loss function and specify optimizer
    criterion = nn.MSELoss(reduction="sum")
    LR = 0.03
    optimizer = optim.Adam(net.parameters(), lr=LR, betas=(0.9, 0.99))

    # train the model
    for i in range(100):
        net.train()
        optimizer.zero_grad()
        for sequence, label in zip(X_train, Y_train):
            message, _ = net(sequence)
            loss = criterion(message, label)
            loss.backward()
        optimizer.step()

        net.eval()
        with torch.no_grad():
            loss_total = 0
            for sequence, label in zip(X_train, Y_train):
                message, _ = net(sequence)
                loss = criterion(message, label)
                loss_total += loss.item()
        print("epoch", i)
        print("### loss: {}".format(loss_total/N))
    print(net.to_message[0].weight)


if __name__ == "__main__":
    main()
    