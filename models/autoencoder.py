from Non_Homophily_Large_Scale.models import *
import torch.nn.init as init

class Encoder(nn.Module):
    def __init__(self, in_channel, out_channel=4, hidden_channel=32) -> None:
        super(Encoder, self).__init__()
        self.fc = nn.Linear(in_features=in_channel, out_features=out_channel)
        
        self.fc1 = nn.Linear(in_features=in_channel, out_features=hidden_channel)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_features=hidden_channel, out_features=out_channel)
        
        self.in_channel = in_channel
        self.out_channel = out_channel
        
    def forward(self,x):
        x = self.fc(x)
        # x = self.relu(x)
        # x = self.fc1(x)
        # x = self.relu(x)
        # x = self.fc2(x)
        return x
    
    def restart(self):
        init.xavier_uniform_(self.fc.weight)
        init.xavier_uniform_(self.fc1.weight)
        init.xavier_uniform_(self.fc2.weight)
        
        if self.fc.bias is not None:
            init.zeros_(self.fc.bias)
        if self.fc1.bias is not None:
            init.zeros_(self.fc1.bias)
        if self.fc2.bias is not None:
            init.zeros_(self.fc2.bias)


class Decoder(nn.Module):
    def __init__(self, out_channel, in_channal=4, hidden_channel=32) -> None:
        super(Decoder, self).__init__()
        self.fc = nn.Linear(in_features=in_channal, out_features=out_channel)
        
        self.fc1 = nn.Linear(in_features=in_channal, out_features=hidden_channel)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_features=hidden_channel, out_features=out_channel)
        self.in_channel = in_channal
        self.out_channel = out_channel
    
    def forward(self,x):
        x = self.fc(x)
        # x = self.relu(x)
        # x = self.fc1(x)
        # x = self.relu(x)
        # x = self.fc2(x)
        return x 

    def restart(self):
        init.xavier_uniform_(self.fc.weight)
        init.xavier_uniform_(self.fc1.weight)
        init.xavier_uniform_(self.fc2.weight)
        
        if self.fc.bias is not None:
            init.zeros_(self.fc.bias)
        if self.fc1.bias is not None:
            init.zeros_(self.fc1.bias)
        if self.fc2.bias is not None:
            init.zeros_(self.fc2.bias)
