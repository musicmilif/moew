import math
import torch
from torch import nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        return torch.mean(F_loss) if self.reduce else F_loss
    
def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class GELU(nn.Module):
    def __init__(self):
        super(GELU, self).__init__()
    
    def forward(self, inputs):
        return gelu(inputs)


def weights_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight.data, nn.init.calculate_gain('relu'))
        m.bias.data.zero_()


class TargetAutoEncoder(nn.Module):
    def __init__(self, input_dim, alpha_size):
        super(TargetAutoEncoder, self).__init__()
        self.input_dim = input_dim+1
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.input_dim//2),
            GELU(),
            nn.Linear(self.input_dim//2, alpha_size),
            GELU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(alpha_size, self.input_dim//2),
            GELU(),
            nn.Linear(self.input_dim//2, self.input_dim),
        )

    def forward(self, x):
        emb = self.encoder(x)
        x = self.decoder(emb)
        x, y = x[:, :-1], torch.sigmoid(x[:, -1])
        
        return x, y, emb.detach()

