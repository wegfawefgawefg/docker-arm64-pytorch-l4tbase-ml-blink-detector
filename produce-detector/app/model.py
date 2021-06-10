import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL = "weights.ptm"
IM_SCALE = (28, 28)
STATE_SHAPE = (1, *IM_SCALE)
NUM_CLASSES = 4
CLASS_NAMES = ["coli", "bell", "apple", "orange"]

OUTPUT_SHAPES = {
    "stride_1, 28x28": 15488,
    "stride_1, 64*64": 107648,
    "stride_1, 128*128": 476288,
    "stride_1, 512*512": 2000000,
    "stride_1, 1024x1024": 8193152,
    "stride_2, 1024x1024": 2064512,
}

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_channels = STATE_SHAPE[0]
        self.conv_out_shape = OUTPUT_SHAPES["stride_1, 28x28"]
        self.fc1Dims = 64
        self.fc2Dims = 64

        self.conv = nn.Sequential(
            nn.Conv2d(self.num_channels, 8, 3, stride=1), nn.ReLU(),
            nn.Conv2d(8, 16, 3, stride=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=1), nn.ReLU(),
        )
        self.classify = nn.Sequential(
            nn.Linear( self.conv_out_shape,  self.fc1Dims), nn.ReLU(),
            nn.Linear( self.fc1Dims, self.fc2Dims), nn.ReLU(),
            nn.Linear( self.fc2Dims, NUM_CLASSES ))

        self.locate = nn.Sequential(
            nn.Linear( self.conv_out_shape,  self.fc1Dims), nn.ReLU(),
            nn.Linear( self.fc1Dims, self.fc2Dims), nn.ReLU(),
            nn.Linear( self.fc2Dims, 4 ))

    def forward(self, x):
        x = self.conv(x)
        x = x.flatten().unsqueeze(0)
        # print(x.shape)
        # quit()
        c = self.classify(x)
        s = F.softmax(c, dim=1)
        m = torch.argmax(s, dim=1)

        loc = self.locate(x)
        return m, loc

if __name__ == "__main__":
    im = np.ones(IM_SCALE)
    im = torch.tensor(im, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    net = ConvNet().float()
    torch.save(net.state_dict(), MODEL)
    # net.load_state_dict(torch.load(MODEL))
    net.eval()
    with torch.no_grad():
        cls, loc = net(im)
        cls = cls.item()
        loc = loc.tolist()[0]
        loc = [max(min(l, 1.0), 0.0) for l in loc]

    print(cls)
    print(loc)
