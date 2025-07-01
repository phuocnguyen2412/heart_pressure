import os.path

from torch import nn, optim
from torch.utils.data import DataLoader

from dataset import PPGDataset, train_files
from optimization import Optimization
from setting import BASE_DIR, DEVICE


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # Sequential block of layer1
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 128, kernel_size=9, stride=1, padding=0),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3))

        self.layer2 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=9, stride=1, padding=0),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3))

        self.adaptive = nn.AdaptiveMaxPool1d(4)


        # lstm and fully connected layer
        self.lstm = nn.LSTM(input_size=256, hidden_size=56, batch_first=True)
        self.fc1 = nn.Linear(4*56, 512)
        self.fc2 = nn.Linear(512, 1)


    def forward(self, x):
        # x: (batch_size, 875)
        x = x.unsqueeze(1)  # (batch_size, 1, 875)
        out = self.layer1(x)          # (batch_size, 128, L1)
        out = self.layer2(out)        # (batch_size, 256, L2)
        out = self.adaptive(out)      # (batch_size, 256, 4)

        # LSTM expects (batch, seq_len, input_size)
        out = out.permute(0, 2, 1)    # (batch_size, 4, 256)
        out, _ = self.lstm(out)       # (batch_size, 4, 56)
        out = out.reshape(out.size(0), -1)  # (batch_size, 4*56=224)

        out = self.fc1(out)           # (batch_size, 512)
        out = self.fc2(out)           # (batch_size, output_dim)
        return out

learning_rate = 0.001
batch_size = 2048
if __name__ == '__main__':
    print("Using device:", DEVICE)
    train_dataset_sbp = PPGDataset(train_files, label='SBP')

    print("Shape of PPG data:", train_dataset_sbp[0][0].shape)
    print("Shape of SBP label:", train_dataset_sbp[0][1].shape)
    val_dataset_sbp = PPGDataset(train_files, label='SBP')

    train_dataloader = DataLoader(train_dataset_sbp, batch_size=batch_size, shuffle=True, num_workers=8)
    val_dataloader = DataLoader(val_dataset_sbp, batch_size=batch_size, shuffle=False, num_workers=8)

    print("Shape of PPG data in train dataloader:", next(iter(train_dataloader))[0].shape)
    print("Shape of SBP label in train dataloader:", next(iter(train_dataloader))[1].shape)

    model = ConvNet()
    print("Model number of parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))


    loss_fn = nn.MSELoss(reduction='mean')
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    save_dir = os.path.join(BASE_DIR, "trained_model", "cnn-lstm")
    os.makedirs(save_dir, exist_ok=True)
    opt = Optimization(model, config=None, loss_fn=loss_fn, optimizer=optimizer, patience=10, delta=0.0, save_dir=save_dir)
    opt.load_checkpoint(os.path.join(save_dir, "best_model.pth"))  # Load the best model if exists
    opt.train(train_loader=train_dataloader, val_loader=val_dataloader, epochs=100, resume=False, resume_path=None)
    opt.plot_losses()
