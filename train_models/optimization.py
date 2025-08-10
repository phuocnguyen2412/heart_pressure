import torch
from matplotlib import pyplot as plt
import copy
from tqdm import tqdm

from setting import DEVICE


class Optimization:
    def __init__(self, model, config, loss_fn, optimizer, scheduler=None, patience=10, delta=0.0, save_dir='./'):
        self.model = model.to(DEVICE)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_losses = []
        self.val_losses = []
        self.patience = patience
        self.delta = delta
        self.config = config
        self.best_val_loss = float('inf')
        self.best_model_wts = copy.deepcopy(self.model.state_dict())
        self.early_stop_counter = 0
        self.best_epoch = 0
        self.save_dir = save_dir
        self.current_epoch = 0

    def validate(self, val_loader):
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc="Validating", leave=False):
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE).view(-1, 1)
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        return avg_val_loss

    def train(self, train_loader, val_loader, epochs=100, resume=False, resume_path=None):
        best_model_path = f"{self.save_dir}/best_model.pth"
        last_model_path = f"{self.save_dir}/last_model.pth"

        # Resume training if requested
        if resume and resume_path:
            self.load_checkpoint(resume_path)

        for epoch in range(self.current_epoch, epochs):
            self.model.train()
            epoch_loss = 0.0
            for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE).view(-1, 1)
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                epoch_loss += loss.item()
            avg_train_loss = epoch_loss / len(train_loader)
            self.train_losses.append(avg_train_loss)

            val_loss = self.validate(val_loader)
            self.val_losses.append(val_loss)

            # Early Stopping logic
            if val_loss < self.best_val_loss - self.delta:
                self.best_val_loss = val_loss
                self.best_model_wts = copy.deepcopy(self.model.state_dict())
                self.early_stop_counter = 0
                self.best_epoch = epoch + 1
            else:
                self.early_stop_counter += 1

            if self.scheduler:
                self.scheduler.step()

            print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}')

            # Early stopping triggered
            if self.early_stop_counter >= self.patience:
                print(f"Early stopping triggered at epoch {epoch + 1}. Best epoch: {self.best_epoch}, Best val loss: {self.best_val_loss:.4f}")
                break

            # Save best model (with config and extra info)
            self.save_model(best_model_path, is_best=True, current_epoch=epoch+1)

            self.current_epoch = epoch+1

        # Save last model (with config)
        self.save_model(last_model_path, is_best=False, current_epoch=self.current_epoch)

    def plot_losses(self):
        save_path = f"{self.save_dir}/losses_plot.png"
        plt.plot(self.train_losses, label="Training loss")
        plt.plot(self.val_losses, label="Validation loss")
        plt.legend()
        plt.title("Losses")
        plt.savefig(save_path)
        plt.close()

    def save_model(self, path, is_best=True, current_epoch=0):
        save_dict = {
            'model_state_dict': self.best_model_wts if is_best else self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'config': self.config,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'current_epoch': current_epoch,
            'best_val_loss': self.best_val_loss,
            'early_stop_counter': self.early_stop_counter,
            'best_epoch': self.best_epoch
        }
        torch.save(save_dict, path)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and checkpoint['scheduler_state_dict'] is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.config = checkpoint['config']
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.current_epoch = checkpoint.get('current_epoch', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.early_stop_counter = checkpoint.get('early_stop_counter', 0)
        self.best_epoch = checkpoint.get('best_epoch', 0)
        self.best_model_wts = copy.deepcopy(self.model.state_dict())
        print(f"Resumed from checkpoint at epoch {self.current_epoch}")
