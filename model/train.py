import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model.c_mamba import CMamba, ModelArgs
import matplotlib.pyplot as plt
import wandb

# Define the training function
def train(args):
    # Initialize a new W&B run
    wandb.init(project=args.project_name)

    # Save model configuration
    wandb.config.update(vars(args))

    # Initialize the model, loss function, and optimizer
    model = CMamba(args)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Tracking lists for loss values
    train_losses = []
    val_losses = []

    # Training loop
    for epoch in range(args.num_epochs):
        model.train()
        running_loss = 0.0
        print(f"\nEpoch [{epoch+1}/{args.num_epochs}]")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']}")

        for i, (X_batch, y_batch) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(X_batch)
            
            # No need to permute y_batch as it's already in the correct shape
            loss = criterion(output, y_batch)
            loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Calculate and log gradient norms and histograms
            grad_norms = {}
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm(2).item()
                    grad_norms[f"grad_norm_{name}"] = grad_norm
                    # Log gradient histograms
                    wandb.log({f"grad_hist_{name}": wandb.Histogram(param.grad.cpu().numpy())})
            
            wandb.log(grad_norms)
            
            optimizer.step()
            
            running_loss += loss.item()
            
            if i % 10 == 9:
                print(f"Step [{i+1}/{len(train_loader)}], Batch Loss: {loss.item():.4f}")
                # Log batch loss to WandB
                wandb.log({"batch_loss": loss.item(), "epoch": epoch + 1, "step": i + 1})

        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        print(f"Epoch [{epoch+1}/{args.num_epochs}], Average Training Loss: {epoch_loss:.4f}")
        
        # Log epoch training loss to WandB
        wandb.log({"train_loss": epoch_loss, "epoch": epoch + 1})

        # Log weight statistics
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    wandb.log({f"weight_mean_{name}": param.mean().item(),
                               f"weight_std_{name}": param.std().item(),
                               f"weight_hist_{name}": wandb.Histogram(param.cpu().numpy())})
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for j, (X_batch, y_batch) in enumerate(test_loader):
                output = model(X_batch)
                
                loss = criterion(output, y_batch)
                val_loss += loss.item()
                
                if j % 10 == 9:
                    print(f"Validation Batch [{j+1}], Batch Loss: {loss.item():.4f}")
                    # Log validation batch loss to WandB
                    wandb.log({"val_batch_loss": loss.item(), "epoch": epoch + 1, "step": j + 1})
        
        val_loss /= len(test_loader)
        val_losses.append(val_loss)
        print(f"Epoch [{epoch+1}/{args.num_epochs}], Validation Loss: {val_loss:.4f}")
        
        # Log epoch validation loss to WandB
        wandb.log({"val_loss": val_loss, "epoch": epoch + 1})

        if (epoch + 1) % 5 == 0:
            model_path = f'c_mamba_model_epoch_{epoch+1}.pth'
            torch.save(model.state_dict(), model_path)
            wandb.save(model_path)

    # Plotting training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, args.num_epochs + 1), train_losses, label='Training Loss')
    plt.plot(range(1, args.num_epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.show()

    # Save final model
    final_model_path = 'c_mamba_model_final.pth'
    torch.save(model.state_dict(), final_model_path)
    wandb.save(final_model_path)
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script for the C-Mamba model.")
    parser.add_argument('--train_dataset', type=str, default="cMamba-project", help='The training dataset.')
    parser.add_argument('--test_dataset', type=str, default="cMamba-project", help='The test dataset.')
    parser.add_argument('--project_name', type=str, default="cMamba-project", help='WandB project name.')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer.')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs for training.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training.')
    parser.add_argument('--seq_len', type=int, required=True, help='Length of the input sequences.')
    parser.add_argument('--forecast_len', type=int, required=True, help='Length of the forecast sequences.')
    parser.add_argument('--input_dim', type=int, required=True, help='Input dimension for the model.')
    parser.add_argument('--hidden_dim', type=int, required=True, help='Hidden dimension for the model.')
    parser.add_argument('--num_layers', type=int, required=True, help='Number of layers in the model.')

    args = parser.parse_args()

    train_loader = DataLoader(args.train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(args.test_dataset, batch_size=args.batch_size)

    model_args = ModelArgs(
        seq_len=args.seq_len,
        forecast_len=args.forecast_len,
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers
    )

    train(model_args)
