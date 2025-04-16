import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prepare_dataloader(data, batch_size=32):
    states = torch.stack([d[0] for d in data]).to(device)
    target_pis = torch.stack([torch.tensor(d[1]) for d in data]).to(device)
    target_vs = torch.tensor([d[2] for d in data],
                             dtype=torch.float32).to(device)

    dataset = TensorDataset(states, target_pis, target_vs)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def train(net, data, batch_size=32, epochs=5, lr=1e-3, return_metrics=False):
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    train_loader = prepare_dataloader(data, batch_size=batch_size)

    total_value_loss = 0
    total_policy_loss = 0
    total_loss = 0
    batches = 0

    for epoch in range(epochs):
        epoch_loss = 0
        epoch_value_loss = 0
        epoch_policy_loss = 0

        for batch in train_loader:
            states, target_pis, target_vs = batch

            log_pi_pred, v_pred = net(states)

            v_pred = v_pred.squeeze(1)
            value_loss = F.mse_loss(v_pred, target_vs)

            policy_loss = -torch.sum(target_pis * log_pi_pred) / states.size(0)
            loss = value_loss + policy_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_value_loss += value_loss.item()
            epoch_policy_loss += policy_loss.item()

            total_loss += loss.item()
            total_value_loss += value_loss.item()
            total_policy_loss += policy_loss.item()
            batches += 1

        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.4f}")

    if return_metrics:
        return {
            "avg_value_loss": total_value_loss / batches,
            "avg_policy_loss": total_policy_loss / batches,
            "avg_total_loss": total_loss / batches
        }
