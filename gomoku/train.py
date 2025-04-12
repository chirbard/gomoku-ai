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


def train(net, data, batch_size=32, epochs=5, lr=1e-3):
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    train_loader = prepare_dataloader(data, batch_size=batch_size)

    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            states, target_pis, target_vs = batch

            log_pi_pred, v_pred = net(states)

            v_pred = v_pred.squeeze(1)
            value_loss = F.mse_loss(v_pred, target_vs)
            # Normalize by batch size
            policy_loss = -torch.sum(target_pis * log_pi_pred) / states.size(0)
            loss = value_loss + policy_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss:.4f}")
