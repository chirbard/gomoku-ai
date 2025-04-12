import torch
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(net, data, epochs=5, lr=1e-3):
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0
        for state, target_pi, target_v in data:
            state = state.unsqueeze(0).to(device)  # Move to GPU
            target_pi = torch.tensor(target_pi).unsqueeze(
                0).to(device)  # Move to GPU
            target_v = torch.tensor([[target_v]], dtype=torch.float32).to(
                device)  # Move to GPU

            log_pi_pred, v_pred = net(state)

            # Loss = policy loss + value loss
            value_loss = F.mse_loss(v_pred, target_v)
            policy_loss = -torch.sum(target_pi * log_pi_pred)
            loss = value_loss + policy_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss:.4f}")
