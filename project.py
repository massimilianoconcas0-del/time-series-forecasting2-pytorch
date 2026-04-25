import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import yfinance as yf

# IMPORTA LA NOSTRA LIBRERIA
from relational_calculus.losses import RelationalMSELoss

print("All libraries loaded")

config = {
    "alpha_vantage": {
        "symbol": "IBM",
    },
    "data": {
        "window_size": 20,
        "train_split_size": 0.80,
    },
    "plots": {
        "xticks_interval": 90,
        "color_actual": "#001f3f",
        "color_train": "#3D9970",
        "color_val": "#0074D9",
        "color_pred_train": "#3D9970",
        "color_pred_val": "#0074D9",
        "color_pred_test": "#FF4136",
    },
    "model": {
        "input_size": 1,
        "num_lstm_layers": 2,
        "lstm_size": 32,
        "dropout": 0.2,
    },
    "training": {
        "device": "cpu",
        "batch_size": 64,
        "num_epoch": 100,
        "learning_rate": 0.01,
        "scheduler_step_size": 40,
    }
}

# 1. BYPASS API PREMIUM CON YFINANCE
def download_data(config):
    symbol = config["alpha_vantage"]["symbol"]
    print(f"Bypassing Alpha Vantage... Downloading {symbol} via yfinance")

    ticker = yf.Ticker(symbol)
    df = ticker.history(period="10y")
    df = df.reset_index()

    data_date = df['Date'].dt.strftime('%Y-%m-%d').tolist()
    data_close_price = df['Close'].tolist()

    num_data_points = len(data_close_price)
    display_date_range = f"{data_date[0]} to {data_date[-1]}"

    print(f"Successfully downloaded {num_data_points} data points.")
    return data_date, data_close_price, num_data_points, display_date_range

data_date, data_close_price, num_data_points, display_date_range = download_data(config)

# 2. ELIMINATO IL NORMALIZER (Usiamo i prezzi reali crudi)
raw_data_close_price = np.array(data_close_price)

def prepare_data_x(x, window_size):
    n_row = x.shape[0] - window_size + 1
    output = np.lib.stride_tricks.as_strided(x, shape=(n_row, window_size), strides=(x.strides[0], x.strides[0]))
    return output[:-1], output[-1]

def prepare_data_y(x, window_size):
    output = x[window_size:]
    return output

data_x, data_x_unseen = prepare_data_x(raw_data_close_price, window_size=config["data"]["window_size"])
data_y = prepare_data_y(raw_data_close_price, window_size=config["data"]["window_size"])

# split dataset
split_index = int(data_y.shape[0]*config["data"]["train_split_size"])
data_x_train = data_x[:split_index]
data_x_val = data_x[split_index:]
data_y_train = data_y[:split_index]
data_y_val = data_y[split_index:]

class TimeSeriesDataset(Dataset):
    def __init__(self, x, y):
        x = np.expand_dims(x, 2)
        self.x = x.astype(np.float32)
        self.y = y.astype(np.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return (self.x[idx], self.y[idx])

dataset_train = TimeSeriesDataset(data_x_train, data_y_train)
dataset_val = TimeSeriesDataset(data_x_val, data_y_val)

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=32, num_layers=2, output_size=1, dropout=0.2):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.linear_1 = nn.Linear(input_size, hidden_layer_size)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(hidden_layer_size, hidden_size=self.hidden_layer_size, num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(num_layers * hidden_layer_size, output_size)
        self.init_weights()

    def init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight_ih' in name:
                nn.init.kaiming_normal_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)

    def forward(self, x):
        batchsize = x.shape[0]
        x = self.linear_1(x)
        x = self.relu(x)
        lstm_out, (h_n, c_n) = self.lstm(x)
        x = h_n.permute(1, 0, 2).reshape(batchsize, -1)
        x = self.dropout(x)
        predictions = self.linear_2(x)
        return predictions[:, -1]

def run_epoch(dataloader, is_training=False):
    epoch_loss = 0
    if is_training:
        model.train()
    else:
        model.eval()

    for idx, (x, y) in enumerate(dataloader):
        if is_training:
            optimizer.zero_grad()

        batchsize = x.shape[0]
        x = x.to(config["training"]["device"])
        y = y.to(config["training"]["device"])

        out = model(x)

        # --- IL CUORE DEL CALCOLO RELAZIONALE ---
        # 1. Trova la "North Star": il prezzo massimo nei 20 giorni di storia per ogni sample
        capacity = torch.max(x, dim=1)[0].squeeze() + 1e-5

        # 2. Passa output(ratio), target assoluto e capacità alla loss
        loss = criterion(out.contiguous(), y.contiguous(), capacity.contiguous())

        if is_training:
            loss.backward()
            optimizer.step()

        epoch_loss += (loss.detach().item() / batchsize)

    lr = scheduler.get_last_lr()[0]
    return epoch_loss, lr

train_dataloader = DataLoader(dataset_train, batch_size=config["training"]["batch_size"], shuffle=True)
val_dataloader = DataLoader(dataset_val, batch_size=config["training"]["batch_size"], shuffle=True)

model = LSTMModel(input_size=config["model"]["input_size"], hidden_layer_size=config["model"]["lstm_size"],
                  num_layers=config["model"]["num_lstm_layers"], output_size=1, dropout=config["model"]["dropout"])
model = model.to(config["training"]["device"])

# 3. SETTATO SGD E LOSS RELAZIONALE
criterion = RelationalMSELoss()
optimizer = optim.SGD(model.parameters(), lr=config["training"]["learning_rate"])
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config["training"]["scheduler_step_size"], gamma=0.1)

for epoch in range(config["training"]["num_epoch"]):
    loss_train, lr_train = run_epoch(train_dataloader, is_training=True)
    loss_val, lr_val = run_epoch(val_dataloader)
    scheduler.step()
    print('Epoch[{}/{}] | loss train:{:.6f}, test:{:.6f} | lr:{:.6f}'
          .format(epoch + 1, config["training"]["num_epoch"], loss_train, loss_val, lr_train))

# ==========================================================
# INFERENZA CON DENORMALIZZAZIONE RELAZIONALE
# ==========================================================

train_dataloader = DataLoader(dataset_train, batch_size=config["training"]["batch_size"], shuffle=False)
val_dataloader = DataLoader(dataset_val, batch_size=config["training"]["batch_size"], shuffle=False)
model.eval()

predicted_train = np.array([])
for idx, (x, y) in enumerate(train_dataloader):
    x = x.to(config["training"]["device"])
    out_ratio = model(x)
    capacity = torch.max(x, dim=1)[0].squeeze()
    out_real_price = out_ratio * capacity # Riconversione matematica
    predicted_train = np.concatenate((predicted_train, out_real_price.cpu().detach().numpy()))

predicted_val = np.array([])
for idx, (x, y) in enumerate(val_dataloader):
    x = x.to(config["training"]["device"])
    out_ratio = model(x)
    capacity = torch.max(x, dim=1)[0].squeeze()
    out_real_price = out_ratio * capacity # Riconversione matematica
    predicted_val = np.concatenate((predicted_val, out_real_price.cpu().detach().numpy()))

# PLOT 1: Train/Val Predizioni Globali
to_plot_data_y_train_pred = np.zeros(num_data_points)
to_plot_data_y_val_pred = np.zeros(num_data_points)

to_plot_data_y_train_pred[config["data"]["window_size"]:split_index+config["data"]["window_size"]] = predicted_train
to_plot_data_y_val_pred[split_index+config["data"]["window_size"]:] = predicted_val

to_plot_data_y_train_pred = np.where(to_plot_data_y_train_pred == 0, None, to_plot_data_y_train_pred)
to_plot_data_y_val_pred = np.where(to_plot_data_y_val_pred == 0, None, to_plot_data_y_val_pred)

fig = figure(figsize=(25, 5), dpi=80)
fig.patch.set_facecolor((1.0, 1.0, 1.0))
plt.plot(data_date, data_close_price, label="Actual prices", color=config["plots"]["color_actual"])
plt.plot(data_date, to_plot_data_y_train_pred, label="Predicted prices (train)", color=config["plots"]["color_pred_train"])
plt.plot(data_date, to_plot_data_y_val_pred, label="Predicted prices (validation)", color=config["plots"]["color_pred_val"])
plt.title("Compare predicted prices to actual prices")
xticks = [data_date[i] if ((i%config["plots"]["xticks_interval"]==0 and (num_data_points-i) > config["plots"]["xticks_interval"]) or i==num_data_points-1) else None for i in range(num_data_points)]
x = np.arange(0,len(xticks))
plt.xticks(x, xticks, rotation='vertical')
plt.grid(visible=True, which='major', axis='y', linestyle='--')
plt.legend()
plt.show()

# PREDIZIONE DOMANI
model.eval()
x_unseen = torch.tensor(data_x_unseen).float().to(config["training"]["device"]).unsqueeze(0).unsqueeze(2)
prediction_ratio = model(x_unseen)
capacity_unseen = torch.max(x_unseen).item()
prediction_real = (prediction_ratio * capacity_unseen).cpu().detach().numpy()

plot_range = 10
to_plot_data_y_val = np.zeros(plot_range)
to_plot_data_y_val_pred = np.zeros(plot_range)
to_plot_data_y_test_pred = np.zeros(plot_range)

to_plot_data_y_val[:plot_range-1] = data_y_val[-plot_range+1:]
to_plot_data_y_val_pred[:plot_range-1] = predicted_val[-plot_range+1:]
to_plot_data_y_test_pred[plot_range-1] = prediction_real

to_plot_data_y_val = np.where(to_plot_data_y_val == 0, None, to_plot_data_y_val)
to_plot_data_y_val_pred = np.where(to_plot_data_y_val_pred == 0, None, to_plot_data_y_val_pred)
to_plot_data_y_test_pred = np.where(to_plot_data_y_test_pred == 0, None, to_plot_data_y_test_pred)

plot_date_test = data_date[-plot_range+1:]
plot_date_test.append("tomorrow")

fig = figure(figsize=(25, 5), dpi=80)
fig.patch.set_facecolor((1.0, 1.0, 1.0))
plt.plot(plot_date_test, to_plot_data_y_val, label="Actual prices", marker=".", markersize=10, color=config["plots"]["color_actual"])
plt.plot(plot_date_test, to_plot_data_y_val_pred, label="Past predicted prices", marker=".", markersize=10, color=config["plots"]["color_pred_val"])
plt.plot(plot_date_test, to_plot_data_y_test_pred, label="Predicted price for next day", marker=".", markersize=20, color=config["plots"]["color_pred_test"])
plt.title("Predicting the close price of the next trading day")
plt.grid(visible=True, which='major', axis='y', linestyle='--')
plt.legend()
plt.show()

print("Predicted close price of the next trading day:", round(to_plot_data_y_test_pred[plot_range-1], 2))
