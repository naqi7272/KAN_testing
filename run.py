# IMPORTS
import os
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame as df
from lmfit import Model
import seaborn as sns
from scipy.optimize import curve_fit
import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from kan import *

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import argparse


parser = argparse.ArgumentParser(description="Process GRB Name as input.")
parser = argparse.ArgumentParser(description="Process GRB Name as input.")
parser.add_argument('--name', type=str, required=True, help="Specify the GRB Name")
parser.add_argument('--path', type=str, required=False, default="", help="Specify the path (optional)")
parser.add_argument('--norm', type=str, required=False, default="y")
parser.add_argument('--over', type=str, required=False, default="n")

args = parser.parse_args()

GRB_Name = str(args.name)

path = args.path

batch_size = 64

normalize = [True if args.norm == "y" else False][0]
override = [True if args.over == "y" else False][0]


# EPOCHS = 25000

if not override and os.path.exists(f"Saved_Outputs/{GRB_Name}"):
    print("PREDICTION ALREADY EXISTS, EXITING...")
    exit()

os.makedirs(f"Saved_Outputs/{GRB_Name}", exist_ok=True)

print(f"\n{GRB_Name}\n")
# PREPROCESSING
print("PREPROCESSING...\n")
header_names=['t', 'pos_t_err', 'neg_t_err', 'flux', 'pos_flux_err', 'neg_flux_err']
GRB_parameters = pd.read_csv("545_GRBs_parameters.csv", header=0, index_col=0)

trimmed_data = pd.read_csv(f"GRBs_trimmed/{GRB_Name}_trimmed.csv", skiprows=1, skip_blank_lines=True, sep=',', dtype=float, header=None, names=header_names)
trimmed_data=trimmed_data.sort_values(by="t").reset_index(drop=True)

density_factor=1

log_T_a = GRB_parameters.loc[GRB_Name, "logTa_best"]
log_T_a_min = GRB_parameters.loc[GRB_Name, "logTa_min"]
log_T_a_max = GRB_parameters.loc[GRB_Name, "logTa_max"]

log_F_a = GRB_parameters.loc[GRB_Name, "logFa"]
log_F_a_min = GRB_parameters.loc[GRB_Name, "logFa_min"]
log_F_a_max = GRB_parameters.loc[GRB_Name, "logFa_max"]

alpha = GRB_parameters.loc[GRB_Name, "alpha_best"]
alpha_min = GRB_parameters.loc[GRB_Name, "alpha_min"]
alpha_max = GRB_parameters.loc[GRB_Name, "alpha_max"]

log_Tt = GRB_parameters.loc[GRB_Name, "logTt"]
log_Tfinal = GRB_parameters.loc[GRB_Name, "logTfinal"]

max_fluxes = np.max(trimmed_data["flux"])
min_fluxes = np.min(trimmed_data["flux"])

max_ts = np.max(trimmed_data["t"])
min_ts = np.min(trimmed_data["t"])

log_max_fluxes = np.log10(max_fluxes)
log_min_fluxes = np.log10(min_fluxes)

log_max_ts = np.log10(max_ts)
log_min_ts = np.log10(min_ts)

positive_ts_err = trimmed_data["pos_t_err"]
negative_ts_err = trimmed_data["neg_t_err"]

positive_fluxes_err = trimmed_data["pos_flux_err"]
negative_fluxes_err = trimmed_data["neg_flux_err"]

ts, fluxes = trimmed_data["t"].to_numpy(), trimmed_data["flux"]
log_ts, log_fluxes = np.log10(ts), np.log10(fluxes)
# ts, fluxes = log_ts, log_fluxes

pos_fluxes= fluxes + positive_fluxes_err
neg_fluxes= fluxes + negative_fluxes_err

fluxes_err_sym= (pos_fluxes - neg_fluxes)/2

ts_error = (positive_ts_err - negative_ts_err )/2
fluxes_error = (positive_fluxes_err - negative_fluxes_err)/2

pos_log_fluxes = np.log10(pos_fluxes)
neg_log_fluxes = np.log10(neg_fluxes)

log_F_a_err = (log_F_a_max - log_F_a_min)/2
log_T_a_err=(log_T_a_max - log_T_a_min)/2
alpha_err = (alpha_max - alpha_min)/2

recon_t = np.geomspace(np.min(ts), np.max(ts), density_factor*len(ts))

o_log_recon_t = np.log10(recon_t)

# print(o_log_recon_t.shape)

gapslist=[]

for ff in range(0,len(log_ts)-1):
    lowbound=log_ts[ff]
    upbound=log_ts[ff+1]

    if np.abs(upbound-lowbound)>=0.03: #np.min(totalgaps): #0.10:
        
        gapslist.append([lowbound,upbound,np.abs(upbound-lowbound)])


pivotkeep=[]
for ii in range(len(o_log_recon_t)):
    for jj in range(len(gapslist)):
        if gapslist[jj][0]<=o_log_recon_t[ii]<=gapslist[jj][1]:
            pivotkeep.append(ii)


logtimekeep=[[o_log_recon_t[f]] for f in pivotkeep]

logtimekeeparray=np.squeeze(np.array(logtimekeep))

log_recon_t = logtimekeeparray



def Willingale_if(t, F_a, alpha, T_a):
    if t<T_a:
        return F_a * np.exp(alpha - (t*alpha)/T_a)
    else:
        return F_a * np.power((t / T_a),(-alpha))

def Willingale(t, F_a, alpha, T_a):
    y = np.zeros(t.shape)
    for j in range(len(y)):
        y[j]=Willingale_if(t[j], F_a, alpha, T_a)
    return y

def log_Willingale_if(logt, logFa, alpha, logTa):
    if logt<logTa:
        return logFa + np.log10(np.e) * alpha * (1.0 - 10**logt/(10**logTa))
    else:
        return logFa - alpha * (logt - logTa)

def log_Willingale(logt, logFa, alpha, logTa):
    y = np.zeros(logt.shape)
    for j in range(len(y)):
        y[j]=log_Willingale_if(logt[j], logFa, alpha, logTa)
    return y



# DATASET
print("PREPARING FOR MODEL...\n")

log_ts = log_ts.reshape(-1, 1)
log_fluxes = log_fluxes.values.reshape(-1, 1)
log_recon_t = np.reshape(log_recon_t, (-1, 1))

if normalize:
    s_log_ts = MinMaxScaler((0, 1)).fit(log_ts)
    log_ts = s_log_ts.transform(log_ts)

    s_log_fx = MinMaxScaler((0, 1)).fit(log_fluxes)
    log_fluxes = s_log_fx.transform(log_fluxes)

    s_log_recon_ts = MinMaxScaler((0, 1)).fit(log_recon_t)
    log_recon_ts = s_log_recon_ts.transform(log_recon_t)

else:
    log_recon_ts = log_recon_t


print("MODEL TRAINING...")

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import KFold


kf = KFold(n_splits=5)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# initialize KAN with G=3
X = torch.tensor(log_ts.reshape(-1, 1), dtype=torch.float32).to(device)
y = torch.tensor(log_fluxes.reshape(-1, 1), dtype=torch.float32).to(device)


model = KAN(width=[1, 5, 1], grid=5, k=9, seed=42).to(device)

epochs = 100
optimizer = Adam(model.parameters(), lr=1e-3)
patience_counter = 0
patience = 25
times = 5
pres = 5
count = 0
best_loss = float("inf")


# for epoch in tqdm(range(epochs)):
#     model.train()
#     optimizer.zero_grad()
#     train_loss = torch.nn.functional.mse_loss(model(X), y)
#     train_loss.backward()
#     optimizer.step()

#     if round(train_loss.item(), pres) < best_loss:
#         best_loss = round(train_loss.item(), pres)
#         count = 0
#     else:
#         count += 1

#     if count >= patience:
#         break

class Data(Dataset):
    def __init__(self, time: np.ndarray, flux: np.ndarray):
        # self.time = torch.tensor(time, dtype=torch.float32).view(-1, 1)
        # self.flux = torch.tensor(flux, dtype=torch.float32).view(-1, 1)
        self.time = time
        self.flux = flux

    def __len__(self):
        return len(self.time)
    
    def __getitem__(self, idx):
        return self.time[idx], self.flux[idx]

epoch = 0

kf_train_mse_losses, kf_test_mse_losses = [], []

with open(f"Saved_Outputs/{GRB_Name}/logs.txt", "w") as file:
    file.write(f"{GRB_Name}\n")

pbar = tqdm(total=epochs)
best_mse = float("inf")
while True:
    epoch += 1

    for i, (train_index, test_index) in enumerate(kf.split(X, y)):
        train_loader = DataLoader(Data(X[train_index], y[train_index]), batch_size=batch_size, shuffle=False)

        train_mse_loss = 0
        for time, flux in train_loader:
            # time = torch.reshape(time, (-1, 1, 1)).to("cuda")
            # flux = torch.reshape(flux, (-1, 1, 1)).to("cuda")

            predictions = model(time)
            loss = torch.nn.functional.mse_loss(flux, predictions)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_mse_loss += loss.item()

        avg_train_mse_loss = train_mse_loss 
        with open(f"Saved_Outputs/{GRB_Name}/logs.txt", "a+") as file:
            file.write(f"MSE: {avg_train_mse_loss:.4f} | ")

        kf_train_mse_losses.append(avg_train_mse_loss)


        test_loader = DataLoader(Data(X[test_index], y[test_index]), batch_size=batch_size, shuffle=False)

        test_mse_loss = 0
        for time, flux in test_loader:
            # time = torch.reshape(time, (-1, 1, 1)).to("cuda")
            # flux = torch.reshape(flux, (-1, 1, 1)).to("cuda")

            predictions = model(time)
            loss = torch.nn.functional.mse_loss(flux, predictions)

            test_mse_loss += loss.item()

        avg_test_mse_loss = test_mse_loss 
        with open(f"Saved_Outputs/{GRB_Name}/logs.txt", "a+") as file:
            file.write(f"MSE: {avg_test_mse_loss:.4f}\n\n")
        kf_test_mse_losses.append(avg_test_mse_loss)

    avg_test_loss = sum([kf_test_mse_losses[d] for d in range(-kf.get_n_splits(), 0, 1)])/kf.get_n_splits()
    pbar.update(1)
    pbar.set_description(f"Test Loss: {round(avg_test_loss, 4)}, Patience: {patience_counter}, Times: {times}")

    if round(avg_test_loss, 4) < best_mse:
        best_mse = round(avg_test_loss, 4)
        patience_counter = 0 
    else:
        patience_counter += 1
    
    if patience_counter >= patience:
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_test_loss:.4f}")
        print(f"Early stopping at epoch {epoch+1} due to no improvement in validation loss")
        break


kf_train_mse_losses = np.array(kf_train_mse_losses).reshape(-1, kf.get_n_splits())
kf_test_mse_losses = np.array(kf_test_mse_losses).reshape(-1, kf.get_n_splits())

np.save(f"Saved_Outputs/{GRB_Name}/train-mse.npy", kf_train_mse_losses)
np.save(f"Saved_Outputs/{GRB_Name}/test-mse.npy", kf_test_mse_losses)

    
print(f"Early stopping at epoch {epoch + 1}")

log_recon_ts = torch.tensor(log_recon_ts.reshape(-1, 1), dtype=torch.float32).to(device)
log_recon_fx = model(log_recon_ts).detach().cpu()
log_recon_fx = np.reshape(log_recon_fx, (-1, 1))

if normalize:
    log_ts = s_log_ts.inverse_transform(log_ts)
    log_fluxes = s_log_fx.inverse_transform(log_fluxes)

    log_recon_t = s_log_recon_ts.inverse_transform(log_recon_ts.detach().cpu())
    log_recon_fx = s_log_fx.inverse_transform(log_recon_fx)

log_fluxes = np.squeeze(log_fluxes)
log_recon_t = np.squeeze(log_recon_t)
recon_fluxes_up = np.squeeze(log_recon_fx)

## JIGGLED POINTS
print("JIGGLING POINTS...\n")
fluxes_error = (positive_fluxes_err - negative_fluxes_err)/2
logfluxerrs = fluxes_error/(fluxes*np.log(10))


errparameters = st.norm.fit(logfluxerrs) #GAUSSIAN FITTING ON ERROR-BAR DISTRIBUTION
err_dist = st.norm(loc=errparameters[0], scale=errparameters[1])

recon_errorbar=np.abs(err_dist.rvs(size=len(log_recon_t)))

#Point specific noise
point_specific_noise = []
for j in range(len(recon_fluxes_up)):
    fitted_dist = st.norm(loc=recon_fluxes_up[j], scale=recon_errorbar[j])
    point_noise = fitted_dist.rvs() - recon_fluxes_up[j]
    point_specific_noise.append(point_noise)

point_specific_noise = np.array(point_specific_noise)

#Jiggle reconstructed points
jiggled_points = recon_fluxes_up + point_specific_noise
jiggled_points = np.squeeze(jiggled_points)



## REALIZATIONS 
print("CALCULAING CONFIDENCE INTERVAL...\n")
# num_samples = 1000  # Number of realizations
# jiggled_realizations = []

# for _ in tqdm(range(num_samples)):
#     point_specific_noise = []
#     for j in range(len(recon_fluxes_up)):
#         fitted_dist = st.norm(loc=recon_fluxes_up[j], scale=recon_errorbar[j])
#         point_noise = fitted_dist.rvs() - recon_fluxes_up[j]
#         point_specific_noise.append(point_noise)
#     jiggled_realizations.append(recon_fluxes_up + np.array(point_specific_noise))

# jiggled_realizations = np.array(jiggled_realizations)

# # Compute mean and 95% confidence intervals
# mean_jiggled = np.mean(jiggled_realizations, axis=0)
# ci_95_lower = np.percentile(jiggled_realizations, 2.5, axis=0)  # 2.5th percentile
# ci_95_upper = np.percentile(jiggled_realizations, 97.5, axis=0)  # 97.5th percentile

num_samples = 1000  # Number of realizations
recon_fluxes_up = np.array(recon_fluxes_up)  
recon_errorbar = np.array(recon_errorbar)

point_specific_noise = np.random.normal(
    loc=0, scale=recon_errorbar, size=(num_samples, len(recon_fluxes_up))
)
jiggled_realizations = recon_fluxes_up + point_specific_noise
jiggled_realizations = jiggled_realizations

mean_jiggled = np.mean(jiggled_realizations, axis=0)
ci_95_lower = np.percentile(jiggled_realizations, 2.5, axis=0)  # 2.5th percentile
ci_95_upper = np.percentile(jiggled_realizations, 97.5, axis=0)  # 97.5th percentile


## PLOTTING
print("SAVING PLOT...\n")
plt.errorbar(log_ts, log_fluxes, yerr=[log_fluxes-neg_log_fluxes,pos_log_fluxes-log_fluxes], label=r"$\log_{10}\,flux$", linestyle="", zorder=4)
plt.errorbar(log_recon_t, jiggled_points, linestyle='none', yerr=np.abs(recon_errorbar), marker='o', capsize=5, color='yellow', zorder=3)

plt.scatter(log_ts, log_fluxes, label='Observations', zorder=5)
plt.plot(log_recon_t, recon_fluxes_up, label='Mean predictions', zorder=2)
plt.fill_between(log_recon_t, np.squeeze(ci_95_lower), np.squeeze(ci_95_upper), color='orange', alpha=0.5, label='95% Confidence Interval', zorder=1)

plt.xlabel('log Time', fontsize=18) 
plt.ylabel('log Flux', fontsize=18) 
plt.title(f'KAN on {GRB_Name}', fontsize=18) 

plt.tick_params(axis='both', labelsize=14)

plt.legend(fontsize=14)
plt.savefig(f"Saved_Outputs/{GRB_Name}/{GRB_Name}.png", dpi=300, bbox_inches='tight')


#CALCULATING TIME ERROR IN LINEAR SCALE
print("SAVING DATAFRAME...")
ts_error = (positive_ts_err - negative_ts_err)/2

#CALCULATING TIME ERROR IN LOG SCALE
log_ts_error = ts_error/(ts*np.log(10))

errparameters = st.norm.fit(log_ts_error) #GAUSSIAN FITTING ON TIME ERROR DISTRIBUTION
err_dist_time = st.norm(loc=errparameters[0], scale=errparameters[1])

recon_logtimeerr=err_dist_time.rvs(size=len(log_recon_t)) # len(log_ts_error)
df = trimmed_data.copy(deep=True)

for k in range(0, len(log_recon_t)):
    new_row = {
        "t": 10**log_recon_t[k],  
        "pos_t_err": 10**recon_logtimeerr[k],
        "neg_t_err": 10**recon_logtimeerr[k],
        "flux": 10**jiggled_points[k],
        "pos_flux_err": 10**jiggled_points[k] * np.log(10) * recon_errorbar[k],
        "neg_flux_err": 10**jiggled_points[k] * np.log(10) * recon_errorbar[k]
    }
    
    new_row_df = pd.DataFrame([new_row])
    df = pd.concat([df, new_row_df], ignore_index=True)


df.to_csv(f"Saved_Outputs/{GRB_Name}/{GRB_Name}.csv")
