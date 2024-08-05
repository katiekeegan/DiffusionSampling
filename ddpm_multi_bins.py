from __future__ import print_function

import argparse
import io

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader

torch.autograd.set_detect_anomaly(True)


from scipy.stats import wasserstein_distance

from models import *
from sampler import *


def compute_prob_distribution(data):
    """Compute probability distribution of the data."""
    counts = torch.histc(data, bins=100, min=data.min().detach(), max=data.max().detach())
    prob_dist = counts / counts.sum()
    return prob_dist


def kl_divergence(p, q):
    """Compute KL divergence between two distributions."""
    return (p * torch.log(p / q)).sum()


def jensen_shannon_divergence(p, q):
    """Compute Jensen-Shannon Divergence."""
    m = 0.5 * (p + q)
    return 0.5 * (kl_divergence(p, m) + kl_divergence(q, m))


def compute_jsd(dataset1, dataset2):
    js_divergence = 0
    for j in range(dataset1.size(0)):
        for i in range(dataset1.size(-1)):
            # Convert datasets to PyTorch tensors with gradient computation enabled
            dataset1_tensor = dataset1[j,...,i].ravel()
            dataset2_tensor = dataset2[j,...,i].ravel()

            # Compute probability distributions
            prob_dist1 = compute_prob_distribution(dataset1_tensor) + 1e-10
            prob_dist2 = compute_prob_distribution(dataset2_tensor) + 1e-10

            # Compute Jensen-Shannon Divergence
            js_divergence += jensen_shannon_divergence(prob_dist1, prob_dist2)

    return js_divergence

def unique_loss(data):
    """
    Returns unique elements from the input tensor within a specified tolerance.

    Args:
        tensor (torch.Tensor): Input tensor.
        tolerance (float): Tolerance within which two elements are considered equal.

    Returns:
        torch.Tensor: A tensor containing the unique elements within tolerance.
    """
    unique_loss_total = 0
    rounded_data = torch.round(data, decimals=4)
    for i in range(0,data.size(0)):
        rounded_data_i = data[i,:]
        unique_elements, counts = torch.unique(rounded_data_i, return_counts=True)
        repeated_counts = counts[counts > 1]
        unique_loss = torch.sum(repeated_counts - 1)
        unique_loss_total += unique_loss
    return torch.log(unique_loss_total)

def data_to_dist(samples, num_bins=100):
    hist = torch.histc(samples, bins=1000)
    probabilities = hist / torch.sum(hist)
    return probabilities

def hist_loss(eps, eps_theta, num_bins=100):
    errors = 0
    loss = torch.nn.MSELoss()
    eps_hist = eps.squeeze()
    eps_theta_hist = eps_theta.squeeze()
    for i in range(0,eps.size(0)):
        for j in range(0,eps_theta.size(-1)):
            errors = errors + loss(data_to_dist(eps_hist[i,:,j]),data_to_dist(eps_theta_hist[i,:,j]))
    return torch.tensor(errors, requires_grad=True)

def interval_loss(tensor, lower_bound=0, upper_bound=1):
    """
    Custom loss function that penalizes values outside a specified range.

    Args:
        tensor (torch.Tensor): Input tensor.
        lower_bound (float): Lower bound of the acceptable range.
        upper_bound (float): Upper bound of the acceptable range.
        penalty_weight (float): Weighting factor for the penalty term.

    Returns:
        torch.Tensor: Loss value.
    """
    penalty = torch.mean(torch.square(torch.max(torch.tensor(0.0), tensor - 1)) + torch.square(torch.max(torch.tensor(0.0), -tensor)))
    return penalty

def criterion(eps, eps_theta):  # Loss function
    print(eps.size())
    print(eps_theta.size())
    loss = torch.nn.L1Loss()
    # eps = eps.squeeze()
    # eps_theta = eps_theta.squeeze()
    # indices = eps[:,:,-1]
    # indices_theta = eps_theta[:, :, -1]
    # data = eps[:,:,0:-1]
    # data_theta = eps[:, :, 0:-1]
    # # # Calculate uniqueness loss
    # rounded_eps_theta = torch.round(data_theta , decimals=5)
    # unique_elements, counts = torch.unique(rounded_eps_theta, return_counts=True)
    # repeated_counts = counts[counts > 1]  # Get counts of repeated values only
    # unique_loss = torch.log(torch.sum(repeated_counts - 1))  # Penalize by the number of repetitions minus 1
    # interval_loss = torch.log(out_of_range_penalty_loss(rounded_eps_theta))
    return loss(eps, eps_theta) # +  0.1*loss(indices, indices_theta) #+ 0.01*unique_loss + 0.001*interval_loss# + torch.nn.L1Loss()(eps,eps_theta)

def criterion_sorted(eps, eps_theta):  # Loss function
    print(eps.size())
    print(eps_theta.size())
    loss = torch.nn.MSELoss()
    eps_sorted, indices = torch.sort(eps, dim=-2)
    eps_theta_sorted, indices = torch.sort(eps_theta, dim=-2)
    # eps = eps.squeeze()
    # eps_theta = eps_theta.squeeze()
    # losses_total = 0
    # for i in range(0,eps.size(-1)-1):
    #     eps_i, indices = torch.sort(eps[...,i], dim=-1)
    #     eps_theta_i, indices = torch.sort(eps_theta[..., i], dim=-1)
    #     losses_total += loss(eps_i, eps_theta_i)
    return loss(eps_sorted,eps_theta_sorted) # +  0.1*loss(indices, indices_theta) #+ 0.01*unique_loss + 0.001*interval_loss# + torch.nn.L1Loss()(eps,eps_theta)

def enforce_constraints(outputs, t, value = 3):
    # Clamp to range [1, 1024]
    indices = (t == 1).nonzero(as_tuple=True)[0]
    for i in range(0,len(indices)):
        outputs_i = outputs[indices[i],:]
        clamped_outputs_i = torch.clamp(outputs_i, min=0, max=1)
        outputs_i[outputs_i != clamped_outputs_i] = value
        outputs[indices[i],:] = outputs_i
    return outputs

def train_loop(
    frames,
    epoch,
    model,
    optimizer,
    data_loader,
    alphabars,
    device,
    losses,
    losses_sorted,
    losses_xT,
    losses_over_epochs,
    args,
    alphas
):
    losses_over_epoch = []
    T = args.T
    for tevents in data_loader:
        # tevents = torch.cat((tevents,(torch.arange(1024)/1024).repeat(args.batch_size,1).unsqueeze(-1)), dim=-1)
        # tevents, indices= torch.sort(tevents,dim=1)
        tevents = tevents.unsqueeze(1)
        # tevents = torch.log(tevents)
        # Zero gradients
        optimizer.zero_grad()
        # Note: the .t() is needed so that the model interprets the data as a SINGLE sample ("sample_size" = 1)
        t_tensor = torch.tensor([])
        xT = torch.tensor([])
        eps = torch.tensor([])
        for i in range(0,tevents.size(0)):
            # randomly choose t uniformly from between 1 and T
            t = np.random.randint(1, T)
            eps_i = torch.randn_like(tevents[i,:,:])
            if t < T:
                xT_i = np.sqrt(alphabars[t]) * tevents[i,:,:] + np.sqrt(1 - alphabars[t]) * eps_i
            else:  # when t == T
                xT_i = eps_i
            eps = torch.cat((eps,eps_i.unsqueeze(0)),dim=0)
            t = torch.tensor([t], dtype=torch.float32, device=device)
            # xT_i = torch.cat((xT_i, torch.arange(1024).unsqueeze(0).unsqueeze(0)), dim=1)
            t_tensor = torch.cat((t_tensor, t),dim=0)
            xT = torch.cat((xT, xT_i),dim=0)
        xT = xT.unsqueeze(1)
        eps_theta = model(t_tensor, xT.float()) # predicts eps_theta in log space
        # _, constraints_loss = enforce_constraints(eps_theta[:,0,:,1])
        t = int(t_tensor[0].item())
        if t == 1:
            z = torch.zeros_like(xT)
        else:
            z = torch.randn_like(xT)
        xTminus1_true = np.sqrt(alphabars[t - 1]) * xT + np.sqrt(
            1 - alphabars[t - 1]
        ) * torch.randn_like(xT)
        inputs = xT # [:,:,:].unsqueeze(0)
        outputs = eps_theta
        xTminus1_pred = (1 / (np.sqrt(alphas[t]))) * (
                inputs - ((1 - alphas[t]) / np.sqrt(1 - alphabars[t])) * outputs
        ) + sigma(alphas, alphabars, t) * z
        # xTminus1_pred[...,-1] = enforce_constraints(xTminus1_pred[...,-1],t_tensor)
        # loss = 0.05*criterion(xTminus1_true,xTminus1_pred) +0.002*criterion_sorted(eps[...,0:-1],eps_theta[...,0:-1])+ 0.002*criterion_sorted(eps[...,-1],eps_theta[...,-1])# + criterion(eps[...,-1],eps_theta[...,-1])# 0.01* criterion_sorted(eps[...,0:-1],eps_theta[...,0:-1]) + # criterion(xTminus1_true[:,:,:,0:-1],xTminus1_pred[:,:,:,0:-1]) + 0.1*torch.log(criterion(xTminus1_true[:,:,:,-1],xTminus1_pred[:,:,:,-1])) +  # + 0.01*unique_loss(xTminus1_pred[:,:,:,-1].squeeze())
        # loss = criterion(eps, eps)
        loss_sorted = 0.05*compute_jsd(xTminus1_true,xTminus1_pred)#torch.mean(1-torch.log(torch.abs(eps_theta)))#criterion_sorted(eps,eps_theta)
        print("JSD")
        print(loss_sorted)
        # losses_sorted.append(loss_sorted.detach().numpy())
        loss_xT = criterion(eps,eps_theta)
        losses_xT.append(loss_xT.cpu().detach().numpy())
        loss = loss_xT #+loss_sorted
        loss.backward()
        losses.append(loss.cpu().detach().numpy())
        print(sum(dict((p.data_ptr(), p.numel()) for p in model.parameters()).values()))
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        # Update weights
        optimizer.step()

        fig1 = plt.figure(1)
        plot_dist(xTminus1_pred[0, 0, :, :].cpu().detach().numpy(), t_tensor[0], fig=fig1,plot_Gaussian=False,plot_other_data=True,other_data=xTminus1_true[0,0,:,:].detach().numpy())
        fig2 = plt.figure(2)
        plot_dist(eps_theta[0, 0, :, :].cpu().detach().numpy(), t_tensor[0], fig=fig2)
        fig3 = plt.figure(3)
        plt.clf()
        # Get the Axes object associated with the figure
        ax = fig3.add_subplot(1, 1, 1)  # This creates a single subplot (1 row, 1 column, index 1)
        ax.plot(np.asarray(losses), alpha=0.5, label="Step Loss")
        ax.plot(np.asarray(losses_xT), alpha=0.5, label="xT Loss")
        ax.plot(np.asarray(losses_sorted), alpha=0.5, label="Sorted Loss")
        ax.set_ylabel("Loss")
        ax.set_xlabel("Step")
        ax.set_title("Training Loss")
        plt.legend()
        plt.tight_layout()

        plt.draw()
        plt.pause(0.0001)
        print(
            f'Epoch {epoch + 1}, Loss: {loss.item()}, Learning Rate: {optimizer.param_groups[0]["lr"]}'
        )

    losses_over_epochs.append(np.mean(losses_over_epoch))

def alphabar(alpha, t):
    alphabar = alpha
    for i in range(0, t - 1):
        alphabar = alphabar * alpha
    return alphabar


def sigma(alphas, alphabars, t):
    numerator = (1 - alphas[t]) * (1 - alphabars[t - 1])
    denominator = 1 - alphabars[t]
    return np.sqrt(numerator / denominator)


def generate_noising_schedule(alpha0, T, beta=0.001):
    alpha_schedule = [alpha0 - beta * i for i in range(T)]
    return alpha_schedule

def plot_marginals(samples,t, plot_Gaussian = True, fig = None, axes=None, plot_other_data=False, other_data=None):
    num_dimensions = samples.shape[1]
    num_rows = min(2, num_dimensions)
    num_cols = (num_dimensions + num_rows - 1) // num_rows

    if fig is None:
        fig = plt.figure(figsize=(8, 3*num_rows))
    else:
        fig.clf()  # Clear current figure

    if axes is None:
        axes = fig.subplots(num_rows, num_cols)
        axes = axes.flatten()

    # Determine global min and max across all dimensions
    global_min = samples.min()
    global_max = samples.max()
    inputs = sample


def plot_dist(samples,t=0, plot_Gaussian = True, fig = None, axes=None, plot_other_data=False, other_data=None):
    num_dimensions = samples.shape[1]
    num_rows = min(2, num_dimensions)
    num_cols = (num_dimensions + num_rows - 1) // num_rows

    if fig is None:
        fig = plt.figure(figsize=(8, 3*num_rows))
    else:
        fig.clf()  # Clear current figure

    if axes is None:
        axes = fig.subplots(num_rows, num_cols)
        axes = axes.flatten()

    # Determine global min and max across all dimensions
    global_min = samples.min()
    global_max = samples.max()

    for i in range(num_dimensions):
        axes[i].hist(samples[:, i], bins=30, density=True, alpha=0.6)
        axes[i].set_title(f"Dim {i + 1}", fontsize=8)
        axes[i].tick_params(axis='both', which='both', labelsize=6)

        # Set uniform x and y axes
        axes[i].set_xlim(global_min, global_max)
        axes[i].set_ylim(0, 1.1 * axes[i].get_ylim()[1])

        if plot_Gaussian == True:
            axes[i].hist(torch.randn(1024), bins=30, density=True, alpha=0.1)
        if plot_other_data ==True:
            axes[i].hist(other_data[:,i], bins=30, density=True, alpha=0.1)

    # Hide unused subplots
    for j in range(num_dimensions, len(axes)):
        axes[j].axis('off')
    plt.suptitle(str(t), fontsize=16)
    plt.tight_layout()
    plt.draw()
    plt.pause(0.1)
    # # Save the current frame as an image
    # buffer = io.BytesIO()
    # plt.savefig(buffer, format="png")
    # buffer.seek(0)
    # frames.append(Image.open(buffer))
    #

def main():
    # Arguments
    parser = argparse.ArgumentParser(
        description="Diffusion Denoising Model for Probability Distribution Parameter Estimation"
    )
    parser.add_argument(
        "--hidden-dim", type=int, default=100, help="Hidden dimension size (default: 1)"
    )
    parser.add_argument(
        "--T", type=int, default=20, help="Number of noising time steps (default: 100)"
    )

    parser.add_argument(
        "--alpha", default=0.999, help="Initial alpha in variance schedule (default: 0.99999)"
    )
    parser.add_argument(
        "--beta", default=0.005, help="Linear rate of variance schedule (default: 0.001)"
    )
    parser.add_argument(
        "--distribution",
        default="mixed_multidimensional_8D",
        help="True data distribution (default: exp)",
    )
    parser.add_argument(
        "--lr", type=float, default=0.01, help="learning rate (default: 0.1)"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed (default: 42)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=1024,
        help="input sample size for training (default: 1024)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="input batch size for training (default: 3)",
    )
    parser.add_argument(
        "--n-true-events",
        type=int,
        default=1024000,
        help="Number of true events for training (default: 10000)",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default='epochs_200_n_true_events_1024000_batch_size_100_mixed',
        help="Filename for results",
    )
    args = parser.parse_args()

    # Hyperparameters
    alpha = 0.99999

    # Set true parameters
    if args.distribution == "normal":
        mu = 1.6  # true mean of Gaussian
        std = 0.8  # true std of Gaussian
        tparams = torch.tensor([mu, std])
    elif args.distribution == "exp":
        rate = 2  # true rate of exponential distribution
        tparams = torch.tensor([rate])
    elif args.distribution == "mixture":
        rate = 2  # true rate of exponential distribution
        tparams = torch.tensor([rate])
    elif args.distribution == "multidimensional":
        rate = 2  # true rate of exponential distribution
        tparams = torch.tensor([rate])
    elif args.distribution == "2D":
        rate = 2  # true rate of exponential distribution
        tparams = torch.tensor([rate])
    elif args.distribution == "mixed_multidimensional_8D":
        p = 5
        n = 8
        # Generate random parameters
        random_means = generate_random_means(p, n)
        random_variances = generate_random_variances(p, n)
        random_weights = generate_random_weights(p)
        tparams = pack_params(random_means, random_variances, random_weights)


    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cpu")
    print("Device: {}".format(device))

    # Create the denoising diffusion model
    model = DenoisingDiffusionModelUNetND(args.T)

    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # tparams =2
    # Generate true events dataset
    dataset = generate_synthetic_data(
        tparams, args.n_true_events, device, option=args.distribution
    ).squeeze()


    # dataset = torch.log(dataset)
    # Generate alphas
    alphas = generate_noising_schedule(alpha, args.T, args.beta)
    alphabars = np.cumprod(alphas)

    # Initialize losses
    losses = []
    losses_xT = []
    losses_sorted = []
    losses_over_epochs = []

    # Create an empty list to store frames for the GIF
    frames = []

    shuffle_data = True
    samples_data_loader = DataLoader(
        dataset=dataset, batch_size=args.sample_size, shuffle=shuffle_data
    )
    dataset_with_subsamples = torch.tensor([]).to(device)
    for tevents in samples_data_loader:
        if tevents.size(0) != 1024:
            continue
        dataset_with_subsamples = torch.cat((dataset_with_subsamples,tevents.unsqueeze(0)),dim=0)

    data_loader = DataLoader(
        dataset=dataset_with_subsamples, batch_size=args.batch_size, shuffle=shuffle_data
    )

    plt.figure(figsize=(15, 3))

    for epoch in range(args.epochs):
        model.train()
        train_loop(
            frames,
            epoch,
            model,
            optimizer,
            data_loader,
            alphabars,
            device,
            losses,
            losses_sorted,
            losses_xT,
            losses_over_epochs,
            args,
            alphas
        )

    torch.save(model.state_dict(),  str(args.filename) + 'model.pt')

    model.eval()
    plt.clf()

    # Inference

    frames = []
    # inputs, indices = torch.sort(inputs, dim=-1)
    tevents = next(iter(samples_data_loader)).t().unsqueeze(0).unsqueeze(0).permute(0,1,3,2)
    inputs = torch.randn_like(tevents)
    # tevents = torch.cat((tevents, (torch.arange(1024)/1024).unsqueeze(0).unsqueeze(0)),
    #                 dim=1)
    # tevents = torch.log(tevents)    inputs = torch.randn_like(tevents)
    # tevents, indices = torch.sort(tevents,dim=-1)
    T = args.T
    with torch.no_grad():
        for t in reversed(range(1, T)):
            t_tensor = torch.tensor([t], dtype=torch.float32, device=device)
            if t == 1:
                z = torch.zeros_like(tevents)
            else:
                z = torch.randn_like(tevents)
            outputs = model(t_tensor, inputs)
            # xT = np.sqrt(alphabars[t - 1]) * tevents + np.sqrt(
            #         1 - alphabars[t - 1]
            #     ) * torch.randn_like(tevents)
            inputs = (1 / (np.sqrt(alphas[t]))) * (
                        inputs - ((1 - alphas[t]) / np.sqrt(1 - alphabars[t])) * outputs
                ) + sigma(alphas, alphabars, t) * z
            # xTminus1_true = np.sqrt(alphabars[t - 1]) * xT + np.sqrt(
            #     1 - alphabars[t - 1]
            # ) * torch.randn_like(xT)
            fig1 = plt.figure(1)
            plt.clf()
            # plot_dist(inputs[0, 0, :, :].detach().numpy(), t, fig=fig1,plot_Gaussian=False,plot_other_data=True,other_data=tevents[0,0,:,:].detach().numpy())
            plot_dist(inputs[0, 0, :, :].cpu().detach().numpy(), t, fig=fig1, plot_Gaussian=False, plot_other_data=True, other_data=tevents[0,0,:,:].cpu().detach().numpy())
            plt.tight_layout()

            plt.draw()
            plt.pause(0.0001)

            # Save the current frame as an image
            buffer = io.BytesIO()
            plt.savefig(buffer, format="png")
            buffer.seek(0)
            frames.append(Image.open(buffer))
    frames[0].save(
        "inference_histograms" + str(args.filename) +".gif",
        save_all=True,
        append_images=frames[1:],
        duration=200,
        loop=0,
    )
#
#
# plt.clf()
# plt.clf()
# # Plot histograms
# plt.scatter(
#     xT_exp[0, :],
#     xT_exp[1, :],
#     alpha=0.5,
#     label="True x_{t-1}",
# )
# plt.scatter(
#     inputs_exp[0, :],
#     inputs_exp[1, :],
#     alpha=0.5,
#     label="x_{t-1} with Generated Noise",
# )
# plt.scatter(
#     tevents_exp[0, :],
#     tevents_exp[1, :],
#     alpha=0.5,
#     label="True Events x_{0}",
# )
# plt.ylabel("Frequency")
# plt.xlabel("Value")
# plt.legend()
# plt.title(f"t {t}")
# # # Set fixed x and y axes
# # plt.xlim([-2, 4])
# # plt.ylim([0, 70])

# # Save the current frame as an image
#             buffer = io.BytesIO()
#             plt.savefig(buffer, format="png")
#             buffer.seek(0)
#             # frames.append(Image.open(buffer))
#
        # plt.draw()
        # plt.pause(0.01)

    # frames[0].save(
    #     "inference_histograms" + str(args.filename) +".gif",
    #     save_all=True,
    #     append_images=frames[1:],
    #     duration=200,
    #     loop=0,
    # )

    # frames[-1].save("inference_histograms" + str(args.filename) + ".png")
    np.save(str(args.filename) + 'losses.npy', losses)

if __name__ == "__main__":
    main()
