from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt
import dload
import torch
import torch.nn.functional as F
import torchvision.models as models
from scipy.linalg import sqrtm


def compute_gram_matrix(x):
    (batch, channels, height, width) = x.size()
    features = x.view(batch, channels, height * width)
    features_t = features.transpose(1, 2)
    return torch.bmm(features, features_t) / (channels * height * width)


def compute_mmd(x, y):
    x_mean = torch.mean(x, dim=0)
    y_mean = torch.mean(y, dim=0)
    x_cov = compute_gram_matrix(x - x_mean.unsqueeze(0))
    y_cov = compute_gram_matrix(y - y_mean.unsqueeze(0))
    return torch.mean(torch.diagonal(x_cov + y_cov - 2 * torch.mm(x_cov, y_cov)))


def compute_kid(real_images, generated_images, batch_size=64, device='cuda'):
    inception = models.inception_v3(pretrained=True, transform_input=False).to(device)
    inception.eval()

    n_batches = len(real_images) // batch_size
    real_features = []
    generated_features = []

    with torch.no_grad():
        for i in range(n_batches):
            real_batch = real_images[i * batch_size: (i + 1) * batch_size].to(device)
            generated_batch = generated_images[i * batch_size: (i + 1) * batch_size].to(device)

            real_features.append(inception(real_batch)[0].view(real_batch.size(0), -1))
            generated_features.append(inception(generated_batch)[0].view(generated_batch.size(0), -1))

        real_features = torch.cat(real_features, 0)
        generated_features = torch.cat(generated_features, 0)

        mmd = compute_mmd(real_features, generated_features)
        return mmd.item()


def download_dataset(dataset_name):
    """Download dataset from https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/
    Args:
        dataset_name: Name of the dataset.
    """
    _url = f'https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/{dataset_name}.zip'
    return dload.save_unzip(_url, "datasets") + f"/{dataset_name}"


def plot_examples(dataloader, n=10):
    """Plot n examples from the dataloader
    Args:
        dataloader: DataLoader object
        n: number of examples to plot
    """
    for _ in range(n):
        sample_image1, sample_image2 = next(iter(dataloader))
        plot_example(sample_image1, sample_image2)


def plot_example(image1, image2, title1=None, title2=None):
    """Plot two images side by side
    Args:
        image1: image tensor
        image2: image tensor
    """
    plot_image(title1, 121, image1)
    plot_image(title2, 122, image2)
    plt.show()


def plot_image(title, subplot_number, image):
    """Plot image
    Args:
        title: title of the image
        subplot_number: subplot number
        image: image tensor
    """
    plt.subplot(subplot_number).set_title(title)
    plt.imshow(to_pil_image(image[0]))
