from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt
import dload


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
