import argparse
import os
import time

import torch
from data_loader import get_loader
from train import Trainer
from utils import plot_example
from model import Generator, Discriminator, process_feature_map


def main():
    # load data
    train_dataloader, test_dataloader = get_loader()

    # plot example
    sample_facade, sample_map = next(iter(train_dataloader))
    plot_example(sample_facade, sample_map, title1='facade', title2='map')

    # model
    generator_g, generator_f = Generator(), Generator()
    discriminator_x, discriminator_y = Discriminator(), Discriminator()

    # Feature map
    disc, feat_map = discriminator_x(sample_facade, return_feature_map=True)
    feat_map = process_feature_map(feat_map)
    x = sample_facade * feat_map

    # Plot Feature map
    plot_example(sample_facade, feat_map, title1='facade', title2='feature map')
    plot_example(sample_facade, x, title1='facade', title2='facade * feature map')

    # Train model
    trainer = Trainer(train_dataloader, generator_g, generator_f, discriminator_x, discriminator_y)
    trainer.train()
    generated = trainer.generate(sample_map)
    plot_example(sample_map, generated, title1='map', title2='generated')


if __name__ == '__main__':
    main()
