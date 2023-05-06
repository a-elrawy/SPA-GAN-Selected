import argparse
from data_loader import get_loader, prepare_image_for_generation
from train import Trainer
from utils import plot_example
from model import Generator, Discriminator, process_feature_map


def main(args):
    # load data
    train_dataloader, test_dataloader = get_loader(dataset_name=args.dataset)

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
    trainer.set_optimizers(lr=args.lr, beta1=args.beta1, beta2=args.beta2)
    trainer.train(epochs=args.epochs, checkpoint_dir=args.checkpoint_dir, save_checkpoint=args.save_checkpoint,
                  load_checkpoint=args.load_checkpoint, use_wandb=args.wandb)

    # Generate
    generated = None
    if args.generate_source:
        generated = trainer.generate(prepare_image_for_generation(args.generate_source), 'g')
    if args.generate_target:
        generated = trainer.generate(prepare_image_for_generation(args.generate_target), 'f')
    if generated is not None:
        plot_example(sample_map, generated, title1='map', title2='generated')

    # Evaluate
    if args.evaluate:
        trainer.evaluate(test_dataloader, dataset=args.dataset)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='facades', help='dataset name')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
    parser.add_argument('--generate_source', type=str, default=None, help='generate from source')
    parser.add_argument('--generate_target', type=str, default=None, help='generate from target')
    parser.add_argument('--save_checkpoint', type=bool, default=True, help='save checkpoint')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='checkpoint directory')
    parser.add_argument('--load_checkpoint', type=bool, default=True, help='load checkpoint')
    parser.add_argument('--evaluate', type=bool, default=False, help='evaluate')
    parser.add_argument('--wandb', type=bool, default=False, help='use wandb')
    args = parser.parse_args()

    main(args)
