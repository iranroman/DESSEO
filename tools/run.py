# created by Iran R. Roman <iran@ccrma.stanford.edu>
"""Wrapper to train and test DESSEO."""
import yaml

from desseo.utils.parser import parse_args, load_config

#from test_net import test
from train_net import train
#from visualization import visualize


def main():
    """
    Main function to spawn the train and test process.
    """
    # parse config arguments
    args = parse_args()
    cfg = load_config(args, args.path_to_config)

    # Perform training.
    if cfg.TRAIN.ENABLE:
        train(cfg=cfg)

    # Perform evaluation.
    if cfg.TEST.ENABLE:
        test(cfg=cfg)

    # Perform model visualization.
    if cfg.TENSORBOARD.ENABLE and (
        cfg.TENSORBOARD.MODEL_VIS.ENABLE
        or cfg.TENSORBOARD.WRONG_PRED_VIS.ENABLE
    ):
        visualize(cfg=cfg)


if __name__ == "__main__":
    main()
