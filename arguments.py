import argparse


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Arguments for training OOD", add_help=True)

    parser.add_argument(
        "--train-data-path", type=str, required=True, help="path to the training dataset folder. It should contain folders where each folder represents one class."
    )
    parser.add_argument(
        "--test-data-path", type=str, required=True, help="path to testing dataset folder. It should contain folders where each folder represents one class."
    )
    parser.add_argument(
        "--known-classes", type=str, required=True, help="string with comma seperated classes which represents known classes or IN distribution classes"
    )
    parser.add_argument(
        "--unknown-classes", type=str, required=True, help="string with comma seperate classes which represents unknown classes  or out distribution classes"
    )
    parser.add_argument(
        "--train-val-test-split", type=str, default="0.8,0.1,0.1", help="string with comma seperate value where value represent train, val and test split respectively. Note carefully that this split is done only on train dataset. default value is 0.8,0.1,0.1"
    )

    parser.add_argument(
        "--train-crop-size", type=int, required=True, help="image crop size which will be used at the time of training"
    )
    parser.add_argument(
        "--test-resize-size", type=int, required=True, help="image resize that will be used at the time of inference."
    )
    parser.add_argument(
        "--test-crop-size", type=int, required=True, help="size of center crop that will be used after the resize at the time of inference."
    )
    parser.add_argument(
        "--hflip-prob", type=float, default=0.0, help="probability of horizontal flip. default value is 0.0"
    )
    parser.add_argument(
        "--random-erase-prob", type=float, default=0.0, help="probability of random erase augmentation. default value is 0.0"
    )

    parser.add_argument(
        "--batch-size", type=int, default=32, help="batch size for training. If not provided, default value 32 will be used."
    )
    parser.add_argument(
        "--num-workers", type=int, default=4, help="number of workers for data loading. If not provided, default value 4 will be used."
    )
    parser.add_argument(
        "--epochs", type=int, default=30, help="number of epochs to run. default value is 30."
    )
    parser.add_argument(
        "--early-stopping-epochs", type=int, default=-1, help="stop the training if the validation accuracy is not increased for these many epochs."
    )

    parser.add_argument(
        "--model", type=str, default="resnet50", help="base model to be used. As of now only resnet variations are allowed. default value is resnet50, imagenet weights will be used."
    )
    parser.add_argument(
        "--label-smoothing", type=float, default=0.0, help="label smoothing for cross entropy loss. default value is 0.0"
    )
    parser.add_argument(
        "--optimizer", type=str, default="sgd", help="optimizer to use. As of now only sgd and adam are supported."
    )
    parser.add_argument(
        "--lr", type=float, default=0.1, help="initial learning rate."
    )
    parser.add_argument(
        "--lr-min", type=float, default=0.0, help="minimum learning rate for cosine schedule."
    )
    parser.add_argument(
        "--momentum", type=float, default=0.9, help="momentum used for the optimizer."
    )
    parser.add_argument(
        "--weight-decay", type=float, default=1e-4, help="weight decay to be used."
    )
    parser.add_argument(
        "--norm-weight-decay", type=float, default=1e-4, help="weight decay for normalization layers, like batchnorm etc."
    )
    parser.add_argument(
        "--lr-warmup-epochs", type=int, default=0, help="number of epochs for linear warmup"
    )
    parser.add_argument(
        "--lr-warmup-decay", type=float, default=0.01, help="the decay of the linear warmup"
    )

    parser.add_argument(
        "--seed", type=int, default=42, help="seed value for the run. If not provided, default value 42 will be used."
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="device to use for training. Supported values are cpu and cuda."
    )
    parser.add_argument(
        "--output-dir", type=str, default="runs", help="path to save the outputs. In the output directory, a directory with unique id will be created and all the ouptut will stored over there."
    )

    return parser
