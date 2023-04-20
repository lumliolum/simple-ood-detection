import os
import uuid

import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torchvision
from loguru import logger
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

import utils
from engine import train_one_epoch, evaluate
from arguments import get_parser
from models import OODResNetModel
from dataset import ImageDataset, ImageTestDataset
from transforms import get_train_transforms, get_test_transforms


def main(args):
    # get the device
    device = torch.device(args.device)
    logger.info(f"Using device = {device}")

    # set the seed for the run
    logger.info(f"Setting the seed value = {args.seed} for the run")
    utils.set_seed(seed=args.seed, device=device)

    known_classes = args.known_classes.split(",")
    unknown_classes = args.unknown_classes.split(",")
    logger.info(f"Known classes = {known_classes}")
    logger.info(f"Unknown classes = {unknown_classes}")

    # get the image paths for train data only using known classes.
    logger.info(f"Reading the image paths, labels from the train directory = {args.train_data_path}")
    image_paths, image_labels = utils.get_image_paths_and_labels(
        dirpath=args.train_data_path,
        include_classes=known_classes
    )
    logger.info(f"Number of images found = {len(image_paths)}")
    # best part of np.unique is it gives the uniques values in sorted order.
    unique_labels = np.unique(image_labels)
    label2idx = {label: idx for idx, label in enumerate(unique_labels)}
    idx2label = {idx: label for idx, label in enumerate(unique_labels)}
    logger.info(f"Number of classes = {len(label2idx)}")
    logger.info(f"label to idx mapping = {label2idx}")

    train_val_test_split = args.train_val_test_split.split(",")
    try:
        train_val_test_split = [float(x) for x in train_val_test_split]
    except Exception as ex:
        raise ValueError(f"Cannot convert {train_val_test_split} to float values, received expection = {ex}")

    # check if the split values are valid or not.
    utils.check_split_values(train_val_test_split)

    logger.info(f"Splitting the train data in ratio of {train_val_test_split}")
    train_image_paths, val_image_paths, train_labels, val_labels = train_test_split(
        image_paths,
        image_labels,
        test_size=1 - train_val_test_split[0],
        random_state=args.seed,
        shuffle=True,
        stratify=image_labels
    )
    val_image_paths, test_image_paths, val_labels, test_labels = train_test_split(
        val_image_paths,
        val_labels,
        test_size=train_val_test_split[2] / (1 - train_val_test_split[0]),
        random_state=args.seed,
        shuffle=True,
        stratify=val_labels
    )

    # get the train and test transforms.
    logger.info("Get the train transforms using the values")
    logger.info(f"Image size = {args.train_crop_size}, hflip prob = {args.hflip_prob}, random erase prob = {args.random_erase_prob}")
    train_transforms = get_train_transforms(
        crop_size=args.train_crop_size,
        hflip_prob=args.hflip_prob,
        random_erase_prob=args.random_erase_prob
    )

    logger.info("Get the test transforms using the values")
    logger.info(f"Image resize size = {args.test_resize_size}, crop size = {args.test_crop_size}")
    test_transforms = get_test_transforms(
        resize_size=args.test_resize_size,
        crop_size=args.test_crop_size
    )

    logger.info("Initializing the datasets")
    train_dataset = ImageDataset(
        image_paths=train_image_paths,
        labels=train_labels,
        transforms=train_transforms,
        label2idx=label2idx
    )
    val_dataset = ImageDataset(
        image_paths=val_image_paths,
        labels=val_labels,
        transforms=test_transforms,
        label2idx=label2idx
    )
    test_dataset = ImageDataset(
        image_paths=test_image_paths,
        labels=test_labels,
        transforms=test_transforms,
        label2idx=label2idx        
    )
    logger.info(f"Length of Train dataset = {len(train_dataset)}")
    logger.info(f"Length of Val dataset = {len(val_dataset)}")
    logger.info(f"Length of Test dataset = {len(test_dataset)}")

    logger.info(f"Initializing data loaders, using batch size = {args.batch_size}")
    genet = torch.Generator()
    genet.manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        worker_init_fn=utils.set_worker_seed,
        generator=genet
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        worker_init_fn=utils.set_worker_seed,
        generator=genet
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        worker_init_fn=utils.set_worker_seed,
        generator=genet
    )

    logger.info("Initializing model with backbone model = {}".format(args.model))
    # by default I am using imagenet v2 weights.
    pretrained_weights = torchvision.models.get_model_weights(args.model).IMAGENET1K_V2 # type: ignore
    logger.info(f"Pretrained weights = {pretrained_weights}")
    backbone = torchvision.models.get_model(args.model, weights=pretrained_weights)
    model = OODResNetModel(backbone, num_classes=len(label2idx)) # type: ignore
    model.to(device)

    logger.info(f"Initializing cross entropy loss function with label smoothing = {args.label_smoothing}")
    loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    logger.info(f"Using weight decay = {args.weight_decay} and norm weight decay = {args.norm_weight_decay}")
    parameters = utils.set_weight_decay(model, args.weight_decay, args.norm_weight_decay)

    logger.info(f"Initializing optimizer = {args.optimizer} with learning rate = {args.lr}, momentum = {args.momentum}")
    opt_name = args.optimizer.lower()
    if opt_name == "sgd":
        optimizer = optim.SGD(
            parameters,
            lr=args.lr,
            momentum=args.momentum,
            nesterov=False
        )
    elif opt_name == "rmsprop":
        optimizer = optim.RMSprop(
            parameters,
            lr=args.lr,
            momentum=args.momentum,
            eps=0.0316,
            alpha=0.9
        )
    elif opt_name == "adam":
        logger.warning("For adam optimizer, momentum parameter will not be used.")
        optimizer = torch.optim.Adam(
            parameters,
            lr=args.lr
        )
    else:
        raise ValueError(f"Invalid optimizer {args.opt}. Only SGD, RMSprop and Adam are supported.")

    # learning rate sceduler
    logger.info(f"Initializing the learning rate scheduler over {args.epochs} epochs (all training)")
    logger.info(f"Received minimum lr = {args.lr_min}, warmup epochs = {args.lr_warmup_epochs}, warmup decay = {args.lr_warmup_decay}")

    main_lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs - args.lr_warmup_epochs,
        eta_min=args.lr_min
    )

    if args.lr_warmup_epochs > 0:
        warmup_scheduler = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=args.lr_warmup_decay,
            total_iters=args.lr_warmup_epochs
        )
        lr_scheduler = optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, main_lr_scheduler],
            milestones=[args.lr_warmup_epochs]
        )
    else:
        logger.warning("Warmup will not be used, and only main learning rate scheduler will be used")
        lr_scheduler = main_lr_scheduler

    # setup the early stopping
    if args.early_stopping_epochs < 0:
        # if it is less than zero, then we will not use any early stopping at all.
        args.early_stopping_epochs = args.epochs + 1


    logger.info("Start training")
    best_validation_acc = -np.inf
    best_validation_loss = +np.inf
    best_checkpoint_path = os.path.join(args.output_dir, "checkpoint.pth")
    change_epoch_counter = 0
    lr_schedule = []

    for epoch in range(args.epochs):
        # https://github.com/pytorch/pytorch/issues/76113
        lr_schedule.append(lr_scheduler.get_last_lr()[0])

        train_loss, train_acc, train_time_taken = train_one_epoch(model, loss_fn, optimizer, train_loader, device)
        val_loss, val_acc, val_time_taken = evaluate(model, loss_fn, val_loader, device)
        lr_scheduler.step()

        logmsg = f"Epoch = {epoch + 1}/{args.epochs}, time taken = {train_time_taken + val_time_taken} seconds, train loss = {train_loss}, train acc = {train_acc}, val loss = {val_loss}, val acc = {val_acc}"
        logger.info(logmsg)

        if val_acc > best_validation_acc:
            logger.info(f"Validation Accuracy improved from {best_validation_acc} to {val_acc}")
            best_validation_acc = val_acc
            best_validation_loss = val_loss

            logger.info(f"Saving the model at path = {best_checkpoint_path}")
            checkpoint = {
                "model": model.state_dict(),
                "epoch": epoch + 1,
                "args": vars(args)
            }
            torch.save(checkpoint,  best_checkpoint_path)
            # as the validation accuracy increased, make the change epoch counter to 0.
            change_epoch_counter = 0
        else:
            change_epoch_counter += 1
            if change_epoch_counter >= args.early_stopping_epochs:
                logger.warning("Early stopping reached as validation acc didn't increase for {args.early_stopping_epochs} epochs")
                break

    logger.info("Training Completed")
    logger.info(f"Best validation accuracy = {best_validation_acc}, best validation loss = {best_validation_loss}")

    plot_save_path = os.path.join(args.output_dir, "lr_schedule.png")
    logger.info(f"Saving the lr schedule in {plot_save_path}")
    plt.plot(lr_schedule)
    plt.savefig(plot_save_path)
    plt.close()

    logger.info(f"Loading the best model from {best_checkpoint_path}")
    model = OODResNetModel(backbone, len(label2idx)) # type: ignore

    checkpoint = torch.load(best_checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    model.to(device)

    logger.info("Evaluating on testing data")
    test_loss, test_acc, test_time_taken = evaluate(model, loss_fn, test_loader, device)
    logger.info(f"Time taken = {test_time_taken}, test accuracy = {test_acc}, test loss = {test_loss}")

    # create a train loader with only test transforms.
    # the original dataloader has train transforms (which may contain horizontal flip etc)
    logger.info("Constructing evaluate train dataset and train loader")
    evaluate_train_dataset = ImageDataset(
        image_paths=train_image_paths,
        labels=train_labels,
        transforms=test_transforms,
        label2idx=label2idx
    )
    logger.info(f"Length of evaluate train dataset = {len(evaluate_train_dataset)}")
    evaluate_train_loader = DataLoader(
        evaluate_train_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )

    # gather the embeddings on train data.
    # so that we can calculate the mean, covariances for each class.
    logger.info("Gathering embeddings on train data")
    train_embds_array, _, train_labels_array = utils.gather_preds_and_embeddings(model, evaluate_train_loader, device, gather_labels=True)

    logger.info("Number of layers = {} from which embeddings are extracted".format(len(train_embds_array))) # type: ignore

    # now for each layer calculate mean for each class and tied covariance.
    logger.info("Calculating means, tied covariance for each layer")
    train_means, train_tied_covs = utils.calculate_mean_and_cov_matrix(train_embds_array, train_labels_array) # type: ignore

    # load the test data which contains both known and unknown samples
    new_test_image_paths, new_test_labels = utils.get_image_paths_and_labels(
        dirpath=args.test_data_path,
        include_classes=known_classes + unknown_classes
    )
    logger.info(f"Number of images found = {len(new_test_image_paths)}")
    logger.info("Initialize the test dataset")
    new_test_dataset = ImageTestDataset(
        image_paths=new_test_image_paths,
        transforms=test_transforms
    )
    logger.info(f"length of new test dataset = {len(new_test_dataset)}")

    # have the new test data with batch size = 1 (single sample prediction)
    logger.info(f"Intializing the dataloader with batch size = {args.batch_size} for prediction on new test data.")
    new_test_loader = DataLoader(
        new_test_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )

    logger.info("Gather predictions and embeddings on new test data")
    new_test_embds_array, new_test_preds_array, _ = utils.gather_preds_and_embeddings(model, new_test_loader, device, gather_labels=False)

    # scores matrix -> (num samples, num layers)
    new_test_score_matrix = np.zeros((len(new_test_preds_array), len(new_test_embds_array))) # type: ignore
    logger.info(f"New test data score matrix shape = {new_test_score_matrix.shape}")

    logger.info("Calculating the scores from each layer")
    for layer_index, test_layer_embds in enumerate(new_test_embds_array): # type: ignore
        train_mean = train_means[layer_index]
        train_tied_cov = train_tied_covs[layer_index]

        # create an zero matrix for this layer confidences -> (num samples, num classes)
        test_layer_confidence_matrix = np.zeros((len(test_layer_embds), len(train_mean)))

        for label, label_mean in train_mean.items():
            test_scores = utils.mahalanobis(test_layer_embds, label_mean, train_tied_cov)
            # small check for my satisfaction
            assert len(test_scores) == len(test_layer_embds)
            test_layer_confidence_matrix[:, label] = test_scores

        # now that we have score for each class, then take the maximum along each class
        # that is for each sample, only consider the highest score.
        test_layer_confidences = np.max(test_layer_confidence_matrix, axis=1)

        new_test_score_matrix[:, layer_index] = test_layer_confidences
    
    # take the mean along the layers to get the final score for each test sample
    # in the paper they have used different weights to combined scores from each layer.
    # these weights were learned from logistic regression on validation data.
    # but here I don't have out distribution validation data, so I will be taking 
    # equal weights that is taking the mean. The idea of average is taken from Generalized ODIN paper.
    logger.info("Calculating the average from the score of each layer")
    new_test_scores = np.mean(new_test_score_matrix, axis=1)
    logger.info(f"Final test score shape = {new_test_scores.shape}")

    new_test_df = pd.DataFrame()
    new_test_df["image_paths"] = new_test_image_paths
    new_test_df["gt"] = new_test_labels
    
    # store the prediction coming from the resnet softmax layer.
    new_test_df["pred"] = new_test_preds_array
    # as these are model idx convert them to labels.
    new_test_df["pred"] = new_test_df["pred"].map(idx2label)

    new_test_df["confidence"] = new_test_scores
    logger.info(f"New test data stored in dataframe, shape = {new_test_df.shape}")

    logger.info("Calculate the metrics on new test data")
    new_test_metrics = utils.calculate_ood_metrics(new_test_df, known_classes, unknown_classes)
    logger.info(f"Calculated metrics = {new_test_metrics}")

    metrics_save_path = os.path.join(args.output_dir, "new_test_metrics.json")
    logger.info(f"Saving the metrics file at path = {metrics_save_path}")
    utils.save_json(new_test_metrics, metrics_save_path)

    logger.info("at last run completed.")


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    run_id = uuid.uuid4().hex[0:10]
    args.output_dir = os.path.join(args.output_dir, run_id)
    os.makedirs(args.output_dir, exist_ok=True)

    logger.add(
        sink=os.path.join(args.output_dir, "run.log"),
        level="DEBUG"
    )
    logger.info(f"Starting the run with run_id = {run_id}")
    logger.info(f"Output will be saved at = {args.output_dir}")

    main(args)
