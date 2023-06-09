python3 train.py \
    --train-data-path data/MNIST_MINI/MNIST_TRAIN \
    --test-data-path data/MNIST_MINI/MNIST_TEST \
    --known-classes 0,1,2 \
    --unknown-classes 3,4 \
    --train-val-test-split 0.8,0.1,0.1 \
    --train-crop-size 32 \
    --test-resize-size 32 \
    --test-crop-size 32 \
    --hflip-prob 0.5 \
    --random-erase-prob 0.1 \
    --cutmix-alpha 1.0 \
    --batch-size 32 \
    --num-workers 4 \
    --epochs 30 \
    --early-stopping-epochs 6 \
    --model resnet34 \
    --label-smoothing 0.1 \
    --optimizer sgd \
    --lr 0.001 \
    --lr-min 0 \
    --momentum 0.9 \
    --weight-decay 0.00002 \
    --norm-weight-decay 0.0 \
    --lr-warmup-epochs 5 \
    --lr-warmup-decay 0.01 \
    --seed 42 \
    --device cuda \
    --output-dir runs
