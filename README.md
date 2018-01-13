# fpn.pytorch
Pytorch implementation of Feature Pyramid Network (FPN) for Object Detection

## Introduction

This project inherit the property of our [pytorch implementation of faster r-cnn](https://github.com/jwyang/faster-rcnn.pytorch). Hence, it also has the following unique features:

* **It is pure Pytorch code**. We convert all the numpy implementations to pytorch.

* **It supports trainig batchsize > 1**. We revise all the layers, including dataloader, rpn, roi-pooling, etc., to train with multiple images at each iteration.

* **It supports multiple GPUs**. We use a multiple GPU wrapper (nn.DataParallel here) to make it flexible to use one or more GPUs, as a merit of the above two features.

* **It supports three pooling methods**. We integrate three pooling methods: roi pooing, roi align and roi crop. Besides, we convert them to support multi-image batch training.

## Progress

Testing performance, will be finished soon!
