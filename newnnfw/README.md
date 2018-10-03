# nnfw

A high-performance, on-device neural network inference framework

## Goal
This project _nnfw_ aims at providing a high-performance, on-device neural network (NN) inference
framework that performs inference of a given NN model on processors, such as CPU, GPU, or NPU, in
the target platform, such as Tizen and Smart Machine Platform (SMP).

## Project Documents
- [Roadmap](docs/roadmap.md)
- [SW Requirement Specification](docs/project/2018_requirement_specification.md)
- [SW High Level Design](docs/project/2018_high_level_design.md)

## Getting started
- For the contribution, please refer to our [contribution guide](docs/HowToContribute.md).
- You can also find how-to documents [HERE](docs/howto.md).

## Feature Request (NEW)

You can suggest development of nnfw's features that are not yet available.

The functions requested so far can be checked in the [popular feature request](https://github.sec.samsung.net/STAR/nnfw/issues?utf8=%E2%9C%93&q=is%3Aopen+is%3Aissue+label%3AFEATURE_REQUEST+sort%3Areactions-%2B1-desc) list.

- If the feature you want is on the list, :+1: to the body of the issue. The feature with the most 
:+1: is placed at the top of the list. When adding new features, we will prioritize them with this reference.
Of course, it is good to add an additional comment which describes your request in detail.

- For features not listed, [create a new issue](https://github.sec.samsung.net/STAR/nnfw/issues/new).
Sooner or later, the maintainer will tag the `FEATURE_REQUEST` label and appear on the list.

We expect most current feature requests to be focused on operator kernel implementations.
It is good to make a request, but it is better if you contribute by yourself. See the following guide, 
[How to Implement Operator Kernel](docs/HowToImplementOperatorKernel.md), for help.

We are looking forward to your participation.
Thank you in advance!
