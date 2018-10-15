# Software Requirement Specification

## Background
Artificial intelligence (AI) techniques are getting popular and utilized in various products and
services.  While the cloud-based AI techniques have been used to perform compute/memory intensive
inferences because of the powerful servers on cloud, on-device AI technologies are recently drawing
attention from the mobile industry for response time reduction, privacy protection, and
connection-less AI service.  Big mobile players, such as Google, Apple, and Huawei, are investing
their research effort on the on-device AI technologies and already announced hardware and software
on-device AI solutions.  Samsung is not leading this trend currently, but since on-device AI area is
just started and still in the initial state, there are still opportunities and possibilities to
reduce the gap between pioneer companies and Samsung.  We believe on-device AI will become a key
differentiator for mobile phone, TV, and other home appliances, and thus developing on-device AI
software stack is of paramount importance in order to take leadership in the on-device AI
technology.

Although the vision of on-device AI is promising, enabling on-device AI involves unique technical
challenges compared to traditional cloud-based approach.  This is because on-device AI tries to
conduct inference tasks solely on device without connecting to cloud resources.  Specifically,
hardware resources on device, such as processor performance, memory capacity, and power budget, are
very scarce and limit the compute capability, which is typically required to execute complicated
neural network (NN) models.  For example, in one product requirement, a mobile device should consume
less than 1.2W and could use at most 2W only for 10 minutes due to thermal issue.  Next, on-device
AI software stack needs to support diverse device environments, since embedded platforms may consist
of heterogeneous compute devices, such as CPU, GPU, DSP, or neural processing unit (NPU), and use
different OS platforms, such as Tizen, Android, or Smart Machine OS.

To tackle the challenges above and to have the leadership on on-device AI technology, this project,
as the first step, aims at developing a neural network inference framework specialized and optimized
for on-device AI.


## Product Context

This project _nnfw_ aims at providing a high-performance, on-device neural network (NN) inference
framework that performs inference of a given NN model on processors, such as CPU, GPU, or NPU, in
the target platform, such as Tizen and Smart Machine Platform (SMP).

### Expected Value

We expect the following would be possible with _nnfw_:

- To improve user experience by reducing the service response time
- To provide AI services without network connection while achieving similar performance
- To protect personal information and company confidential by limiting data transfer to the network


### Success Criteria

The goals of this project are:

- To support all 50 TensorFlow (TF) Lite operations on ARM CPU and GPU
- To support all 29 operations of Android Neural Network (NN) API on ARM CPU and GPU
- To support InceptionV3 and MobileNet, written in TF Lite model format, on ARM CPU and GPU


### Target

_nnfw_ targets two platforms with two target devices:

- ODroid XU4 running Tizen 5.0
- MV8890 running Smart Machine Platform 1.0


### Product Roadmap

- March: Set up milestones, tasks, workgroups, initial code structure, and build/test infra
- April: Run InceptionV3 using ARM Compute Library (ACL) on ODroid XU4 running Tizen
- May: Run MobileNet on Tizen / Tizen M1 release
- June: Run ADAS models on Tizen
- July: STAR Platform preview release
- October: Tizen M2 release / SMP v1.0 release / STAR Platform v1.0 release


## Requirements

### Functionality Requirements

_nnfw_ has the following functionality requirements:

1. Run InceptionV3 on Tizen
   - Description
      - Support InceptionV3, written in TF Lite model format, on Tizen
      - Run on ARM CPU and GPU
   - Validation
      - Run the test code that executes InceptionV3 on Tizen CPU
      - Run the test code that executes InceptionV3 on Tizen GPU
      - Compare the results of test codes with that using the TF Lite interpreter
1. Run MobileNet on Tizen
   - Description
      - Support MobileNet, written in TF Lite model format, on Tizen
      - Run on ARM CPU and GPU
   - Validation
      - Run the test code that executes MobileNet on Tizen CPU
      - Run the test code that executes MobileNet on Tizen GPU
      - Compare the results of test codes with that using the TF Lite interpreter
1. Support 50 TF Lite operations and 29 NN API operations
   - Description
      - Support 50 TF Lite operations on Tizen for ARM CPU and GPU
      - Support 50 TF Lite operations on SMP for ARM CPU and GPU
      - Support 29 NN API operations on Tizen for ARM CPU and GPU
      - Support 29 NN API operations on SMP for ARM CPU and GPU
   - Validation
      - Run the test code for operations on Tizen CPU
      - Run the test code for operations on Tizen GPU
      - Run the test code for operations on SMP CPU
      - Run the test code for operations on SMP GPU
      - Compare the results of test codes with that using the TF Lite interpreter


### Non-Functionality Requirements

_nnfw_ does not have non-functionality requirements.
