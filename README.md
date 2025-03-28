# Deep Research on the TOSA Architecture in Arm

## Executive Summary

This research document provides a comprehensive analysis of the Tensor Operator Set Architecture (TOSA) and its implementation in Arm hardware. TOSA is a standardized set of whole-tensor operations designed to enable consistent execution of machine learning workloads across diverse hardware platforms. Arm has adopted TOSA as a key component of its machine learning strategy, particularly in its Ethos Neural Processing Units (NPUs). This document explores TOSA's architecture, specifications, operators, and its practical implementation in Arm hardware, with a focus on the Ethos-U85 NPU.

## Table of Contents

1. [Introduction to TOSA](#introduction-to-tosa)
2. [TOSA Architecture Overview](#tosa-architecture-overview)
3. [TOSA Operators and Specifications](#tosa-operators-and-specifications)
4. [Arm's Implementation of TOSA](#arms-implementation-of-tosa)
5. [TOSA in Arm Ethos-U85 NPU](#tosa-in-arm-ethos-u85-npu)
6. [TOSA and the Arm ML Ecosystem](#tosa-and-the-arm-ml-ecosystem)
7. [Conclusion](#conclusion)
8. [References](#references)

## Introduction to TOSA

Tensor Operator Set Architecture (TOSA) is a specification for a set of whole-tensor operations commonly employed in Deep Neural Networks (DNNs). It was developed to address the fragmentation in machine learning frameworks and hardware/software inference platforms. TOSA provides a standardized layer between high-level machine learning frameworks (like TensorFlow and PyTorch) and the underlying hardware implementations.

The primary goal of TOSA is to enable a variety of implementations running on diverse processors, with consistent results across those implementations. This allows applications or frameworks targeting TOSA to be deployed on a wide range of different processors, including SIMD CPUs, GPUs, and custom hardware such as NPUs/TPUs, with defined accuracy and compatibility constraints.

## TOSA Architecture Overview

### Core Principles

TOSA is built on several key principles:

1. **Whole-Tensor Operations**: TOSA specifies operators only for whole tensors, not individual elements. This approach allows for a variety of implementations and optimizations.

2. **Hardware and Software Independence**: TOSA is designed to be independent of any specific hardware or software design, making it versatile across different platforms.

3. **Precision and Numerical Operation**: TOSA architectures precise functional descriptions of every operator, including numerical behavior for precision, saturation, scaling, etc., especially for quantized datatypes.

4. **Framework Agnosticism**: TOSA is not tied to any single high-level framework, compiler backend stack, or particular target.

5. **Minimal and Stable Operator Set**: TOSA defines a minimal and stable set of tensor-level operators to which higher-level operators can be lowered in a consistent way.

### Goals of TOSA

The primary goals of TOSA include:

- Providing a minimal and stable set of tensor-level operators to which machine learning framework operators can be reduced
- Supporting both quantized integer and floating-point content
- Offering precise functional descriptions of operator behavior
- Remaining agnostic to any single high-level framework, compiler backend, or target
- Enabling precise code construction for diverse targets (SIMD CPUs, GPUs, custom hardware)

### Stability and Consistency

TOSA serves as a standardized, stable layer between machine learning frameworks and inference platforms. This stability enables:

- Fast evolution of frameworks while stabilizing the platform below the layer
- A finite set of composable primitives that enable an infinite set of operators
- Guaranteed execution of ML models built using TOSA on any platform supporting TOSA

## TOSA Operators and Specifications

### Operator Categories

TOSA defines several categories of operators:

1. **Tensor Operators**: These form the core of TOSA and operate on whole tensors.
2. **Operator Graphs**: These define how operators connect to form a complete neural network.
3. **Constant Operators**: These define constant values used in the network.

### Operator Arguments

Operators in TOSA process input arguments to produce output arguments. Their behavior can be configured using attribute arguments. Arguments may have one of the following types:

- `tensor_t<element_type>`: Represents a tensor whose elements are of type `element_type`
- `tensor_list_t`: Represents a list of tensors
- `tosa_graph_t`: Represents a TOSA graph

Arguments belong to one of three categories:
- **Input**: Must be a tensor or list of tensors used to provide data read by the operation
- **Output**: Must be a tensor or list of tensors into which the data produced by the operation is written
- **Attribute**: Is constant, with its value always known at compilation time

### Supported Data Types

TOSA supports various number formats, including:

- Integer types (i8_t, i16_t, i32_t)
- Floating-point types (fp16_t, fp32_t)
- Boolean type (bool_t)
- Extended types through profiles (bf16_t, fp8e4m3_t, fp8e5m2_t)

### Example Operators

Here are some key TOSA operators:

1. **ARGMAX**: Returns the index with the largest value across a given axis of the input tensor.

2. **AVG_POOL2D**: Performs average pooling over the given input tensor with a sliding window.

3. **CONV2D**: Performs a 2D convolution with the given input tensor and filter weights.

4. **RESCALE**: Performs scaling operations, particularly important for quantized operations.

5. **MATMUL**: Performs matrix multiplication between two input tensors.

### Precision Requirements

TOSA specifies precise requirements for operator implementations:

- Integer results must be exact
- Floating-point operations have specific rules for handling NaN values
- Dot product operations have specific accuracy requirements

### Profiles and Levels

TOSA defines profiles and levels to specify implementation requirements:

- **Profiles**: Define sets of operations and data types that must be supported
- **Levels**: Define operator argument ranges that an implementation must support

## Arm's Implementation of TOSA

Arm has embraced TOSA as a key component of its machine learning strategy. The company is leading efforts to provide standardization in the ML space with TOSA, addressing fragmentation in ML frameworks and hardware/software inference platforms.

### Arm's TOSA Strategy

Arm's approach to TOSA includes:

1. **Open Standard Promotion**: Arm promotes TOSA as an open standard, with a license that grants rights to IP required to implement the specification.

2. **Integration with ML Frameworks**: Arm works to integrate TOSA with popular ML frameworks like PyTorch and TensorFlow.

3. **Hardware Implementation**: Arm implements TOSA in its neural processing units (NPUs), particularly the Ethos series.

4. **Software Toolchain Support**: Arm provides software tools that support TOSA, enabling efficient compilation and execution of neural networks.

### ExecuTorch and TOSA

Arm has worked with Meta to introduce support for Arm platforms in ExecuTorch, a new end-to-end solution for enabling on-device AI for PyTorch. This collaboration leverages TOSA as a standardized specification for machine learning operators.

The ExecuTorch and TOSA integration enables:

- Direct export of PyTorch models to Arm platforms
- Support for a wide range of neural networks
- Efficient execution on Arm CPUs, GPUs, and NPUs
- Quantization support for optimal performance

## TOSA in Arm Ethos-U85 NPU

The Arm Ethos-U85 NPU is a third-generation neural processing unit that explicitly targets TOSA architecture. It's designed to accelerate Edge AI inference in both constrained Cortex-M and Cortex-A based systems.

### Ethos-U85 and TOSA Integration

The Ethos-U85 NPU is designed with TOSA as a core architectural component:

1. **Native TOSA Support**: The NPU targets the Tensor Operator Set Architecture (TOSA) and TFLite integer quantized operations.

2. **TOSA Operator Implementation**: The NPU implements TOSA operators in hardware, including specific support for operators like the TOSA Rescale operator in its scaling unit.

3. **Transformer Support**: With native support for transformer networks through TOSA, the Ethos-U85 can handle advanced models like LLMs at the edge.

4. **Quantization Support**: The NPU supports both 8-bit and 16-bit integer quantized networks through TOSA.

### Technical Implementation

The Ethos-U85 NPU implements TOSA in several key components:

1. **Scaling Unit**: The scaling unit implements most of the activation data processing and supports the scaling defined by the TOSA Rescale operator, along with several extra rounding modes.

2. **MAC Unit**: The multiply-accumulate unit is designed to efficiently execute TOSA tensor operations.

3. **Command Stream**: Neural networks are compiled using an open-source compiler to produce a command stream that describes the steps necessary for the NPU to execute TOSA operators.

4. **Software Stack**: The NPU is supported by software that can lower neural networks to the TOSA integer profiles supported by the Arm Ethos-U software.

### Performance and Capabilities

The Ethos-U85 with TOSA support offers:

- Scalable performance from 128 to 2048 MAC units, providing up to 4 TOPs at 1 GHz
- 20% more energy efficiency than previous Ethos NPUs
- Support for transformer-based models at the edge
- Compatibility with both Cortex-M and Cortex-A based systems

## TOSA and the Arm ML Ecosystem

TOSA plays a central role in Arm's broader machine learning ecosystem, connecting various components and enabling efficient execution of neural networks.

### TOSA in the ML Toolchain

The TOSA architecture fits into Arm's ML toolchain in several ways:

1. **Framework Integration**: High-level frameworks like PyTorch and TensorFlow can be lowered to TOSA.

2. **Compiler Flow**: The Vela compiler has been enhanced with a TOSA front-end, making it possible to compile models for all products in the Ethos-U family.

3. **Runtime Support**: The ExecuTorch runtime has been extended to include support for the Ethos-U device driver, enabling efficient execution of TOSA operations.

4. **Deployment Flow**: A complete flow from PyTorch model to deployed application is enabled through TOSA, with tools for quantization, compilation, and runtime execution.

### Ecosystem Benefits

The adoption of TOSA in Arm's ecosystem provides several benefits:

1. **Unified Toolchain**: Partners can benefit from seamless migration and leverage investments in Arm-based machine learning tools.

2. **Standardization**: TOSA provides a standard that reduces fragmentation in the ML space.

3. **Performance Optimization**: The detailed functional and numerical description in TOSA enables precise code construction for diverse targets.

4. **Future-Proofing**: As a stable layer, TOSA allows frameworks to evolve rapidly while maintaining compatibility with existing hardware.

## Conclusion

The Tensor Operator Set Architecture (TOSA) represents a significant advancement in standardizing machine learning operations across diverse hardware platforms. Arm's adoption and implementation of TOSA, particularly in its Ethos-U85 NPU, demonstrates the architecture's practical value in enabling efficient execution of neural networks on edge devices.

TOSA addresses the fragmentation in machine learning frameworks and hardware/software inference platforms, providing a stable layer that allows for both rapid evolution of frameworks and consistent execution across different hardware. Its detailed specifications for operators, data types, and numerical behavior ensure precise and predictable results.

Arm's implementation of TOSA in the Ethos-U85 NPU showcases how this architecture can be leveraged to create high-performance, energy-efficient solutions for edge AI. The integration with ExecuTorch further extends TOSA's reach, enabling seamless deployment of PyTorch models on Arm platforms.

As machine learning continues to evolve and expand into more applications, standardized architectures like TOSA will play an increasingly important role in ensuring compatibility, performance, and efficiency across the diverse landscape of hardware and software platforms.

## References

1. MLPlatform.org. "Tensor Operator Set Architecture (TOSA)." https://www.mlplatform.org/tosa/

2. MLPlatform.org. "TOSA 1.0.0 draft specification." https://www.mlplatform.org/tosa/tosa_spec.html

3. Arm Community. "Tensor Operator Set Architecture (TOSA)." https://community.arm.com/cfs-file/__key/communityserver-blogs-components-weblogfiles/00-00-00-37-98/Eric-Kunze-_2D00_-ODIW-2021-_2D00_.pdf

4. MLIR - LLVM. "Tensor Operator Set Architecture (TOSA) Dialect." https://mlir.llvm.org/docs/Dialects/TOSA/

5. Arm Developer. "Arm Ethos-U85 NPU Technical Overview." https://developer.arm.com/documentation/102684/latest/Functional-description-of-the-Arm--Ethos-U85--NPU/Activation-output-unit/Scaling-unit

6. Arm Developer. "Arm Ethos-U85 NPU Technical Reference Manual." https://developer.arm.com/documentation/102685/latest/Description-of-the-Arm--Ethos-U85--NPU--

7. Arm.com. "Ethos-U85 | Advanced NPU with Scalable Performance and Efficiency." https://www.arm.com/products/silicon-ip-cpu/ethos/ethos-u85

8. PyTorch.org. "Getting started with PyTorch, ExecuTorch, and Ethos-U85 in three easy steps." https://pytorch.org/blog/pt-executorch-ethos-u85/

9. Arm Community Blogs. "ExecuTorch and TOSA enabling PyTorch on Arm platforms." https://community.arm.com/arm-community-blogs/b/ai-blog/posts/executorch-and-tosa-enabling-pytorch-on-arm-platforms
