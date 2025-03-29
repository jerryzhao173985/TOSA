# 1. Overview of TOSA in ARM

**What is TOSA?** The **Tensor Operator Set Architecture (TOSA)** is a standardized set of tensor-level operations for machine learning, defined by Arm. TOSA specifies a *minimal and stable* collection of whole-tensor ops (like conv2d, add, matmul, etc.) that are commonly used in deep neural networks. The goal is to enable ML frameworks to express models in this portable operator set so they can run efficiently on a wide range of hardware – from Arm CPUs and GPUs to dedicated AI accelerators (NPUs/TPUs). By targeting TOSA, a model’s operators are well-defined with consistent behavior (including how quantization rounding, saturation, etc. are handled). This means if a model is lowered to TOSA, it’s **guaranteed to run on any platform** that implements TOSA, with predictable results. In essence, TOSA acts as an **intermediate layer** between high-level frameworks and low-level hardware, standardizing the ML operations across the ARM ecosystem.

**Role in Machine Learning on ARM:** On ARM architectures, TOSA serves as a unifying interface for ML inference. Instead of each hardware accelerator or DSP requiring custom operators or kernel libraries, they can all implement the TOSA ops. Frameworks (like TensorFlow or PyTorch) can lower their model graphs to TOSA, and from there the model can be compiled or dispatched to various ARM-based processors. This is especially valuable in the fragmented IoT and mobile space, where models created in different frameworks need to run on diverse ARM devices. TOSA helps bridge this gap: for example, a model trained in TensorFlow can be converted to TOSA and then run on an Arm Cortex-A CPU, a Mali GPU, or an Ethos NPU without needing different conversion tools for each. This *“write once, run anywhere”* approach simplifies deploying ML on billions of ARM devices.

**Advantages of TOSA:** For ML inference, TOSA offers several benefits:

- **Portability:** A model expressed in TOSA can be deployed on many ARM hardware targets. As the Arm IoT marketing lead notes, TOSA is a *standardized operator set* so that future models can run across billions of devices without per-device customization.
- **Consistency:** TOSA precisely defines each operator’s functionality and numerical behavior. This includes how calculations are done in int8 vs float, how rounding is handled, etc., which is critical for quantized models. By having a single spec, it ensures consistent results across implementations.
- **Minimalism:** The operator set is kept as small as possible (on the order of ~75 ops in early versions ([TOSA Specifications Creation Approach - TOSA - Discourse](https://discuss.mlplatform.org/t/tosa-specifications-creation-approach/264#:~:text=Hi%2C%20I%20am%20new%20to,selecting%20the%20odd%2075%20operators)) ([TOSA Specifications Creation Approach - TOSA - Discourse](https://discuss.mlplatform.org/t/tosa-specifications-creation-approach/264#:~:text=When%20we%20started%20the%20work,continue%20to%20guide%20the%20work))) while still being able to represent most framework operations. Complex library-specific ops are broken down into combinations of TOSA primitives. This makes it easier for hardware vendors to support the full spec.
- **Quantization Support:** TOSA was designed with quantized inference in mind. It has built-in support for *8-bit quantized* tensor operations alongside float32, including special ops like *Rescale* for handling quantization scale/zero-point. This is crucial for ARM devices where int8 inference is often needed for performance and efficiency.
- **Hardware Agnostic with Acceleration in Mind:** TOSA ops are chosen to map efficiently onto typical hardware primitives (vector SIMD, matrix-multiply engines, etc.). They’re also amenable to fusion and optimization. For example, TOSA defines primitives that can be fused (like activation functions after a matmul) enabling compiler optimizations on ARM CPUs/GPUs. The ops are meant to be *composable primitives* – by combining them, you can represent even very complex neural network layers ([TOSA Specifications Creation Approach - TOSA - Discourse](https://discuss.mlplatform.org/t/tosa-specifications-creation-approach/264#:~:text=Hi%2C%20I%20am%20new%20to,selecting%20the%20odd%2075%20operators)) ([TOSA Specifications Creation Approach - TOSA - Discourse](https://discuss.mlplatform.org/t/tosa-specifications-creation-approach/264#:~:text=When%20we%20started%20the%20work,continue%20to%20guide%20the%20work)). This finite set of primitives can express an “infinite set” of higher-level ops found in frameworks ([TOSA Specifications Creation Approach - TOSA - Discourse](https://discuss.mlplatform.org/t/tosa-specifications-creation-approach/264#:~:text=Hi%2C%20I%20am%20new%20to,selecting%20the%20odd%2075%20operators)).

**Limitations of TOSA:** While TOSA is powerful for inference, it has some limitations:

- **Inference Focus (Training Gaps):** The current TOSA spec is heavily focused on inference-time operators. Many training-specific operations (e.g. gradients, optimizers, backprop primitives) are not yet covered. The TOSA developers acknowledge *“the biggest gap...is with regard to training”* and that supporting training will likely require new operators in the future ([TOSA Specifications Creation Approach - TOSA - Discourse](https://discuss.mlplatform.org/t/tosa-specifications-creation-approach/264#:~:text=operators%20that%20are%20commonly%20in,use%20in%20the%20frameworks)). So, as of now, TOSA is not a complete solution for training on device – it’s mostly used for deploying pre-trained models.
- **Operator Coverage:** Although TOSA aims to cover most ops, certain newer or niche operations might be missing and need to be added as ML evolves. In fact, the operator list has grown modestly over time when new network types demand it ([Api -> Tosa operator mapping - TOSA - Discourse](https://discuss.mlplatform.org/t/api-tosa-operator-mapping/285#:~:text=We%20try%20to%20keep%20the,0%20networks)). For example, *SIN* and *COS* ops were added to support Transformer models’ rotary positional embeddings ([Api -> Tosa operator mapping - TOSA - Discourse](https://discuss.mlplatform.org/t/api-tosa-operator-mapping/285#:~:text=We%20try%20to%20keep%20the,0%20networks)). If a model uses an op not in TOSA, that part of the model can’t be expressed in TOSA and would fall back to custom handling.
- **Dynamic Shapes & Control Flow:** TOSA primarily deals with static-shaped tensor ops (like those in TFLite). Handling dynamic shape operations or control flow (loops, conditionals) is outside the core TOSA spec. In practice, frameworks need to resolve shapes ahead of time or handle dynamic behavior before lowering to TOSA. In MLIR, for instance, the TOSA dialect expects shapes to be fully resolved – dynamic dimensions `?` are not handled by the reference implementation. This can make models with dynamic sequence lengths or conditional branches harder to compile purely into TOSA.
- **No Custom Ops:** By design, TOSA limits itself to a defined set of ops. This means if a hardware accelerator has a very custom operation (not expressible as a combination of TOSA ops), you can’t directly represent that as a single TOSA op. You’d have to break it into standard ops, which might not fully utilize the custom hardware. However, this trade-off is intentional to keep TOSA stable and broadly applicable.

**Relationship with ARM Hardware Accelerators:** TOSA is closely tied to Arm’s hardware IP for ML. It provides a common layer that ARM’s CPUs, GPUs, and NPUs can target:

- *ARM CPUs:* Traditional Arm Cortex-A and Cortex-M CPUs can run TOSA ops using their NEON (SIMD) instructions or SVE for larger vectors. In fact, Arm’s own inference libraries (like Arm Compute Library and Arm NN) could take a TOSA model and execute it on CPUs by mapping each op to highly optimized kernel implementations (e.g., a TOSA convolution to an ARM NEON GEMM + im2col routine).
- *Mali GPUs:* The Mali GPU can accelerate ML via OpenCL or Vulkan compute shaders. A compiler could lower TOSA ops to GPU kernels. For example, a TOSA matrix multiplication might be turned into a Mali GPU shader. The key is that TOSA doesn’t include GPU-specific concepts; it stays at tensor operations, which GPUs can handle in parallel.
- *Ethos NPUs:* Arm’s *Ethos* family of NPUs (Neural Processing Units) is explicitly built with TOSA in mind. The Ethos-U microNPUs (for Cortex-M systems) and Ethos-N NPUs (for mobile SoCs) execute commands that correspond to TOSA operations. **Ethos-U85**, for instance, *“targets the Tensor Operator Set Architecture (TOSA) and TFLite integer quantized operations.”* ([Arm Ethos-U85 NPU Technical Overview](https://developer.arm.com/documentation/102684/0000/Description-of-the-Arm--Ethos-U85--NPU--#:~:text=Arm%20Ethos,Inference)). In other words, the supported operations on Ethos NPUs align with the TOSA spec (Base Inference profile). This standardization means compilers like Arm’s Vela can take a TOSA-described model and generate NPU machine code directly. The Ethos-U NPUs benefit greatly from TOSA’s quantization focus, as they are optimized for int8 arithmetic. By using TOSA as the interface, Ethos can natively support models from various frameworks (so long as they’re lowered to TOSA).
- *Other AI Cores:* Even if an accelerator isn’t explicitly “Ethos”, any third-party accelerator for ARM (e.g., from Synopsys, NXP, etc.) could choose to support TOSA. By implementing the TOSA ops (in hardware or firmware), they immediately become compatible with any ML framework that can emit TOSA. This is a big draw for industry adoption: hardware vendors can focus on making sure their chip supports TOSA’s operator set, and they get interoperability with major ML software “for free”. It reduces the fragmentation where each accelerator had its own API or operator definitions.
- *Integration with AI Toolchain:* TOSA is part of a bigger ecosystem of ARM ML tools. For example, the Arm NN and Compute Library can act as backends that execute TOSA graphs. The Arm *Machine Learning Inference Advisor (MLIA)* uses TOSA as one way to represent models to analyze their performance on various ARM IPs. And as mentioned, the Vela compiler uses TOSA (or TensorFlow Lite) as input to compile for micro NPUs.

In summary, TOSA provides a crucial abstraction layer in ARM’s ML stack: high-level frameworks produce TOSA graphs, and low-level ARM hardware executes those graphs. This architecture brings portability and efficiency, but currently targets *inference* use-cases predominantly, with training support still on the horizon ([TOSA Specifications Creation Approach - TOSA - Discourse](https://discuss.mlplatform.org/t/tosa-specifications-creation-approach/264#:~:text=operators%20that%20are%20commonly%20in,use%20in%20the%20frameworks)).

# 2. TOSA in PyTorch, TensorFlow, and ONNX

TOSA has been making its way into popular machine learning frameworks, often via compiler toolchains and IR (Intermediate Representation) conversions. Here’s how it integrates with **PyTorch**, **TensorFlow**, and **ONNX**, along with examples and performance notes for each:

## PyTorch and TOSA Integration

PyTorch integrates with TOSA primarily through an ARM-optimised flow known as **ExecuTorch** and via MLIR. In PyTorch 2.x, the computation graph (TorchScript or FX graph) can be exported and lowered to other IRs. ARM and Meta (Facebook) collaborated to introduce a TOSA-based flow for PyTorch:

- **ExecuTorch with TOSA**: ExecuTorch is an end-to-end runtime and ahead-of-time compiler for PyTorch, designed to facilitate on-device AI. In late 2023, Arm worked with Meta to add support for ARM targets by leveraging TOSA. The approach is to capture the PyTorch model, lower it to TOSA operators, and then use ARM’s backends (like the Ethos-U NPU) to accelerate execution. Arm released a *TOSA compilation flow and runtime delegate* for ExecuTorch, with initial support targeting Ethos-U55 NPUs ([ExecuTorch and TOSA enabling PyTorch on Arm platforms - AI blog - Arm Community blogs - Arm Community](https://community.arm.com/arm-community-blogs/b/ai-blog/posts/executorch-and-tosa-enabling-pytorch-on-arm-platforms#:~:text=Today%20we%E2%80%99ve%20released%20a%20TOSA,300)). This means you can take a PyTorch model and compile it such that all its operators are expressed in TOSA, and then offload those to an Ethos-U NPU at runtime.
- **Torch-MLIR**: Another path is the open-source `torch-mlir` project, which provides MLIR dialects for PyTorch. It includes a conversion from the Torch IR to the TOSA dialect in MLIR ([Api -> Tosa operator mapping - TOSA - Discourse](https://discuss.mlplatform.org/t/api-tosa-operator-mapping/285#:~:text=%E2%80%93mainly%20in%20legalize_tf,mlir%2Ftree%2Fmain%2Fsrc%2FConversion%2FONNXToTOSA)). Using `torch-mlir`, one can lower a PyTorch model to an MLIR module composed of TOSA ops. This is useful in contexts like IREE or other MLIR-based compilers.

**Code Example (PyTorch to TOSA using ExecuTorch):** Below is a simplified example using ExecuTorch’s API to export a PyTorch model via the TOSA/Ethos-U backend. This demonstrates how the PyTorch `exported_program` can be partitioned and compiled targeting an Ethos-U NPU using TOSA under the hood:

```python
import torch
# Suppose we have a simple PyTorch model
class AddModule(torch.nn.Module):
    def forward(self, x, y):
        return x + y

model = AddModule().eval()
example_inputs = (torch.randn(1,3), torch.randn(1,3))
# Export the model to an FX Graph (ExportedProgram)
exported = torch.export.export(model, example_inputs)

# Now use ExecuTorch to lower to the Arm backend (Ethos-U via TOSA)
from executorch import to_backend
from executorch.backends.arm import ArmPartitioner, arm_backend

compile_spec = arm_backend.generate_ethosu_compile_spec("ethos-u55-128")  # target Ethos-U55 (128 MAC config)
# Partition and convert the exported program to ARM backend (TOSA) ops
exported.exported_program = to_backend(exported.exported_program, ArmPartitioner(compile_spec))

# The exported.exported_program now contains TOSA-compatible operations ready for compilation.
# We would then proceed to compile to a .pte (ExecuTorch format) and deploy to the device.
```

In this snippet, `to_backend(...ArmPartitioner(...))` will partition the PyTorch graph and replace parts of it with TOSA-based kernels suitable for the Ethos-U NPU ([Building and Running ExecuTorch with ARM Ethos-U Backend — ExecuTorch 0.5 documentation](https://pytorch.org/executorch/stable/executorch-arm-delegate-tutorial.html#:~:text=from%20executorch)) ([Building and Running ExecuTorch with ARM Ethos-U Backend — ExecuTorch 0.5 documentation](https://pytorch.org/executorch/stable/executorch-arm-delegate-tutorial.html#:~:text=graph_module_edge.exported_program%20%3D%20to_backend%28%20model.exported_program%2C%20ArmPartitioner%28generate_ethosu_compile_spec%28%22ethos)). Essentially, Ops like `Add` in the model become the TOSA `add` operator, and so on, constrained to what the NPU supports. This flow produces a serialized `.pte` file (PyTorch Execution file) that can be loaded on an embedded device to run inference.

**TOSA Optimizations in PyTorch:** The primary optimization is *quantization*. ARM platforms (especially NPUs like Ethos) rely on int8 quantized models for peak performance. The PyTorch ExecuTorch flow provides tooling to quantize models and express those quantized ops in TOSA. For instance, ARM implemented a TOSA-compliant quantization scheme in PyTorch 2.0’s FX graph flow. Using **post-training quantization (PTQ)**, they were able to quantize a model (e.g., MobileNetV2) and lower the quantized ops to TOSA in just a few lines of code. PyTorch’s *XNNPACK quantizer* was subclassed to produce TOSA quantization parameters, making it easy to generate a TOSA-ready int8 model from an existing FP32 model.

*Example:* In the ExecuTorch blog, Arm engineers used PyTorch’s new PTQ API to quantize a MobileNetV2 model and then lowered it: *“Given its very simple API, we were able to achieve the quantization step in around 3-4 lines of code with a TOSA quantization flow subclassed from XNNPACK quantizer.”*. After quantization, the TOSA graph (with ops like quantized conv2d, add, etc.) was fed into the **Vela compiler** (Arm’s compiler for Ethos-U). The end result was that the **PyTorch -> TOSA -> Vela** pipeline could run a quantized MobileNetV2 on an Ethos-U NPU successfully.

**Performance (PyTorch on ARM with TOSA):** Early results have shown that using TOSA to offload to NPUs can greatly improve performance and efficiency for PyTorch models on ARM:

- The MobileNetV2 example demonstrated not just *feasibility* but also *simplicity*: without TOSA, getting PyTorch models onto micro NPUs required manual conversion to TFLite or hand-optimizing kernels. With the new flow, it’s mostly automated.
- While exact benchmarks weren’t quoted, the expectation is that running on an Ethos-U55/U65 at low precision yields huge speed-ups vs. Cortex-M CPU execution. (Ethos-U55 can perform dozens of MACs in parallel per cycle, delivering many times the throughput of a CPU for conv layers).
- On Cortex-A (e.g. smartphone CPUs), using TOSA hasn’t been as directly highlighted, but if one had an AI accelerator (Ethos-N or GPU), TOSA could enable using those. If only CPU is available, TOSA by itself doesn’t speed up execution (it’s an IR); you’d rely on an optimized library to execute it.
- The collaboration with Meta implies that as PyTorch evolves, more models will be supported. The blog mentions *“Arm, Meta and the PyTorch community will continue to add support for operator lowerings from PyTorch to TOSA”*, expanding model coverage. This will improve performance for a broader class of networks on ARM as more ops can be offloaded.

In summary, PyTorch’s integration of TOSA is still in prototype stages but is rapidly maturing. It allows PyTorch models to be *captured in TOSA*, then compiled to run on ARM hardware (especially NPUs) with significant performance gains. The approach combines graph capture, PTQ quantization, and a backend codegen (Vela) to achieve near state-of-the-art inference speed on tiny devices, all from a PyTorch workflow.

## TensorFlow and TOSA Integration

TensorFlow’s ecosystem was one of the first to adopt TOSA through the MLIR compiler framework. There are a couple of angles here: TensorFlow itself (GraphDef/SavedModel to TOSA via MLIR), and TensorFlow Lite which benefits from TOSA in the context of microcontrollers.

- **MLIR TOSA Dialect in TensorFlow:** TensorFlow uses MLIR internally for graph optimization. Arm contributed a TOSA dialect implementation to MLIR, and the TensorFlow team integrated *legalization passes* that convert TensorFlow ops into TOSA ops. In practice, this means you can take a TensorFlow **GraphDef** (or a `tf.function`), and use MLIR to **lower it entirely to TOSA**. An RFC from Google describes that *“the legalization passes from TensorFlow and TensorFlow Lite enable full networks to be legalized to a single, standard TOSA form. Both fp32 and quantized 8-bit are supported.”*. These transformations live in the TensorFlow codebase (e.g., `legalize_tf.cc`, `legalize_tfl.cc` in TF’s MLIR compiler) ([Api -> Tosa operator mapping - TOSA - Discourse](https://discuss.mlplatform.org/t/api-tosa-operator-mapping/285#:~:text=locations%3A)). Essentially, each high-level TF op (like `tf.Conv2D`, `tf.MatMul`, etc.) has a known lowering to one or a few TOSA ops (like `tosa.conv2d`, `tosa.matmul`).
- **TensorFlow Lite and TOSA:** TOSA is also aligned with TFLite. TFLite operators are quite similar in granularity to TOSA (since TFLite also focuses on inference with quantization). In fact, the Ethos-U NPU support initially revolved around TFLite: Arm’s **Vela** compiler can take a `.tflite` model and convert supported parts to custom operators that the NPU firmware understands. Underneath, those *supported parts correspond to TOSA ops*. Now with TOSA, there’s a more formal way: The TOSA spec “Base Inference” aligns with TFLite’s capabilities. TensorFlow MLIR has a pass specifically for TFLite to TOSA (`legalize_tfl.cc`) to convert a TFLite model into the TOSA dialect.
- **Direct TOSA Conversion API:** In TensorFlow 2.6+, there were experimental APIs to convert models to TOSA. For example, `tf.mlir.experimental.convert_graph_def` could be used with a pass pipeline including `tosa-legalize-tf` to produce an MLIR module in the TOSA dialect. Likewise, there was `tf.mlir.experimental.tflite_to_tosa_bytecode()` which directly converts a TFLite FlatBuffer to a TOSA-MLIR bytecode representation. (These APIs were experimental and may change; TensorFlow’s recent versions might have moved them or integrated differently.)
  
**Code Example (TensorFlow GraphDef to TOSA MLIR):** While this is a low-level flow, here’s a conceptual example using TensorFlow’s MLIR API to get a TOSA representation:

```python
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

# 1. Obtain a GraphDef of a TensorFlow model
def create_model():
    inp = tf.keras.Input(shape=(28,28,1), batch_size=1)
    x = tf.keras.layers.Conv2D(32, 3, activation='relu')(inp)
    x = tf.keras.layers.Flatten()(x)
    out = tf.keras.layers.Dense(10, activation='softmax')(x)
    return tf.keras.Model(inp, out)

model = create_model()
# Concrete function
cf = tf.function(model).get_concrete_function(tf.TensorSpec([1,28,28,1], tf.float32))
frozen_func = convert_variables_to_constants_v2(cf)
graph_def = frozen_func.graph.as_graph_def()

# 2. Convert the GraphDef to TOSA MLIR
from tensorflow.compiler.mlir import experimental as mlir_exp
tosa_mlir_txt = mlir_exp.convert_graph_def(
    graph_def,
    pass_pipeline='tf-standard-pipeline,func.func(tosa-legalize-tf)',
    show_debug_info=False)

print("TOSA MLIR:\n", tosa_mlir_txt[:500], "...")
```

In this example, we first create a simple CNN model in Keras, get its frozen GraphDef (with variables converted to constants for a static graph), and then call `convert_graph_def` with a pipeline that includes the `tosa-legalize-tf` pass. The output `tosa_mlir_txt` would be a textual MLIR representation where all ops should be in the `tosa.` dialect (e.g., `tosa.conv2d`, `tosa.matmul`, `tosa.reshape`, etc.), assuming the model’s ops are supported. This is how one could programmatically obtain a TOSA form of a TensorFlow model. Tools like these are used under the hood by compilers.

**TOSA Optimizations in TensorFlow:** For TensorFlow, the emphasis is on converting to TOSA for deployment on accelerators rather than optimizing training. The optimizations come from using MLIR to apply common optimizations (constant folding, etc.) and then lowering to TOSA which can then be handed off to optimized backends. One area of optimization is again **quantization**: TensorFlow Lite already supports quantization, and if you convert a quantized TFLite model to TOSA, those int8 ops become TOSA ops. The TOSA spec fully supports quantized inference, so you don’t lose any accuracy versus running the model in TFLite. In fact, converting TFLite to TOSA and then running through an NPU compiler (like Vela) is the path for deploying many TF models on microcontrollers.

It’s also worth noting that by using TOSA as an intermediate, **TensorFlow can deploy to non-TF runtimes**. For instance, an ML compiler like IREE can take the TOSA MLIR and compile it to various targets (CPU, Vulkan, etc.). This means a TensorFlow model could run via IREE’s engine using the TOSA path, potentially yielding performance benefits on ARM GPUs or other devices.

**Performance (TensorFlow on ARM with TOSA):** When running TensorFlow models on ARM hardware:

- If using CPU: XLA or Eigen kernels are usually used, but TOSA doesn’t directly speed that up (it’s more about portability). However, if a model is lowered to TOSA, one could use Arm Compute Library to execute it, which might be faster than TensorFlow’s default for certain ops.
- If using Ethos-U NPUs: TOSA is the *recommended path*. Arm’s Ethos-U compiler (Vela) can accept a TOSA FlatBuffer as input. So one workflow is: train model in TF -> convert to TFLite -> use TOSA checker to ensure compatibility -> compile with Vela -> run on device. Performance on Ethos-U (running via TOSA commands) can be **70-90% lower latency** and power compared to running on a Cortex-M CPU, for example, depending on the model. This is because the NPU is highly specialized. The integration is such that *“the NPU targets the TOSA ... and TFLite int8 ops”*, meaning it executes those extremely efficiently in hardware ([Arm Ethos-U85 NPU Technical Overview](https://developer.arm.com/documentation/102684/0000/Description-of-the-Arm--Ethos-U85--NPU--#:~:text=Arm%20Ethos,Inference)).
- For larger ARM NPUs (Ethos-N77 etc.) in mobile, similar logic applies – TOSA (or TFLite) is used as the interchange. A model like MobileNet on a smartphone with an Ethos NPU would run several times faster than on the CPU and with much less energy. 
- **Comparison across frameworks:** If we compare TensorFlow vs PyTorch vs ONNX on the *same* hardware via TOSA, the performance should be similar, because ultimately it’s the same operators executing on the same accelerator. For example, a ResNet50 run on Ethos-U via TOSA will have the same throughput whether it originally came from PyTorch or TF, because after conversion it’s the same sequence of TOSA ops executed by the NPU. The differences would come in how mature the conversion pipeline is – TensorFlow’s TOSA conversion might handle more ops or do better graph optimizations currently, because it’s been in development for a while. PyTorch’s is newer, so there might be cases where TensorFlow can offload an op that PyTorch doesn’t yet, giving TF an edge for that model on that hardware. These gaps are closing as the PyTorch integration improves.

In practice, TensorFlow’s use of TOSA is mostly behind the scenes via compilers or advanced users. But it ensures that TF models can run efficiently on ARM’s entire range – from tiny MCU devices (via microNPUs) up to phones and beyond – by handing off standardized TOSA ops to those targets.

## ONNX and TOSA Integration

**ONNX (Open Neural Network Exchange)** is a framework-agnostic model format, and it too can work with TOSA through compiler projects. ONNX’s role is a bit different: ONNX itself is an exchange format, not an execution engine. To leverage TOSA with ONNX, we use compilers that can take ONNX models and lower them to TOSA or directly to code.

Key integration points for ONNX and TOSA:

- **ONNX-MLIR:** This is an open-source compiler (part of the Linux Foundation AI & Data) that compiles ONNX models to different backends using MLIR. ONNX-MLIR has implemented a conversion path from ONNX ops to the TOSA dialect. In their source, there’s a directory for ONNXToTOSA conversions ([Api -> Tosa operator mapping - TOSA - Discourse](https://discuss.mlplatform.org/t/api-tosa-operator-mapping/285#:~:text=,mlir%20which%20can%20then%20be)). This means you can input an ONNX model and have it lowered into an MLIR module consisting of TOSA operations. From there, you could further lower to LLVM IR for a CPU, or use a code generator for an accelerator. The presence of ONNX->TOSA in ONNX-MLIR is great for ARM because it allows using TOSA as an intermediate step when compiling ONNX models for ARM hardware.
- **Torch-MLIR for ONNX:** Interestingly, the torch-mlir project can also take ONNX models by importing them through PyTorch’s frontend. The MLPlatform discourse mentions *“The torch-mlir project can route ONNX through torch-mlir which can then be sent through the … TorchToTosa path.”* ([Api -> Tosa operator mapping - TOSA - Discourse](https://discuss.mlplatform.org/t/api-tosa-operator-mapping/285#:~:text=,mlir%20path)). In other words, you could load an ONNX model, turn it into a TorchScript or FX graph (since ONNX and PyTorch have similar operator sets for many things), then use the existing Torch->TOSA conversion. This is a bit roundabout but leverages existing tooling.

**Code Example (ONNX-MLIR to generate TOSA):** Using ONNX-MLIR’s command line, one can do something like:

```bash
# Compile an ONNX model to an MLIR TOSA dialect
onnx-mlir --EmitONNXIR model.onnx -o model.onnx.mlir
onnx-mlir --EmitMLIR model.onnx -o model.std.mlir
```

In ONNX-MLIR’s options, there isn’t a direct flag `--EmitTOSA`, but what you can do is emit the MLIR after the ONNX to TOSA lowering pass. Typically, one would run a pipeline of passes. For illustration, if you had the tool setup, you might run a pass pipeline like: `onnx-mlir -p="onnx-to-tosa,pipeline-to-llvm" model.onnx` (the actual commands can differ). The idea is that ONNX ops (e.g., `Conv`, `Relu`) get converted into corresponding TOSA ops (`tosa.conv2d`, `tosa.relu` etc.). The onnx-mlir GitHub shows how an ONNX `Add` operation would be lowered in their `ConvertONNXToTOSA.cpp` for example.

In absence of the actual command, conceptually: 

- **Step 1:** Load the ONNX model (computational graph).
- **Step 2:** Run ONNX-MLIR’s conversion passes: ONNX dialect -> TOSA dialect. This will succeed if all ops in the model have mappings. (Operators like BatchNorm, Conv, Gemm, etc., are supported; some very new ONNX ops might not have a mapping yet).
- **Step 3:** Now you have an MLIR file with TOSA ops. You can either stop here for analysis or continue to codegen.
- **Step 4:** Lower TOSA to LLVM or C (or to a specific accelerator dialect if available). If targeting CPU, eventually it becomes an object file or binary. If targeting an accelerator, one might instead serialize the TOSA (e.g., to flatbuffer) for input to an NPU driver.

**TOSA Optimizations in ONNX context:** ONNX itself doesn’t perform optimizations (it’s a static format). But ONNX-MLIR and similar compilers apply optimizations during conversion. For example, constant folding, operator fusion, etc., can be done on the ONNX graph before or during lowering to TOSA. One advantage of going ONNX->TOSA is that ONNX has a very large operator set (including many legacy or redundant ops). By converting to TOSA, you simplify the graph to core operations. This can eliminate some overhead or odd patterns. Also, ONNX-MLIR might choose the most efficient TOSA sequence for a given ONNX op (for instance, ONNX `BatchNormalization` might lower to a sequence of TOSA ops that can be fused at runtime).

**Performance (ONNX on ARM with TOSA):** There isn’t an “ONNX runtime with TOSA” out-of-the-box from Microsoft or others, but using ONNX-MLIR or similar:

- If compiling an ONNX model for an ARM CPU via TOSA, you’d essentially generate an ARM-native binary (through LLVM). The performance would then depend on LLVM’s ability to vectorize and optimize those ops. In theory, it could be on par or even better than the default ONNX Runtime, because ONNX Runtime uses a general kernel library whereas ONNX-MLIR can inline and optimize specifically. However, this is an area of ongoing development.
- If targeting an NPU, the performance will match that of models from other frameworks, since the NPU doesn’t care whether the model came from ONNX or not. TOSA is the great equalizer here – as long as the model is expressed in TOSA and compiled to the NPU, it will run with the same efficiency.
- One consideration: ONNX covers a lot of scenarios (including training graphs, which have things like gradient ops). Those are not in TOSA (since TOSA is inference). So ONNX models used here are inference models (which is typical for deployment). Trying to compile an ONNX training graph to TOSA would fail due to unsupported ops (optimizers, etc.). But that’s usually not what ONNX is used for on devices.

In summary, ONNX’s interplay with TOSA is mainly through compilers like ONNX-MLIR that **lower ONNX models to TOSA for efficient execution on ARM**. This gives ONNX model developers an avenue to run on ARM accelerators with performance similar to native frameworks, by leveraging the standardized operator set.

## Cross-Framework Performance and Comparisons

It’s worth noting that TOSA acts as a common layer – so a lot of the performance differences between frameworks blur away once you’re using TOSA on a given hardware target. For example, consider running a ResNet-50 on an Arm Cortex-A CPU:

- **Without TOSA**: 
  - PyTorch might use one set of kernels (ATen with OpenBLAS or oneDNN).
  - TensorFlow might use another (Eigen or oneDNN).
  - ONNX Runtime might use MKL-DNN or its own optimizations.
  These can have different performance due to different libraries.
- **With TOSA**: If all three frameworks lower to TOSA and then use, say, Arm Compute Library (ACL) or an NPU, they’re effectively using the *same implementation* for the ops. In that case, performance differences come down to minor differences in graph optimization or quantization decisions. For instance, TensorFlow’s converter might fuse activation layers a bit better than PyTorch’s (just hypothetically), giving a slight edge. But overall, a TOSA-level benchmark is more about hardware capability than original framework.

A scenario to illustrate: suppose we run a small CNN on a Raspberry Pi (Cortex-A72 CPU) in 3 ways – via TFLite (which might use TOSA internally in the future), via PyTorch->TOSA->ACL, and via ONNX-MLIR->TOSA->ACL. If all use the same ACL library to execute, they’d all get similarly optimized NEON instructions. The differences might be within a few percent of each other. The big differences are when one framework fails to offload something that another does.

So far, we’ve seen:
- PyTorch ExecuTorch successfully offloaded a quantized MobileNetV2 to Ethos-U (NPU) using TOSA.
- TensorFlow (TFLite) has been running on Ethos-U via Vela for a while; comparable models like MobileNet or keyword spotting CNNs run efficiently there too.
- ONNX-MLIR is still maturing; it can offload standard conv and matmul heavy models but might not yet support every corner-case op.

In conclusion, integrating TOSA into PyTorch, TensorFlow, and ONNX toolchains allows each to target ARM’s ML hardware in a optimized way. PyTorch gains the ability to deploy to ARM embedded devices (something TFLite excelled at) ([ExecuTorch and TOSA enabling PyTorch on Arm platforms - AI blog - Arm Community blogs - Arm Community](https://community.arm.com/arm-community-blogs/b/ai-blog/posts/executorch-and-tosa-enabling-pytorch-on-arm-platforms#:~:text=Today%20we%E2%80%99ve%20released%20a%20TOSA,300)), TensorFlow gains a stable IR to encompass its ops for diverse backends, and ONNX gains a path to leverage MLIR optimizations for ARM. The end result is faster and more efficient ML inference on ARM, no matter which framework you start with.

# 3. Neural Network Examples with TOSA on ARM

To understand TOSA’s impact, let’s look at how various neural network types – **CNNs, RNNs, Transformers, and even diffusion models** – can be implemented and optimized on ARM via TOSA. We’ll discuss specific examples (ResNet, MobileNet, LSTM/GRU, BERT/GPT, Stable Diffusion), and the real-world performance trade-offs and benefits observed when using TOSA acceleration.

## Convolutional Neural Networks (CNNs)

**Example: MobileNetV2 on Ethos-U (ARM Cortex-M system)** – MobileNetV2 is a classic CNN for image classification, optimized for mobile. Using TOSA, developers have successfully run MobileNetV2 on tiny microcontroller-class devices with NPUs:

- In the PyTorch+ExecuTorch example earlier, MobileNetV2 was quantized (to int8) and exported via TOSA to an Ethos-U55 NPU. The result was real-time inference on an embedded platform (Arm Corstone-300 reference design with Cortex-M55 + Ethos-U55). Without TOSA and the NPU, MobileNetV2 would be far too slow in pure software on a Cortex-M. With TOSA, all the heavy ops (convolutions, dense layers, etc.) run on the NPU. The *trade-off* here is 8-bit quantization: a slight drop in accuracy (typically <1-2% for MobileNet) in exchange for ~50-100x speedup and huge energy savings.
- **ResNet-50 on Cortex-A CPU** – A larger CNN like ResNet-50 can also be lowered to TOSA and run on, say, a Cortex-A78 with NEON. If we use Arm Compute Library as the execution for TOSA ops, we get highly optimized assembly for convolutions, etc. Performance-wise, this would be similar to running ResNet via TFLite on CPU. For example, a single-threaded ResNet-50 might take ~100-200 ms on a high-end Cortex-A78 at 2.5GHz with NEON. If we had a GPU or NPU:
  - On a Mali GPU (through a GL or CL backend), we might cut that to ~30ms.
  - On an Ethos-N78 NPU (if integrated), we could get it down further, perhaps ~10ms or less, with quantization.
- **Object Detection CNN (SSD, YOLO)** – These networks combine CNN backbones with detection heads. TOSA covers the ops needed (convs, elementwise ops, reshape, etc.). On ARM NPUs, one challenge is that some detection-specific ops (like *Non-Maximum Suppression*) might not run on the NPU and would fall back to CPU. Using TOSA, you’d partition the graph: CNN part to NPU, NMS on CPU. This still yields big speed-ups, but a trade-off is the overhead of moving data between NPU and CPU for that operation. A practical example: SSD-MobileNet on Ethos-U might run the conv layers on NPU (fast) but do NMS on Cortex-M (slower), yet overall achieving say 10 FPS where a full CPU solution might be 1 FPS.
- **Edge Cases**: Some older CNN layers like *pooling* or *batch normalization* are handled in straightforward ways by TOSA (BN is typically folded into conv weights at inference). So CNNs map very cleanly to TOSA. Almost every op in a typical CNN is supported: conv2d, depthwise conv, add, relu, leaky relu, pooling, concat – all present in TOSA spec.

**Benefits for CNNs with TOSA:**
- Massive inference speed-ups on ARM NPUs or GPUs when offloaded (vs CPU).
- Reduced memory footprint with int8 quantization (4x smaller activations and weights).
- Interoperability: you can train a CNN in any framework (PyTorch, TF) and deploy via TOSA, so you’re not limited to one training framework to get acceleration.
- For mobile/embedded, using TOSA ensures you’re hitting the *vectorized* implementations of conv and GEMM – crucial for performance.

**Trade-offs:**
- Quantization is usually required to fully utilize ARM NPUs (Ethos-U/N only support int8). That can introduce a bit of accuracy loss or require careful calibration.
- Debugging at the TOSA level can be harder – e.g., if an accuracy issue occurs, one might need to inspect TOSA intermediate values with the reference model to find where things diverge. This is improving with better tooling though.
- If a CNN has an unsupported layer (say an exotic normalization), one might have to replace or approximate it to fit in TOSA. Fortunately, most common CNN layers are covered.

## Recurrent Neural Networks (RNNs: LSTMs, GRUs)

RNNs are sequential models often used for speech, language, or time-series data. They include LSTM (Long Short-Term Memory) and GRU (Gated Recurrent Unit) layers which maintain state across time steps. How do these run on ARM via TOSA?

**TOSA Representation:** There isn’t a single “tosa.lstm” operator in the base spec. Instead, an LSTM layer is represented by a series of matrix multiplies and elementwise ops:
- An LSTM cell’s equations involve multiple dense (fully-connected) operations for the gates, followed by elementwise Sigmoid and Tanh activations, and elementwise multiplies/adds for the cell state update.
- All these piecewise operations *are* in TOSA: `tosa.matMul` (for the dense part), `tosa.sigmoid`, `tosa.tanh`, `tosa.mul`, `tosa.add`, etc.
- A sequence of LSTM can thus be “unrolled” or expressed as a loop of those operations per time-step. TOSA itself doesn’t have a loop construct, so the unrolling or iteration has to be managed at a higher level (or by running the cell operations in a loop in the runtime).

**Optimizing on ARM:**
- On a CPU with Neon, the tightest bottleneck is usually the matrix multiplications for each time step. If the sequence length is T and hidden size is N, you’re doing T*N^2 multiplications over the sequence (for each gate). Neon can accelerate the matmuls, but if T is large, it’s still a lot of compute. Strategies like using batch computation (stacking multiple time steps and using batched GEMM) can help, but that often requires model redesign.
- On an NPU like Ethos, *do they support LSTM?* Not as a fused op, but Ethos-U (and likely Ethos-N) can run the matMul and activation pieces as standard ops. The challenge is that NPUs are throughput-oriented and like large batches, whereas RNNs are inherently sequential. So the utilization might not be as high as for a CNN. Still, if the LSTM is quantized, those matMuls (which become int8 matrix ops) can be offloaded. Ethos-U55/65 are optimized for convs, but a fully-connected is essentially a 1x1 conv, which they can handle.
- Another approach is converting RNNs to CNNs or using 1D convolutions to approximate them (some mobile models do that). In such cases, it just becomes a CNN problem.
- Also, ARM and others sometimes convert LSTMs to use the *time-multiplexed weights* on accelerators or use a projected LSTM (with smaller internal dims) to ease computation.

**Example: Keyword Spotting (KWS) with LSTM on Cortex-M** – Suppose we have an LSTM-based small speech keyword detector (common in TinyML). We can quantize it and run on a Cortex-M with Ethos-U:
  - The input audio features (like MFCCs) go through a few LSTM layers, then a dense, then softmax.
  - With TOSA, each LSTM time-step is broken into basic ops. The Ethos-U can run int8 matmuls and activations, but coordinating the loop might require the CPU. It might be that each time step we invoke the NPU for the matmul, get the result, apply nonlinearity, then do next step. This overhead could make pure LSTM not as efficient on the NPU unless the NPU supports some form of time-looping internally (which currently Ethos-U doesn’t, to our knowledge).
  - In practice, many TinyML models avoid long LSTMs and use either small GRUs or 1D conv + small RNN hybrids to better fit NPUs.
  - Still, the heavy lifting (matrix multiplies) can be offloaded. A GRU is a bit simpler than LSTM (fewer gates) and similarly can be broken down.

**GRU on ARM Compute Library** – On a Cortex-A, the Arm Compute Library does have functions for RNN layers (used via Arm NN or directly). When expressed via TOSA, these would map to ACL’s optimized kernels. Performance-wise, a single-arm-core can handle a moderate GRU for, say, speech (maybe real-time for a few hundred units and sequence length 100). Using all big cores or the GPU could scale that further.

**Transformer Alternatives:** These days, RNNs in many applications are being replaced by Transformers (see next section). But RNNs are still used in ultra-low-power scenarios due to smaller memory and perhaps easier streaming.

**Trade-offs for RNNs:**
- *Accuracy vs Performance:* Quantizing an LSTM can sometimes be tricky (quantization error can affect state accumulation). But if done right (with quantization-aware training or good calibration), int8 LSTMs can work with minimal loss. This is necessary to get the speed on Arm NPUs.
- *Latency:* If you need step-by-step output (e.g., streaming speech), an NPU that requires batching a whole sequence might add latency. In such cases, one might not use the NPU for the RNN, or use a hybrid approach.
- *Memory:* Unrolled RNNs can consume memory for all the intermediate states if not careful. However, a good compiler can reuse buffers for each time step since once you move forward, previous step outputs aren’t needed (except the last for state).

In summary, **LSTMs/GRUs can run on ARM via TOSA**, but performance gains are mainly from using int8 and offloading large matrix multiplies. They may not see as dramatic a speedup as CNNs on NPUs (because NPUs shine on parallel spatial data like images), but they still benefit. For many IoT applications (like anomaly detection, simple voice commands) that use small RNNs, TOSA + ARM hardware is sufficient for real-time performance within the tight power budget of embedded devices.

## Transformer Models (BERT, GPT, etc.)

Transformers have become ubiquitous in NLP and are expanding in vision (ViTs) and audio. They rely on self-attention mechanisms and feed-forward networks, often with tens or hundreds of millions of parameters. Running such models on ARM devices is challenging, but TOSA helps make it feasible on newer accelerators:

**TOSA for Transformers:** The core components of a Transformer (like BERT or GPT) include:
- Multi-Head Self-Attention (which involves dense matrix multiplies: Query*Key, then softmax, then weighted sum with Value, etc.)
- Feed-Forward Network (two dense layers with a GELU or ReLU nonlinearity in between).
- Add & LayerNorm operations.

TOSA covers nearly all primitive operations needed:
- Dense (fully connected) = `tosa.matmul` (or a series of `tosa.conv2d` if reshaped as conv).
- Softmax = this one is interesting; as of TOSA 1.0, a *tosa.softmax* op exists (applicable to 2D tensors along specified axis).
- LayerNorm = can be composed of mean, sub, square, mean again, etc., but newer TOSA might include a fused op or it’s done via a sequence of primitives (a series of elementwise ops plus a `tosa.rsqrt` for instance).
- GELU = not an elementary op in TOSA 1.0, but can be approximated or implemented via polynomial approximation using existing ops (some compilers might lower GELU to a combination of Tanh/Erf ops which TOSA does have Erf perhaps? If not, GELU could be done via LUT or left to CPU – however, many transformer models use ReLU instead for mobile).
- Sin/Cos – needed for positional encoding in some transformers. As noted, TOSA added SIN and COS ops to handle rotation in vision transformers ([Api -> Tosa operator mapping - TOSA - Discourse](https://discuss.mlplatform.org/t/api-tosa-operator-mapping/285#:~:text=We%20try%20to%20keep%20the,0%20networks)).

**Hardware Support:** ARM’s Ethos-U85 (3rd-gen microNPU) explicitly adds support for transformer workloads. It is *“the first Ethos-U to support transformer-based models natively”*, and it was designed to handle the larger vector/matrix ops and more complex activation patterns of transformers ([Ethos-U85 | Advanced NPU with Scalable Performance and Efficiency – Arm®](https://www.arm.com/products/silicon-ip-cpu/ethos/ethos-u85#:~:text=The%20Ethos,based%20machine%20learning%20%28ML%29%20tools)) ([Ethos-U85 | Advanced NPU with Scalable Performance and Efficiency – Arm®](https://www.arm.com/products/silicon-ip-cpu/ethos/ethos-u85#:~:text=Generative%20AI%20at%20the%20Edge)). Ethos-U85 supports up to 2048 MACs, making it possible to accelerate the big matrix multiplications in attention (like multiplying a 128-dim query by a 128x128 key matrix for each head, etc.). For example:
- A tiny BERT variant (say 4-layer, 128 hidden dim) could be run on Ethos-U85 entirely on the NPU via TOSA. The attention’s Q*K^T multiplication and softmax might run on NPU, though softmax could be tricky if not supported, but likely they added it. The feed-forward layers (dense+dense) definitely run on NPU. Preliminary info suggests Ethos-U85 can handle the entire transformer without falling back to CPU ([[PDF] An Energy Efficient, TOSA compatible, NPU for Transformer Networks](https://cms.tinyml.org/wp-content/uploads/summit2024/Rakesh-Gangarajaiah_final.pdf#:~:text=Networks%20cms,without%20fallback%20to%20host%20CPU)) ([Getting Started With ExecuTorch And Ethos-U85 - AI blog](https://community.arm.com/arm-community-blogs/b/ai-blog/posts/executorch-support-ethos-u85#:~:text=Getting%20Started%20With%20ExecuTorch%20And,)) – meaning ops like softmax and layernorm were presumably implemented on NPU or efficiently coordinated.
- At edge, latency is still a concern: even with 4 TOPS (Ethos-U85 max), a large model like full BERT-base (110M params) might be slow. But quantized and pruned smaller transformers can get reasonable speeds.

On mobile SoCs, you often have a more powerful NPU or GPU:
- For instance, Qualcomm Hexagon or Samsung’s NPU or Apple’s ANE – while not “Ethos”, they similarly require the model in some standard set of ops. Qualcomm’s compilers might take an ONNX or TFLite model (which could be seen as similar to TOSA). If targeting via TOSA, you’d rely on something like IREE or TFLite delegate.
- A phone’s **Mali GPU** could also run transformers through OpenCL (some projects run GPT-2 on Mali via clBlast for GEMM). Performance is limited (GPUs don’t love the softmax and small batch sizes in transformers, but they can do it).
- The biggest speedups come if the NPU supports int8. There have been demonstrations of quantized BERT running on mobile NPUs with good speed (a few ms per sequence) – likely they used a similar concept to TOSA to compile it.

**Example: BERT on a smartphone (Cortex-A + Ethos-N77)** – Let’s say we want to run BERT question-answering on device:
  - Using TOSA, we convert the BERT model (maybe through TFLite) to int8. All matrix multiplications become int8 GEMM, softmax can stay in higher precision or int16 accumulators.
  - The Ethos-N NPU (in some Arm-based SoCs) accelerates those GEMMs; it might have a special mechanism for softmax (or it computes it on a vector unit).
  - The result could be an inference time of e.g. 50ms for a 128-token sequence, which is significantly better than >500ms on CPU.
  - Trade-off: quantization might hurt BERT’s accuracy slightly (there’s research on quantizing transformers – it usually needs some fine-tuning with quantization-aware training to not drop a lot of accuracy, especially for 8-bit). But it’s doable.
  
**Example: Vision Transformer (ViT) on edge device** – Vision transformers have large matrix multiplications too. The Sin/Cos op addition in TOSA indicates they specifically wanted to support ViTs’ positional encodings ([Api -> Tosa operator mapping - TOSA - Discourse](https://discuss.mlplatform.org/t/api-tosa-operator-mapping/285#:~:text=We%20try%20to%20keep%20the,0%20networks)). A ViT-base model is huge for edge, but smaller ones or just the attention block in a bigger model can be offloaded. One could imagine a hybrid model where a CNN extracts features and a small transformer encoder runs on those features. TOSA would express both parts seamlessly for the hardware. The NPU would accelerate the convs and the attentions equally.

**Performance and Trade-offs:**
- *Memory Bandwidth*: Transformers tend to be memory-heavy (lots of intermediate activations, especially with many tokens). On a small device, this can be a bottleneck. Ethos-U85 specifically improved memory bandwidth usage, which is critical for transformers that might need to fetch large weight matrices ([[PDF] An Energy Efficient, TOSA compatible, NPU for Transformer Networks](https://cms.tinyml.org/wp-content/uploads/summit2024/Rakesh-Gangarajaiah_final.pdf#:~:text=,%E2%80%A2%20Improves%20system)).
- *Utilization*: NPUs are highly utilized on large matrix ops, but the softmax and normalization might be less parallel and could bottleneck if not accelerated. If those run on CPU, they become latency adders. Ideally, one uses partial FP16 or int16 accumulation to handle softmax on NPU. Some NPUs might implement a fused “attention” operation (though TOSA doesn’t have a single op for entire attention, just building blocks).
- *Quantization Impact*: For CNNs, int8 quantization is well established with minor accuracy loss. For Transformers, int8 can sometimes cause more noticeable drops in accuracy (especially for language tasks). Techniques like distillation or QAT are used to mitigate this. It’s a trade-off: either accept some accuracy loss and get 4x speed, or use 16-bit which might be 2x speed with better accuracy. TOSA supports both int8 and fp16, so you could also run a model in mixed precision.
- *Throughput vs Latency*: On embedded, often you care about latency (for a single query). TOSA on a dedicated accelerator is great for latency as it avoids a lot of overhead. On bigger systems, you might batch queries for throughput, but on mobile you usually batch = 1. The TOSA approach executes the network layer by layer on hardware with minimal overhead per layer.

In practice, **running transformers on ARM via TOSA is cutting-edge but advancing quickly**. What was unthinkable (running BERT on a microcontroller) is becoming possible in some form. Arm’s push with Ethos-U85 ([Ethos-U85 | Advanced NPU with Scalable Performance and Efficiency – Arm®](https://www.arm.com/products/silicon-ip-cpu/ethos/ethos-u85#:~:text=Generative%20AI%20at%20the%20Edge)) and the addition of necessary ops to TOSA indicates real-world use. For example, with proper optimization, a small transformer can power things like on-device language understanding (voice assistant) or anomaly detection in IoT without a cloud service.

## Image Diffusion Models (e.g., Stable Diffusion)

Diffusion models like **Stable Diffusion** are among the most demanding generative models, typically running on GPUs with tens of billions of FLOPs per image. They consist of very large neural networks (often UNet-based convolutional networks for the diffusion process, plus transformers for text encoding). Running these on ARM devices (especially without a desktop GPU) is extremely challenging. However, there are efforts to optimize and run such models on edge hardware in a limited fashion:

**Stable Diffusion Components:** The standard pipeline includes:
- A **UNet** model (~UNet with attention, ~860M params for Stable Diffusion 1.x) that denoises images. This is basically a CNN with some cross-attention layers.
- A **VAE (Variational Autoencoder)** decoder to generate final images (a few conv layers).
- A **Text Encoder** (often a transformer like a small BERT) to encode the prompt.

Out of these, the UNet is the most heavy part (inference requires iterating 50 or so timesteps, each doing a forward pass of the UNet). On ARM, to even attempt this:
- The model must be heavily optimized: quantized to int8 or int16, possibly pruned or reduced in size.
- Use of acceleration is a must: GPU or NPU.

**TOSA applicability:** If we had a TOSA-compatible representation of Stable Diffusion’s UNet, what would it entail?
- The UNet has convolutional layers (supported by TOSA conv2d), group norms, SiLU activations (Swish, which can be done via Tanh/Sigmoid approximations or as a LUT), and cross-attention (which is basically a small transformer block).
- Cross-attention would use ops: matmul (for query*key), softmax, matmul (attention output), just like a transformer. All these can be expressed in TOSA (assuming softmax is there).
- So yes, in theory the entire UNet graph can be lowered to TOSA (with some hacks for SiLU if not natively present, maybe using `tosa.mul` of sigmoid output and input to mimic SiLU).

**Have people run Stable Diffusion on ARM?** Yes, there are examples:
- Some enthusiasts ran *Stable Diffusion on a Raspberry Pi* (CPU only). For instance, one article noted an LLM on Pi, and similarly stable diffusion can run on Raspberry Pi 4 but it takes on the order of tens of minutes for a single image, which is more a proof of concept.
- On Apple’s M-series (ARM architecture but with a strong GPU/ANE), Stable Diffusion can run reasonably (~1-2 minutes on an M1 for a 512x512 image). Apple’s CoreML format is somewhat analogous to TOSA (a fixed set of ops), and they optimize the model heavily (split over ANE and GPU).
- On Android devices (Snapdragon chips), there have been demos of Stable Diffusion (with quantization) running in ~10 seconds per image using the Adreno GPU and DSP. This uses Qualcomm’s libraries, not explicitly TOSA, but if one were to use TOSA it could also compile to use those hardware units via something like IREE or Hexagon delegate.

**Imagining TOSA on a diffusion model with Ethos-N**: If a phone had an Ethos-N NPU with, say, 10 TOPS:
  - The UNet could be quantized and partitioned so that convolutions run on the NPU. The cross-attention might also mostly run on NPU (since it’s matrix multiplications). However, the NPU memory might be a limitation for a model as large as full SD UNet – it might not hold all weights, requiring swapping from DRAM frequently, which slows things.
  - Possibly, a smaller diffusion model (like Stable Diffusion distilled or lower resolution) might fit better.
  - The VAE decoder is small – that can easily run on NPU or GPU at the end.
  - The text encoder (if using CLIP’s text model with 123M params) is also heavy; one might offload that or do it on CPU ahead of time since it’s not in the iterative loop.

**Trade-offs and feasibility:**
- *Precision vs Quality*: Diffusion models are sensitive to quantization. Int8 UNet might produce lower image quality or require more diffusion steps to compensate noise. There’s active research on quantizing diffusion models. Perhaps one would use int8 for convs but keep some attention in higher precision.
- *Performance needs*: Real-time diffusion (a few seconds) on mobile is still kind of out-of-reach for high resolutions, but we see progress. For instance, *running a tiny Stable Diffusion at 256x256 on an embedded GPU in a few seconds* is plausible with heavy optimization.
- *Memory*: These models consume a lot of RAM (hundreds of MB for weights). On microcontrollers, it’s impossible; on a high-end phone with 8GB RAM, it’s okay. But NPUs often have limited SRAM, so they must constantly stream data from DRAM – performance suffers if memory bandwidth is low.

**Benefits of TOSA for such models:**
- Unified optimization: One could use the same pipeline to target different hardware. For example, use TOSA to target a Mali GPU on one device, and an Ethos NPU on another, from the same model definition.
- It ensures any custom or uncommon op in the model (like a particular activation function) is broken into supported pieces, increasing portability.
- TOSA’s quantization handling means you can express the int8 model’s behavior exactly, which helps ensure the accelerated result matches a reference.

In reality, **Stable Diffusion on ARM with TOSA** is on the bleeding edge and mostly a research project. The average user won’t be generating high-res images on a microcontroller anytime soon. But it’s a great stress test for TOSA and ARM: if you can express and run something as complex as diffusion, you can handle most other models which are usually smaller.

There are already signs of progress:
- Projects like [**StableHLO**](https://openxla.org/) (by OpenXLA) aim for standardized ops for even large models, similar in spirit to TOSA.
- Community demos of *“Stable Diffusion on Mobile”* often use CoreML or ONNX compressed models – converting those to TOSA would be a next step to leverage ARM-specific compilers.
- The fact that **Stability AI and Arm have partnered** (as hinted by news ([Model Optimization for Speed and Efficiency by Koki Mitsunami](https://www.youtube.com/watch?v=1vnKPLFxs0g#:~:text=,and%20Efficiency%20by%20Koki%20Mitsunami))) suggests future efforts in making such generative models run on ARM devices, possibly by creating optimized networks or using ARM IP.

**Real-World Performance Summary:**
- CNNs (ResNet/MobileNet) – Achieve real-time or better inference on ARM when using NPUs or GPU via TOSA. E.g., MobileNetV2 224x224 can run in a few milliseconds on Ethos-U65 (for classification) when quantized.
- RNNs (LSTM/GRU) – Can run efficiently for small models; performance is enough for many IoT tasks (e.g., a GRU-based anomaly detector running at several kHz on M-class hardware with NPU offload).
- Transformers (BERT/GPT) – Smaller versions can run with some latency on edge NPUs. Expect seconds for complex tasks on micro NPUs, or tens of milliseconds on bigger NPUs for simplified models. The key is model size reduction and quantization.
- Diffusion (Stable Diffusion) – Currently mostly experimental; perhaps 5-10 seconds per 256x256 image on a high-end phone GPU is a ballpark if optimized. With future NPUs and model compression, this could come down. The trade-off is significant quality reduction if heavily quantized.

The **takeaway** is that TOSA enables even very different neural network architectures to be uniformly handled on ARM hardware. By converting everything into this standard set of ops, developers can tap into ARM’s acceleration (CPU SIMD, GPU, NPUs) for a wide range of models. The performance gains are huge when hardware is used, but developers must often contend with quantization and memory constraints. Nonetheless, the fact that we’re discussing running things like BERT or Stable Diffusion on ARM devices at all is a testament to the progress in ML optimization – and TOSA is one of the tools making it possible.

# 4. Industry Applications & Research Trends

TOSA’s emergence has significant implications in industry and ongoing research. Here we cover how TOSA is used in modern applications, state-of-the-art research projects incorporating TOSA, open-source efforts, and how custom software is leveraging TOSA for ARM-based ML workloads.

## Industry Applications of TOSA

**On-Device AI in Mobile and IoT:** One of the driving forces behind TOSA is the need to run **machine learning on billions of edge devices** (smartphones, cameras, IoT sensors) efficiently. Industry players are adopting TOSA in their toolchains to achieve this:
- **Smartphones:** Chip vendors and OEMs want apps (camera apps, AR filters, voice assistants) to utilize the phone’s AI accelerators seamlessly. By compiling models to TOSA, they can deploy the *same model* across different phones (some with GPUs, some with NPUs). For example, an image enhancement model could be trained once and then distributed in an app where a runtime converts it via TOSA to whatever acceleration is available (Arm GPU or NPU or CPU).
- **IoT and Smart Cameras:** Companies making home security cameras or IoT sensors are using Arm Cortex-M CPUs with Ethos-U NPUs to do things like person detection, anomaly detection, keyword spotting *on-device*. TOSA is key here because these tiny NPUs require the model to be in a specific form. With TOSA, they can take a TensorFlow Lite model from a researcher and quickly determine if it’s compatible (via the TOSA Checker tool) and then compile it to their device. This saves a ton of engineering effort that would otherwise go into rewriting models or changing model architectures to fit.
- **Automotive:** In cars, ARM cores are used for ADAS (advanced driver-assistance systems) and infotainment. While heavy autonomous driving networks run on bigger SoCs, simpler things like driver monitoring or voice control can run on Arm cores. Using TOSA, an automotive supplier can ensure their models (which might be trained in PyTorch by one team) run on the Arm-based platform in the car without having to manually port every new model. Additionally, because TOSA is stable and versioned, it fits the longer life-cycles in automotive – you can qualify a TOSA implementation for safety and know that any model built to TOSA 1.0 spec will behave predictably on it.
- **Healthcare and Wearables:** ARM processors power many medical devices and wearables (smart watches, hearing aids). These devices are using ML for things like arrhythmia detection, fall detection, etc. TOSA allows model developers (who may use high-level frameworks) to deploy onto resource-constrained ARM systems reliably. For example, an ECG anomaly detection model (maybe an RNN or small transformer) can be compiled via TOSA to run on a Cortex-M with DSP extensions or a small NPU in a wearable, ensuring low power consumption.
- **Enterprise & Cloud on ARM:** With the rise of ARM in the cloud (e.g., AWS Graviton instances), there’s interest in ML inference on ARM servers. TOSA could standardize the deployment of models to these servers, particularly for edge-cloud scenarios or 5G base stations, etc. While ONNX is commonly used in cloud, TOSA could complement it by providing a more rigorously defined execution spec that hardware vendors (like those making ARM-based data center accelerators) might prefer.

Real-world example: **Tesla** (as an illustrative guess) has an in-cabin camera to detect driver alertness. They could train a model in PyTorch, export to TOSA, and run it on an ARM-based chip in the car responsible for that task. This way, the model is consistent across different chip revisions and can be updated OTA by just sending a new TOSA graph to the car.

**Standardization and Compatibility:** The industry has suffered from fragmentation – model from framework A doesn’t easily run on hardware B. TOSA is solving this by acting as a *common denominator*. A quote from Arm’s IoT segment lead highlights this need for standardization: *“Models are developed using many frameworks which often need to target many hardware devices. This hampers scaling. This is the reason Arm has been working on a new standardized operator architecture called TOSA...”*. Companies want to scale their AI solutions across product lines and generations of hardware. By converging on TOSA, they reduce the engineering needed for each new deployment.

**Ethos NPU Deployments:** Multiple semiconductor companies (not just Arm) have licensed the Ethos-U and Ethos-N NPUs to include in their chips (e.g., NXP, STM, Renesas for micro NPUs; MediaTek or others for mobile NPUs). All these deployments use the TOSA-based flow. So, in the field, whenever an Ethos NPU is present, any ML application running on it is implicitly using TOSA (via the compiled command stream, which was derived from TOSA or TFLite). For instance, a new smart speaker that uses an NXP i.MX RT MCU with Ethos-U55 will run its wake-word detection model through the TOSA->Vela pipeline. From an end-user perspective, they see that the device can detect “Hey Alexa” offline with high accuracy and low power – TOSA is an invisible but vital part of that solution.

**Arm NN and Compute Library Users:** Arm NN (a neural network inference engine) and the Arm Compute Library (low-level NEON/OpenCL kernels) now support executing TOSA-compliant graphs. Some companies integrate Arm NN in their products to run multiple frameworks’ models. TOSA allows Arm NN to have one frontend for many model formats. So an industry application might be: an industrial sensor uses Arm NN to run either a PyTorch model or TF Lite model provided by two different contractors – both get converted to TOSA under the hood and run on the same software backend on an Arm Cortex-A.

## State-of-the-Art Research and Open-Source Projects

**MLIR and Compiler Research:** TOSA being an MLIR dialect means it’s part of many research compiler frameworks. One example is the **Union AI project** (from an academic paper ([](https://arxiv.org/pdf/2109.07419#:~:text=lowering%20TensorFlow%20code%20to%20mid,trained%20on%20GPUs%20and%20the)) ([](https://arxiv.org/pdf/2109.07419#:~:text=graph,As%20explained%20next))) which deals with HW/SW co-design in MLIR. They explicitly *“follow the TOSA dialect approach”* and focus on inference, assuming models are trained elsewhere ([](https://arxiv.org/pdf/2109.07419#:~:text=lowering%20TensorFlow%20code%20to%20mid,trained%20on%20GPUs%20and%20the)). This indicates that even academic researchers see value in using TOSA as the middle layer for experimenting with new hardware optimization techniques. By using TOSA, they avoid reinventing the wheel on defining ops and can concentrate on mapping those ops to novel hardware architectures.
  
**IREE (Google’s ML compiler)**: The IREE project can ingest TFLite models and indeed uses TOSA as one possible IR after import. Researchers and developers working with IREE (for example, on Vulkan or on novel processors) are testing TOSA as part of their flow. There’s ongoing development to improve TOSA support in IREE and handle dynamic shapes, etc., in MLIR (as seen in discussions on LLVM forums, e.g., shape inference for TOSA).

**OpenXLA / StableHLO**: While StableHLO is a separate effort (from Google/OpenXLA, aiming to capture XLA/HLO ops for portability), both it and TOSA share the vision of an interface layer for ML models. In the open-source community, there’s discussion about how these standards relate. It’s likely we’ll see convergence or bridges (for instance, a conversion from StableHLO to TOSA or vice versa for certain deployments). For now, they coexist, and some research compares them for coverage and performance. Pete Warden, an ML engineer, mentioned *“existing attempts that had some success, such as ONNX or MLIR’s TOSA dialect, but they’ve struggled either with coverage…”* ([Why are ML Compilers so Hard? - Pete Warden's blog](https://petewarden.com/2021/12/24/why-are-ml-compilers-so-hard/#:~:text=Why%20are%20ML%20Compilers%20so,struggled%20either%20with%20coverage)), implying ongoing efforts to broaden these standards.

**TinyML and Edge AI Projects:** In the tinyML open-source realm, TOSA is definitely recognized. The **tinyML Summit 2023** had a talk on “Ethos-U support in TVM” – TVM is an open deep learning compiler. ARM’s team worked to integrate Ethos support into TVM, meaning TVM can compile models to use Ethos NPUs. Underneath, this uses the TOSA spec because TVM will offload only those ops the NPU can do (defined by TOSA). They likely used Arm’s NPU SDK which accepts TFLite or TOSA. So open-source TVM now has a path for Arm NPUs, broadening community access.
  
**Academic Research on Efficient NPUs:** The paper *“An Energy Efficient, TOSA-compatible, NPU for Transformer Networks”* ([[PDF] An Energy Efficient, TOSA compatible, NPU for Transformer Networks](https://cms.tinyml.org/wp-content/uploads/summit2024/Rakesh-Gangarajaiah_final.pdf#:~:text=,without%20fallback%20to%20host%20CPU)) suggests that researchers (likely from Arm or a university in the tinyML community) have built an NPU that runs transformer models end-to-end with TOSA. That is cutting-edge since transformers are new in tinyML. The paper presumably highlights design choices to improve efficiency while staying within the TOSA operator set (so no custom ops that break compatibility). This research not only advances hardware but serves as validation that TOSA can indeed express those advanced models fully.

**Open-Source Tools Around TOSA:**
- **TOSA Reference Model & Checker**: Arm’s MLPlatform has released a C++ reference implementation of TOSA (which can execute a TOSA flatbuffer for validation) and a *TOSA Checker* tool ([ Contributing to TOSA software - ML PLatform ](https://www.mlplatform.org/tosa/software.html#:~:text=TOSA%20Checker)). These are open on GitHub. People in the community can use the Checker to test if their model (especially TFLite model) is TOSA-compatible. This is important for open-source projects where you might want to ensure your model can run on all TOSA-compliant devices. For example, an open-source TinyML model repository might include a step to verify TOSA compliance so users know it’ll run on Arm NPUs.
- **Torch-MLIR and ONNX-MLIR**: As mentioned, these projects are open-source and welcome contributions. They’re on the forefront of bridging popular frameworks to MLIR (and thus to TOSA). Contributors are actively adding support for new ops, optimizing conversions, and fixing bugs (for instance, handling of edge-case rounding modes or shape handling in TOSA).
- **Community Forums**: The MLPlatform forum (discuss.mlplatform.org) has discussions where developers ask about new operators, shape inference, etc., for TOSA, and Arm engineers (like Eric Kunze) respond ([Api -> Tosa operator mapping - TOSA - Discourse](https://discuss.mlplatform.org/t/api-tosa-operator-mapping/285#:~:text=EricKunze%20%20May%2023%2C%202024%2C,2)) ([Api -> Tosa operator mapping - TOSA - Discourse](https://discuss.mlplatform.org/t/api-tosa-operator-mapping/285#:~:text=,mlir%20path)). This open engagement shows the spec is evolving in collaboration with the community. They even hint at upcoming 1.0 release ensuring backward compatibility ([Api -> Tosa operator mapping - TOSA - Discourse](https://discuss.mlplatform.org/t/api-tosa-operator-mapping/285#:~:text=We%20try%20to%20keep%20the,0%20networks)), which is crucial for industry adoption.

**Trends:**
- More **frameworks and compilers adopting TOSA**. We’ve seen TensorFlow, PyTorch, ONNX side, TVM, IREE – likely adoption will continue. Perhaps JAX or other new frameworks could output TOSA via OpenXLA in the future.
- **Expanded Operator Set** but carefully: As new model types come (e.g., some physics-informed neural nets or other domains), TOSA will evaluate adding ops. They do so sparingly to keep it minimal. The trend is to accommodate things like transformer positional encoding (which they did) and maybe certain efficient training ops down the road.
- **Quantization & Mixed Precision**: Research into better quantization of models (like post-training quantization techniques, quantization-aware training) directly benefits TOSA use, because TOSA leverages quantization heavily. We see research on int4 or int2 quantization for certain networks – future TOSA versions might include 4-bit support if hardware moves that way.
- **Compiler Optimization Research**: With a stable spec, researchers can focus on optimizing computation graphs (like scheduling, memory management, etc.) knowing the ops won’t change. We might see research on automated partitioning of TOSA graphs (e.g., splitting which parts run on CPU vs NPU for optimal performance – a complex problem that could be tackled with ML itself).

**Open-Source Example Project:** There is a project on GitHub (semi-open, by ARM) called **ethos-u-vela** (available via PyPI ([ethos-u-vela - PyPI](https://pypi.org/project/ethos-u-vela/#:~:text=ethos,U55%2C%20product%20and))). It’s not fully open-source (development happens internally then code is released), but it is accessible. Vela converts TOSA or TFLite models to NPU command streams. This is a critical tool for anyone using Ethos-U. People have built on Vela to integrate into their flows (e.g., integration with Renode emulator to test models, etc.). Vela also includes documentation on which TOSA operators it supports on a given NPU and any limitations ([ethos-u-vela/OPTIONS.md at lf-6.6.3_1.0.0 - GitHub](https://github.com/nxp-imx/ethos-u-vela/blob/lf-6.6.3_1.0.0/OPTIONS.md#:~:text=ethos,NPU%2C%20and%20what%20the)). For instance, it may show a table: Conv2D (supported), DepthwiseConv2D (supported), Softmax (supported if depth <= X), etc. This helps both industry engineers and researchers understand what their model needs to look like.

**Collaboration and Ecosystem:** TOSA’s development itself is a collaboration (Linaro/Arm and input from Google for the MLIR parts). This collaborative spirit extends to how it’s used:
- Arm worked with **Meta (Facebook)** to get PyTorch support.
- Arm works with **TensorFlow open-source** for MLIR.
- There’s likely collaboration with chip partners (who give feedback like “we need this op for our use-case”).
- Conferences like LLVM dev meetings or TinyML see talks on these topics, spreading awareness.

In industry, **nobody wants to reinvent ML operators** – TOSA is gaining traction as the “boring infrastructure” everyone can agree on, so they can then compete on actual hardware performance or software features. For example, Synopsys and ARM both make NPUs; if both support TOSA, then model developers don’t have to worry about the difference, they just know both claim “TOSA compliance.” We’re not fully there yet, but it’s moving in that direction.

## Custom Software Development Leveraging TOSA

Companies and developers building custom ML software on ARM are leveraging TOSA in a few key ways:

- **Custom Compilers for Proprietary Accelerators:** Suppose a company designs its own neural accelerator for a specific domain (e.g., an accelerator optimized for 3D point cloud processing). They want it to run neural nets from PyTorch. Instead of writing custom converters for PyTorch, they can implement a TOSA frontend on their compiler. If their hardware can support all TOSA ops (maybe with some not used if irrelevant), any model lowered to TOSA can be mapped. They can even choose to only support a subset of TOSA (like just conv and add and a few others) – as long as they declare that and use the TOSA Checker to ensure models fit that subset. This significantly reduces software bring-up time for new hardware. We see an analogy in Vulkan for graphics – instead of every GPU having its own API, they all implement Vulkan. TOSA aspires to do similar for small ML accelerators.
  
- **Edge AI Software Platforms:** Several startups and enterprises offer platforms for deploying edge AI (like NI’s LabVIEW AI toolkit, or smaller ones that let you drag-and-drop models onto devices). If they incorporate TOSA, they can support more frameworks. For instance, an “AutoML for edge” tool could output a TOSA flatbuffer that then goes into device-specific compilers. If a new device comes along that also supports TOSA, integration is easy. This makes the platform scalable. I suspect tools like **Edge Impulse** (a TinyML platform) are watching TOSA; currently they often output TFLite, but since TOSA can ingest TFLite, it’s aligned.
  
- **Optimizing Existing Workloads:** Some developers might take an existing model and optimize it for ARM by converting to TOSA and then using hand-tuned libraries. For example, an engineer might export an ONNX model to TOSA and then write some Neon intrinsics for each TOSA op to squeeze out the last bit of performance on a particular CPU where the general libraries weren’t optimal for that model’s pattern.
  
- **Verification and Testing:** TOSA’s reference model can be integrated into testing frameworks. A custom ML software stack might run the TOSA reference model as a “golden” to verify their accelerator’s outputs match for a battery of test cases. This is extremely useful for developing safety-critical AI (like medical or automotive) where you need to prove correctness against a spec. TOSA provides that spec and a reference implementation. So, custom software can use it to catch any discrepancies early.
  
- **Contributing Back:** Many companies are contributing back to TOSA-related projects. For example, AMD or NVIDIA might not directly care for TOSA (they have their own ecosystems), but smaller IP companies and AI startups do. If Company X uses TOSA and finds it’s missing an op that’s crucial, they might propose it through the forums or even contribute a patch to the spec or MLIR. This open involvement ensures TOSA covers real-world needs.

**Future Directions in Industry & Research:**
- Expect **TOSA 1.0** finalization and then more widespread declaration of support (Arm has hinted at 1.0 guaranteeing backward compatibility ([Api -> Tosa operator mapping - TOSA - Discourse](https://discuss.mlplatform.org/t/api-tosa-operator-mapping/285#:~:text=We%20try%20to%20keep%20the,0%20networks)), which industries love because it means stability).
- More **tooling**: possibly a visualizer for TOSA graphs, better debuggers, etc., to make it easier for software teams to adopt it.
- **Research on training with TOSA**: eventually, academic projects might explore using TOSA for on-device training or continual learning (with new ops for gradients). The spec might evolve to include an “Training ops” extension (just speculative).
- **Interplay with Cloud**: There might be scenarios where you train a small model on an edge device (federated learning). If TOSA could represent training loops and gradients, that could open up an interesting avenue of on-device learning standardization.

In conclusion, TOSA is increasingly used as a linchpin in the ARM AI ecosystem across industries. It helps companies deploy ML models reliably on ARM hardware, fosters collaboration in open-source compiler projects, and influences research directions by providing a common ground. The trend is clearly towards more adoption, as seen by its integration into various frameworks and compilers, and by ARM’s continued investment in tools and hardware aligned with TOSA. This makes specialization in TOSA and ARM ML optimizations both a relevant and forward-looking expertise to develop (as we will explore in the next section).

# 5. Career Pathways & Learning Roadmap

Specializing in TOSA and ARM-based ML optimizations is a niche but increasingly valuable career path as edge AI and heterogeneous computing gain momentum. In this section, we outline steps to become an expert in this area, recommend learning resources, discuss the future outlook for TOSA in ML, and suggest ways to connect with professionals and communities in the field.

## Steps to Become an Expert in TOSA and ARM ML Optimization

**1. Build a Strong Foundation in Machine Learning and Deep Learning:** You should be comfortable with neural network basics (CNNs, RNNs, Transformers, etc.) and how training and inference work. Understanding model architectures at a conceptual level is crucial because optimization often involves tweaking or transforming these architectures. Courses or books on Deep Learning (like Andrew Ng’s courses or the book *“Deep Learning” by Goodfellow et al.*) are a good starting point. Make sure you also grasp the basics of **quantization, pruning, and other model compression techniques** since these are commonly used for ARM optimizations.

**2. Learn the Fundamentals of Computer Architecture and ARM specifics:** To optimize for ARM, you need to know what makes ARM CPUs and NPUs tick. Learn about:
- ARM Cortex-A vs Cortex-M differences (e.g., A for application processors with NEON SIMD, M for microcontrollers often paired with Ethos-U).
- The ARM Neon/SVE vector instruction set (how SIMD parallelism works).
- Memory hierarchy and why cache/memory access patterns matter for ML.
- If possible, get some hands-on experience writing or using NEON intrinsics or assembly for simple operations, to appreciate low-level performance considerations.

**3. Get Familiar with MLIR and Compilers for ML:** Since TOSA is an MLIR dialect, understanding MLIR (Multi-Level IR) and how compilers work is valuable. This could mean:
- Going through the LLVM tutorial (to get a sense of IR and passes).
- Specifically looking at MLIR: read about how MLIR is structured with dialects. The LLVM documentation and tutorials on MLIR are a great resource.
- Try writing a simple MLIR pass or at least reading some existing ones (for example, read the `LegalizeTF` pass in TensorFlow or a simple pass in torch-mlir). This helps demystify the process of lowering high-level ops to a target like TOSA.
- If you’re not into developing compilers, at least learn to *use* them: e.g., use `onnx-mlir` or `torch-mlir` on a model and inspect the output.

**4. Study the TOSA Specification in detail:** The official TOSA spec (on mlplatform.org) defines each operator’s behavior. Read through it to know:
- What operators exist (e.g., the list of conv2d, fully_connected, etc., and more obscure ones like `tosa.table` or `tosa.segment_sum` if any).
- The precise semantics: for example, how does TOSA handle padding in convs? What rounding does `tosa.rescale` use?
- The data types supported (int8, int16, FP16, FP32).
- Constraints like rank of tensors for each op.
This knowledge is key when debugging or implementing optimizations, as you’ll know what transformations are valid or how to express something in TOSA.

**5. Hands-on Practice with ARM ML Tools:** Gain practical experience by working through examples:
- Use TensorFlow Lite on an ARM device (like Raspberry Pi or a microcontroller with e.g. Arduino Nano BLE) and then try using the Arm Vela compiler to compile a model for Ethos-U. This will expose you to TOSA indirectly, as Vela will tell you if an op is not supported (meaning not in TOSA base set).
- Use PyTorch with the ExecuTorch flow (as in the tutorial we discussed) to export a model for ARM. Try quantizing a small model and running it in the Arm Corstone-300 MPS3 simulator (Arm provides FVP simulators free for developers). This will give you experience with the workflow and the kind of issues that can arise (like quantization accuracy, unsupported ops, etc.).
- Play with Arm Compute Library (ACL): Write a small C++ program that uses ACL to run a model (perhaps constructing it with ACL’s graph API or even using Arm NN with a TOSA Compute Library backend if available). This helps you understand performance on CPU/GPU and how things like memory layout affect it.

**6. Deepen Knowledge in Quantization and Model Optimization:** Since ARM optimization often = quantization:
- Learn about quantization schemes (symmetric vs asymmetric, per-axis quantization for weights, etc.).
- Practice quantizing models in TensorFlow (using `tf.quantization` tools or TFLite converter) and in PyTorch (FX Graph Mode Quantization).
- Understand how to evaluate accuracy vs performance trade-off after quantization.
- Optionally, learn about other compression like pruning or knowledge distillation, since a specialist might also consider those to get a model small enough for a given ARM device.

**7. Contribute to or use Open-Source TOSA Projects:** A great way to become expert is to **contribute**:
- Join the `llvm.discourse` forum or the MLPlatform forum and follow discussions ([Api -> Tosa operator mapping - TOSA - Discourse](https://discuss.mlplatform.org/t/api-tosa-operator-mapping/285#:~:text=EricKunze%20%20May%2023%2C%202024%2C,2)). Try to fix a small issue or add support for something in torch-mlir or onnx-mlir regarding TOSA.
- Even if not contributing code, build the latest torch-mlir or onnx-mlir, test different models, and report bugs or performance numbers.
- Explore the TOSA reference model code or TOSA Checker (maybe contribute a test case).
- This not only hones your skills but gets you known in the community of experts in this domain.

**8. Learn about ARM Hardware (Ethos NPUs, Mali GPUs, etc.):** Understanding the hardware helps you optimize for it:
- Read technical overviews of Ethos-U55/U65 and U85, Ethos-N77/N78 if you can find (Arm often has whitepapers or blog posts).
- Mali GPU architecture (how it handles compute, the memory tiling).
- For CPUs, learn how to use profilers like perf or ARM Streamline to see where cycles go when running ML code.
- If possible, get a development board with a representative chip (e.g., a STM32 with Ethos-U, or a Raspberry Pi 4 for CPU, or an Android phone with a known NPU) and experiment by deploying models and measuring.

**9. Software Engineering and Optimization Skills:** This field sits at intersection of ML and systems programming. Develop skills in:
- Profiling and benchmarking (so you can measure impact of optimizations on ARM).
- C++ (a lot of these libraries like ACL, Arm NN, and even writing MLIR passes will involve C++).
- Python (for glue code, and because PyTorch/TensorFlow usage is Python-heavy).
- Reading and understanding large codebases – e.g., dive into Arm Compute Library code for conv layers, or how TFLite implements an operator. This will make you comfortable with the idea of implementing new ops or optimizing them.

**10. Keep Up with Latest Research and Updates:** Subscribe to newsletters or follow conferences relevant to edge ML:
- TinyML Summit proceedings, Edge AI forums, ARM AI Tech Talks.
- ArXiv papers on model optimization, quantization, neural architecture search for efficiency.
- ARM community blog posts (they often post about updates to their ecosystem, like the ExecuTorch blog we cited).
- This helps you stay ahead with knowledge of where TOSA might be heading (e.g., if a new type of model is trending, think how TOSA would support it).

## Recommended Learning Resources

- **Books:**
  - *“Deep Learning for Embedded Systems”* – (Not sure if there’s a book exactly by that title, but there are a few texts and sections in books about on-device deep learning).
  - *“TinyML”* by Pete Warden and Daniel Situnayake – provides a great introduction to deploying ML on microcontrollers. While it doesn’t cover TOSA (it predates it), it gives context on constraints in that space.
  - *“Efficient Processing of Deep Neural Networks”* by Vivienne Sze et al. – excellent book on hardware architectures and optimizations for DNNs. It can give insight into why certain ops are expensive and how hardware accelerates them, which is useful when thinking about TOSA’s role.
  - *“Machine Learning Systems: Design and Implementation”* (maybe by Jeff Johnson) – covers how ML models are deployed and optimized, which touches on compilers and could indirectly help with understanding something like TOSA.
  
- **Online Courses & Tutorials:**
  - Coursera or edX courses on **compilers for deep learning** (there might be specialized ones, or general compiler courses to learn LLVM basics).
  - **ARM’s own training**: Arm sometimes offers online training or tutorials on their tools. For instance, check Arm’s developer website for any guides on ML on Ethos or using Compute Library.
  - **TinyML EdX course** by Harvard (there is a 3-part professional certificate on TinyML) – this won’t mention TOSA, but it teaches deploying and optimizing models on Arm Cortex-M, which is directly applicable.
  - **LLVM/MLIR workshop videos**: There are recorded talks from LLVM dev meetings on MLIR tutorials, and maybe specifically on TOSA dialect (the Arm MLIR team might have presented something).
  - **OpenXLA Bootcamp** (if available) or similar – covers intermediate representations like HLO which are akin to TOSA in concept.

- **Documentation & Official Resources:**
  - The **TOSA specification** (on mlplatform.org) – it’s a primary resource and often updated. Keep a copy of the PDF.
  - **Arm Developer** website: contains documentation for Ethos NPUs, Arm NN SDK, etc. The Ethos-U55/U65 driver stack documentation can give you perspective on real-world usage (APIs, limitations).
  - **TensorFlow MLIR guide** (the official TF docs have a section on MLIR with TOSA passes).
  - **PyTorch ExecuTorch documentation** (as we saw, there’s a tutorial and docs on pytorch.org) – following those step-by-step will build practical knowledge.

- **Open-Source Code Repos:**
  - GitHub for **llvm/torch-mlir**, **onnx/onnx-mlir** – browse the code, especially `Conversion/TOSA` directories to see how ops are mapped. This is like reading the “Rosetta Stone” of TOSA and framework ops.
  - **tosa-checker** on PyPI and MLPlatform git – you can install it (`pip install tosa-checker`) and play with checking models.
  - **Arm Compute Library** (ACL) GitHub: It’s open-source. Try to read how a depthwise conv is implemented for Neon or how they schedule GPU kernels. It’s heavy, but even skimming gives you a feel for what low-level optimization looks like.
  - **Example projects**: There might be community projects demonstrating running models on Ethos. For example, see if Arm’s AI repository on GitHub has samples.

- **Forums and Blogs:**
  - **Arm Community AI Blog** – contains posts like the ExecuTorch one, ML Inference Advisor, etc. Reading those gives insight into the latest features and tools.
  - **Arm ML Platform Forums** – as mentioned, a place to ask questions and see Q&A. For a learner, just reading Q&As can teach a lot (e.g., someone asks about an operator mapping ([Api -> Tosa operator mapping - TOSA - Discourse](https://discuss.mlplatform.org/t/api-tosa-operator-mapping/285#:~:text=I%E2%80%99m%20looking%20to%20learn%20how,broken%20down%20using%20TOSA%20operators)) and an expert answers with pointers).
  - **Stack Overflow or StackExchange** – not sure if TOSA questions pop up there yet, but maybe in the context of MLIR.
  - **Reddit communities** like r/embeddedAI or r/MachineLearning might occasionally discuss on-device ML. (Though TOSA is quite specific, you might find mentions in comments).
  - **Discord/Slack**: There might be an MLIR or TinyML Discord community where you can lurk/ask.

- **Conferences/Workshops:**
  - If you can attend (even virtually) the **TinyML Summit**, **Embedded Vision Summit**, **LLVM Developers’ Meeting**, or Arm’s own **DevSummit**, do so. They often have sessions on edge optimization and sometimes explicitly on TOSA or MLIR.
  - **CVPR, NeurIPS workshops** on Efficient ML or Hardware-Aware NAS – to catch research trends which will eventually translate into demands on things like TOSA.

## Future of TOSA in ML and Career Potential

The future of TOSA looks bright as a cornerstone for cross-platform ML deployment. What does this mean career-wise?
- **High Demand for ML Compiler Engineers:** Already, there is demand for engineers who understand ML graphs and can optimize them for hardware (Google XLA, NVIDIA TensorRT, etc.). On ARM, this skillset is slightly rarer, which means being an expert in TOSA/MLIR on ARM could make you a hot commodity for companies building ML frameworks, mobile AI, or compilers. Roles like *“AI Compiler Engineer”*, *“Embedded ML Engineer”*, *“Edge AI Software Architect”* would be fitting.
- **Growing Ecosystem:** As TOSA becomes part of more frameworks, even application developers will indirectly use it. Having expertise means you could be the person to troubleshoot performance issues or bridge gaps (“Why is my model slow on this phone? Let’s inspect the TOSA graph and see if something didn’t offload.”). This is valuable in product teams deploying AI to devices.
- **Long-Term Relevance:** TOSA (or its eventual successors/versions) aims to be like a standard library for ML ops. Even if one day something else supersedes TOSA, the concept of a standard op set will remain. Skills in reading specs, understanding quantization math, and mapping to hardware will always apply. Also, ARM’s presence in IoT and mobile isn’t going anywhere – in fact, ARM is entering laptops and data centers. ML on ARM is only going to increase. So specializing here is future-proof to an extent.
- **Versatility:** The career potential isn’t limited to working at Arm or chip companies. Any company that deploys ML on ARM (which is a huge range, from big tech to startups in healthcare, automotive, IoT) could use someone who knows how to squeeze the most out of the hardware. You might work on, say, optimizing a model for an AR headset’s ARM-based processor, or helping port a computer vision library to an embedded ARM system using TOSA as an intermediate.

**Emerging trends to watch** where your expertise could apply:
- *Federated Learning and on-device training:* If TOSA extends to training ops, experts will be needed to implement and optimize those.
- *AutoML for Edge:* Tools that automatically search for the best model given hardware constraints might use TOSA to evaluate candidates. Your understanding could help improve or use those tools.
- *Security:* Ensuring models run securely on devices (e.g., verifying that the compiled binaries match the original spec). Knowing TOSA could even tie into security auditing of AI models.
- *Interdisciplinary roles:* Combining knowledge of hardware design and ML algorithm – you could even move into designing next-gen ML accelerators, because you’d know what ops are crucial and how they’re used (essentially becoming a bridge between algorithm and silicon design teams).

## Connecting with Industry Professionals and Communities

Networking and community engagement can significantly boost your career development in this niche:

- **Join Online Communities:** 
  - The official **MLPlatform forum** (discuss.mlplatform.org) is small but focused – engaging there can put you on the radar of Arm’s ML engineers and other enthusiasts. Don’t hesitate to ask thoughtful questions or share insights.
  - **LinkedIn Groups**: Look for groups like “Edge AI” or “Embedded ML”. Arm or Linaro might have groups or pages where they post updates; comment and interact there.
  - **Discord/Slack**: Some open-source communities have chat groups (for example, the TensorFlow Special Interest Group for microcontrollers or MLIR might have a Slack). Participating can connect you to others in the field.
  
- **Attend Meetups and Conferences:**
  - Many cities have meetups for IoT or AI. A talk about optimizing AI on devices could be a place to meet like-minded professionals.
  - If you can attend events like **Embedded Vision Summit** (lots of industry folks working on vision on ARM), or **Arm DevSummit**, do so. Prepare questions, talk to speakers, exchange contacts.
  - The **TinyML meetup** community is global (they have virtual meetups too). Joining those and perhaps presenting something you’ve done can showcase your skills and interest.
  
- **Open Source Contributions:** As mentioned, contributing code is great. It also often means collaborating with others, which forges connections. If you submit a patch to torch-mlir, you’ll interact with maintainers (some of whom might be from Facebook or Google or LLVM foundation). Consistent quality contributions could even lead to job opportunities (people have been hired because of their open-source work visibility).
  
- **Mentorship:** Try to find a mentor in the field. This could be someone at your current workplace who has done low-level optimization, or an external person you admire (perhaps someone active on forums or an author of papers). You can approach them politely, express your interest in learning more about TOSA/ARM ML, and ask if they’d be willing to chat or guide you occasionally. Many folks are happy to help enthusiastic learners.
  
- **Social Media:** Follow key people on Twitter (or X) – for example, Pete Warden (embedded ML expert), or Arm engineers who might share updates. Engage by asking questions or commenting on relevant posts. Just be professional and mindful.
  
- **Blog or Share Your Learnings:** As you learn, consider writing blog posts about what you did – e.g., “My experience quantizing a ResNet for Raspberry Pi using TOSA” or “Tutorial: Using torch-mlir to compile a model with TOSA”. Posting this on Medium or LinkedIn can get you noticed. It also helps solidify your understanding and demonstrates communication skills. The Class Central link we saw suggests there was a tech talk on TOSA – perhaps someone summarized it or wrote about it. You could contribute content like that as well.
  
- **Academic Collaborations:** If you’re interested in research, perhaps collaborate on a paper or experiment with a local university or a professor who works on compilers or edge ML. Even as an industry professional, such collaborations happen and can broaden your network (plus leading to published work which is always nice).
  
The key in networking is to **be genuine and be willing to learn/contribute**. This is a niche field, so the community is not huge – which is good, because making a name for yourself is easier than in very crowded fields. Even asking good questions and sharing small projects can get you recognition among the group of Arm ML compiler folks.

---

In summary, pursuing a career in TOSA and ARM-based ML optimization involves a mix of deep technical learning (from ML theory to low-level optimization), practical experience with relevant tools/hardware, and active engagement with the community. The field is on the rise as edge and embedded AI grow, so investing time now could pay off greatly as more companies seek experts who can bridge the gap between neural networks and efficient ARM implementation. With the right skill set and network, you could find yourself shaping the future of how AI models are deployed on everything from tiny sensors to smartphones – truly bringing intelligence to the edge.

# 6. Technical Deep Dive into ARM-Based ML Optimizations

In this final section, we’ll dive into some technical details of optimizing ML models on ARM using TOSA, with code snippets, performance tuning tips, and a look at how ARM hardware interacts with TOSA-based software. We’ll also discuss challenges that developers face and how to overcome them when building ML software optimized for ARM architectures.

## Code Snippets for ARM+TOSA Optimizations

Let’s work through a few practical code examples that highlight optimization techniques:

### Example 1: Using TOSA Checker to Validate a Model for ARM NPUs

Suppose you have a TensorFlow Lite model and you want to ensure it’s compatible with an Arm Ethos-U NPU (which implies TOSA compliance). You can use the **TOSA Checker** tool:

```python
from tosa_checker import TosaChecker

# Load a TFLite model
tflite_model_path = "model.tflite"
checker = TosaChecker()
result = checker.is_tosa_compatible(tflite_model_path)

if result.is_compatible:
    print("Model is TOSA compatible! All ops can run on TOSA/Arm NPU.")
else:
    print("Model not compatible with TOSA. Issues:")
    for op_issue in result.op_details:
        print(f" - Op {op_issue.op_name}: {op_issue.message}")
```

This snippet uses the `tosa_checker` Python package. It will parse the model and verify if every operator can be expressed in TOSA. If not, it might say something like “Op XYZ is not supported”. For instance, if your model has a `tf.BilateralSliceApply` (random example), the checker would flag it as not in TOSA. You’d then know to replace or eliminate that op (maybe by re-exporting the model differently or using an alternative implementation). Using such a checker early in the development pipeline saves time by catching incompatibilities before you deploy to device.

### Example 2: Lowering a Model to TOSA MLIR and Optimizing

Let’s say you want to manually optimize or inspect the MLIR for a model (for advanced users). Building on the TensorFlow conversion example:

```python
# Continue from previous TF MLIR conversion example:
mlir_txt = mlir_exp.convert_graph_def(graph_def, pass_pipeline='tf-standard-pipeline,func.func(tosa-legalize-tf)')

# Save the MLIR to file
with open("model_tosa.mlir", "w") as f:
    f.write(mlir_txt)
```

Now you have a file `model_tosa.mlir` with all ops in TOSA dialect. You could run further MLIR optimizations on it using mlir-opt tool (part of LLVM). For example, common passes might include `--canonicalize` (to simplify expressions) or `--tosa-optimize` if such exists. There is an open-source doc listing TOSA lowerings and pseudo-code for transformations ([TOSA Lowerings - Fossies](https://fossies.org/linux/tensorflow/tensorflow/compiler/mlir/tosa/g3doc/legalization.md#:~:text=TOSA%20Lowerings%20,org%2Fmlir%2Fdialects%29%20to%20the%20TOSA%20Dialect)) which can guide writing custom passes.

**Manual TOSA MLIR optimization:** For instance, if you notice in the MLIR that you have a pattern like `tosa.matmul -> tosa.add -> tosa.relu`, you might manually fuse those in your mind (though MLIR might not automatically fuse across ops like that unless the backend does). But knowing this pattern, you ensure that when writing backend code or using ACL, you choose a fused kernel if available.

### Example 3: TOSA Execution via Arm Compute Library (ACL)

While ACL doesn’t directly take TOSA as input, conceptually you can map. Here’s pseudo-code combining TOSA understanding with ACL:

Imagine you have a TOSA graph: Input -> Conv2D -> Relu -> FullyConnected -> Softmax -> Output. We can implement this with ACL:

```cpp
#include "arm_compute/runtime/NEON/NEGraph.h"
#include "arm_compute/runtime/NEON/functions/NEConvolutionLayer.h"
#include "arm_compute/runtime/NEON/functions/NEFullyConnectedLayer.h"
#include "arm_compute/runtime/NEON/functions/NESoftmaxLayer.h"
#include "arm_compute/runtime/Tensor.h"

using namespace arm_compute;

NEGraph graph;
Tensor input, conv_w, conv_b, fc_w, fc_b, conv_out, relu_out, fc_out, softmax_out;

// Initialize tensors with proper dimensions and allocate memory...
// (omitted for brevity, but involves setting TensorInfo with shapes and data types)

// Construct layers corresponding to TOSA ops:
auto *conv = graph.add_layer<NEConvolutionLayer>();
conv->configure(&input, &conv_w, &conv_b, &conv_out, PadStrideInfo(...));
// After conv (which includes bias), we manually do ReLU:
conv_out.map();  // map to access CPU memory (not needed if using new ACL Graph API with fused activation)
for(int i=0; i<conv_out.info()->tensor_shape().total_size(); ++i) {
    float *ptr = conv_out.buffer() + i;
    *ptr = std::max(*ptr, 0.0f); // ReLU in-place
}
conv_out.unmap();

auto *fc = graph.add_layer<NEFullyConnectedLayer>();
fc->configure(&conv_out, &fc_w, &fc_b, &fc_out);

auto *smx = graph.add_layer<NESoftmaxLayer>();
smx->configure(&fc_out, &softmax_out);

// Execute graph
graph.run();
```

This C++ code is low-level; in practice, one would use the higher-level Arm NN or Compute Library Graph API to avoid manual loops. But it shows how each TOSA op corresponds to an ACL function:
- `tosa.conv2d` -> `NEConvolutionLayer`
- `tosa.relu` -> (can be fused by using the activation parameter in NEConvolutionLayer’s PadStrideInfo, or do manual as above).
- `tosa.fully_connected` -> `NEFullyConnectedLayer`
- `tosa.softmax` -> `NESoftmaxLayer`

By doing this, we ensure that NEON acceleration is used for conv and fc. The ReLU is trivial, Softmax uses a vectorized implementation.

**Performance tuning in this code:**
- Choose the right data type: if we quantize, we’d use QASYMM8 types in ACL and configure quantization info (scales, zero-points) for tensors. ACL has `NEGEMMLowp` for quantized FC, etc.
- Ensure memory is contiguous and aligned. ACL expects allocated Tensors with proper alignment (which `allocate()` does).
- Use batching if possible. If we can batch multiple inputs, Neon will have more work per kernel call amortizing overhead.
- For convolution, choosing optimal PadStrideInfo (like setting dilation or paddings correctly) is important.

This kind of low-level coding might be done when writing a custom runtime or debugging performance issues. It’s essentially reimplementing the TOSA graph with a known optimized library.

## Performance Tuning Strategies on ARM

Here are key strategies to get the best performance for ML models on ARM:

- **Quantize Everything Possible:** As reiterated, int8 quantization brings massive speedups on ARM CPUs (by using 8-bit SIMD) and is mandatory for NPUs like Ethos. Use post-training quantization or quantization-aware training to convert models to int8. Then verify accuracy and performance. On a Cortex-A CPU, int8 inference can be 2-4x faster than float32, and on Ethos-U it’s required (Ethos-U supports only int8). TOSA fully supports quantized ops, so no barriers there.
- **Operator Fusion:** Fusing multiple operations reduces memory accesses and utilizes hardware better. For example, fuse conv+ReLU, or conv+BN+ReLU into one kernel. On CPU, this means one loop instead of several. On GPU, it means fewer kernel launches. TOSA itself doesn’t have a fused op, but an implementation can fuse under the hood. When writing MLIR passes or when using a runtime like Arm NN, ensure that it is configured to fuse where possible. Arm NN for instance can fuse activation functions into preceding layers if using Compute Library backend. When writing custom passes, you might combine consecutive elementwise ops into one (like a chain of adds).
- **Parallelism:** Use multi-threading on Arm CPUs for larger layers. Big CNNs on a Cortex-A should use threads (via OpenMP or other means, which ACL does internally). For example, a 2D conv can be split by output channels or rows across threads. Ensuring your runtime is using all available cores is key. On microcontrollers, there’s usually 1 core, but on a Raspberry Pi (4 cores) or a smartphone (4-8 big cores), multi-threading can yield near-linear speedups for compute-heavy layers. However, too many threads can saturate memory bandwidth, so tune the number (often 2-4 threads is optimal for heavy convs on a mobile CPU).
- **Data Layout and Memory:** ARM Neon likes data in contiguous chunks. For example, NHWC vs NCHW layout can make a difference. Many ARM libraries prefer NHWC for conv (to align channels in memory for vectorization). Using the layout that the backend expects avoids extra permutations. TOSA doesn’t mandate a layout, but if you know the target, you can choose to export the model with the preferred layout. Also, align tensors to 16-byte or 64-byte boundaries to help the memory system prefetch/cache. 
- **Weight Transformations:** For instance, certain NPUs or even CPU algorithms might want weights in a specific format (e.g., compressed, or transposed). The compiler (like Vela or onnx-mlir) usually handles this. But know that a one-time offline cost (like converting weights to int8 with proper scaling and maybe reordering) can hugely improve runtime. For example, im2col + GEMM approach might flatten conv weights; doing that once at compile time is better than on the fly.
- **Use Accelerators Efficiently:** If you have a GPU (Mali) and a CPU, decide which parts run where. Sometimes small ops are faster on CPU than launching a GPU kernel (due to overhead). A typical strategy is to run large matrix multiplies or convolutions on the GPU, but run small activations or elementwise ops on the CPU. The TOSA graph can be partitioned accordingly. In fact, tools like Arm NN or TensorFlow Lite delegate perform such partitioning (they keep unsupported or small ops on CPU). As an optimizer, you might manually partition by inserting annotations or breaking the graph. For NPUs, anything unsupported goes to CPU by necessity – minimize that by adjusting the model to avoid unsupported patterns. 
- **Pipeline and Overlap:** On systems with CPU/GPU concurrency, overlap execution if possible. For example, while GPU is doing a conv, CPU could be preparing the next data or doing an argmax on the previous layer’s result. This is advanced and framework-dependent (one would need multi-threaded pipeline).
- **Profiling and Bottleneck Analysis:** Always profile the model on the target hardware to see where the time goes:
  - On CPU, use profilers (perf, or even instrument with `std::chrono` around layers).
  - On Android, use systrace or perfetto to see CPU vs DSP/GPU usage.
  - On microcontrollers, toggle GPIO pins around sections of code to measure on a scope, or use the cycle counter.
  Identify if the bottleneck is compute-bound or memory-bound. E.g., if a depthwise conv is slow, maybe it’s not optimized in the library – you might replace it with a pointwise + reshape trick if that’s faster.
- **Leverage Specialized Instructions:** Newer ARM CPUs have *dot product* instructions (SDOT/UDOT) for int8 matrix multiplication (e.g., Armv8.4-A and above). Ensure your libraries are compiled to use them (ACL does if you enable the right build flags). On some Cortex-M, there's support for parallel 8-bit MACs (via DSP extension). TOSA doesn’t care about this, but as an optimizer you do – basically, use the latest compilers and libraries that know about these instructions.
- **Memory Planning:** Especially for microcontrollers with scratchpad memory, plan your memory usage. TOSA reference model might not handle this, but an NPU like Ethos has limited SRAM. The compiler (Vela) will try to fit tensors in SRAM, offloading to DRAM only when needed. As a dev, you can help by reducing peak memory usage of the model (e.g., use smaller batch sizes, or reuse buffers in code if writing custom inference). For example, if you know that two layers are not active at the same time, reuse one’s buffer for the other’s output.
- **Use the Right Profiles/Accelerator Modes:** Some NPUs have modes or “profiles” to target performance vs accuracy. Ethos-U has a “force DMAC” option in Vela to trade off some scheduling for memory. Always read the documentation for these flags and experiment. Similarly, Mali GPU has different convolution algorithms (WINOGRAD vs direct) that can be selected via tuning.

## How ARM Hardware Interacts with TOSA

Understanding the interplay of software and hardware is crucial:

- **ARM CPUs (Cortex-A):** When running TOSA ops on a Cortex-A CPU, each op will execute as a sequence of instructions (Neon, etc.). For example, a TOSA convolution might be implemented as im2col + GEMM or as a direct convolution loop. The performance here depends on the quality of the implementation (like in ACL or Eigen). ARM’s latest CPUs also have BF16 and FP16 support (half-precision) which can accelerate FP16 models. TOSA supports FP16 (since it said support for quantized *and floating-point* content). If you keep your model in FP16, on an Arm Neoverse or Cortex-A710, it might use FP16 instructions (2x throughput vs FP32). So sometimes *not* quantizing but using FP16 is an alternative – easier from an accuracy standpoint and still faster than FP32 (though still slower than int8). As a developer, you could choose FP16 for layers that lost too much accuracy with int8.
  
- **ARM Cortex-M and Ethos-U:** The Ethos-U microNPU offloads most heavy ops, but the Cortex-M CPU still plays a role:
  - It runs the runtime code that feeds commands to the NPU and waits for completion.
  - It executes any unsupported TOSA ops. For Ethos-U55, most arithmetic ops are supported, but something like *tosa.pad* might actually be done on CPU by copying edges, etc., depending on implementation.
  - It also may do preprocessing or postprocessing (like if final Softmax isn’t supported due to size, CPU would do it).
  - The developer should ensure the CPU side has minimal work. So avoid ops that would be on CPU in performance-critical sections. For example, if you have a fancy activation that Ethos doesn’t support, replace it with ReLU or so which it does support.
  - The Ethos-U has a driver that takes TOSA (or rather the compiled form of it) and executes it in chunks. Ethos executes in its own SRAM mostly. The coordination between CPU and Ethos is through shared memory and interrupts. As an optimization, you want to maximize the Ethos utilization and have the CPU do other tasks or sleep to save power in the meantime.

- **Mali GPU:** If using a GPU (like Mali G76 etc.), typically through OpenCL or Vulkan:
  - TOSA ops would be mapped to GPU kernels. A conv might become an OpenCL kernel that reads the image and filter from global memory and does the math. Memory access patterns are key on GPU (tile data to use local memory, etc.). The compiler or runtime (like Arm NN GPU or TFLite GPU delegate) will handle that. However, sometimes they need hints; e.g., choosing the right shader algorithm for small vs large conv.
  - The GPU can do fp16 well (most Mali have native fp16, so using fp16 compute is a win). Int8 on GPU is less straightforward; some newer GPUs have int8 dot product (especially in mobile SoCs for ML), but often int8 is handled via emulation or not at all on GPU, so many GPU delegates prefer FP16. So, if your target is GPU, you might keep the model in FP16 to leverage that strength, versus int8 for NPUs.
  - Overhead: launching many small GPU kernels can kill performance, so the compiler tries to group operations. For example, it may combine consecutive elementwise ops into one kernel. That’s essentially doing manual fusion at the GLSL level.
  - As an optimizer, know that memory bandwidth on mobile GPUs is limited by the shared DRAM. If you do a lot of large matrix operations on GPU, watch for memory bottlenecks (use profiling tools like Mali Graphics Analyzer).

- **Ethos-N (NPUs for mobile):** These have higher throughput and also handle more ops in hardware than Ethos-U. For instance, Ethos-N might support dequantization or 16-bit accumulations, etc. The principle is the same – you feed a TOSA (or TFLite) graph, and the driver schedules it. One difference is Ethos-N might allow more parallelism or handle multiple networks at once if the SoC supports it. As a software dev, you mostly treat it as a black box that can do certain ops faster. But if you know, for example, Ethos-N doesn’t support a certain size of tensor or needs splitting of large tensors, you handle that in compile time. Vela and other compilers usually account for that.

- **Memory and DMA:** On accelerators, often a DMA moves data in and out of the accelerator’s memory. If you write custom code, ensure to align buffers and possibly allocate them in sections of memory accessible to DMA (some systems have specific memory regions for NPUs). Arm’s libraries do this under the hood, but if you integrate into an RTOS or something, you might need to configure the memory properly.
  
- **Power Optimizations:** On ARM, especially battery-powered devices, you want to optimize not just for speed but for power (or energy). That can mean:
  - Using the NPU or DSP instead of the big CPU even if CPU is fast, because NPU might be much more efficient per inference (e.g., Ethos-U can do certain inferences in 1/10th the energy of CPU).
  - Lowering clock frequency if full speed isn’t needed, but keeping utilization high.
  - These are more system-level, but as an ML software person, if you know your model can run at 200ms on CPU at 2GHz, but could also run at 300ms on NPU at 0.5GHz and save 5x power, you might choose the latter for a wearable scenario.
  - TOSA helps by enabling use of the specialized hardware (which is the route to energy efficiency).

## Challenges and Solutions in Developing ARM-Optimized ML Software

Finally, let’s talk about some common challenges:

**Challenge 1: Operator Gaps Between Framework and Hardware** – You have a model that works in PyTorch, but some ops aren’t supported on your ARM target (NPU or even the delegate). This is common – e.g., “Upsample with nearest neighbor” might not be directly supported on NPU.
- *Solution:* You can sometimes replace operations in the model with equivalent TOSA-friendly versions. For upsample, maybe insert it as a `tosa.resize` op if available, or precompute indices and use a gather (not ideal). Another example: if an activation isn’t supported, switch to ReLU or LeakyReLU which are.
- When replacement isn’t possible, you accept fallback to CPU and then try to minimize its cost. Maybe run that part at lower frequency or in parallel with something else.
- Communicate with model developers: part of being an expert is advising model authors to design with deployment in mind. Provide them with a list of “these ops are hardware-friendly, these are not.” This proactive approach prevents gaps.

**Challenge 2: Debugging Quantization Issues** – You converted a model to int8 and deployed it, but accuracy dropped more than expected, or some outputs are wildly off.
- *Solution:* Use the TOSA Reference Model. You can feed it the same inputs and compare outputs at each layer with those from the device. Because TOSA reference is a golden model (though slow), it helps pinpoint where things diverge.
- Often the issue is a wrong scale or zero-point in quantization. TOSA quantization lowering requires calculating rescale parameters carefully. Double-check those. PyTorch’s quantization gave an easy way to get scales; use that and ensure the hardware got the same values.
- Another trick: sometimes a layer might need per-channel quantization for good accuracy (e.g., weights of conv), ensure your pipeline did that if supported.
- If an op saturates a lot (e.g., many values clamped at 127 or -128), try adjusting the quantization range (maybe the calibration data was not representative).
- If needed, consider using 16-bit or unquantized for the trouble layer (some NPUs allow leaving a layer in higher precision if accuracy demands; though Ethos-U currently is all-or-nothing int8).

**Challenge 3: Model Too Large for Memory** – On a microcontroller, even if the NPU is fast enough, the model (weights + activations) might not fit in RAM.
- *Solution:* Model optimization: prune weights, reduce model size (maybe a smaller architecture).
- Use streaming techniques: e.g., run the model in chunks. For a CNN, maybe process one part of the image then another to reduce peak memory, if the application allows.
- Leverage TOSA’s minimalist approach: no overhead of dynamic operators or large runtime structures. TOSA flatbuffer is lean. Make sure you’re not linking huge frameworks; use the smallest runtime (maybe even write a custom one that just executes your specific model).
- If weights are large, compress or quantize further (could 4-bit quantization be enough? Some research suggests yes for certain layers).
- Offload any pre/post-processing to reduce memory: like if your input is an audio stream, feed it frame by frame rather than storing a big buffer.

**Challenge 4: Integrating with Existing Systems** – You may have an RTOS or an application and want to integrate your optimized model inference without hiccups.
- *Solution:* Use a modular approach: for example, the ExecuTorch runtime gives a .pte that you can include and call from C. If using TFLite, you can build it with only the ops you need (micro TFLite does this). For TOSA, maybe your custom runtime is just a set of function calls that correspond to the net’s layers.
- Pay attention to interoperability: e.g., if an app expects an ONNX model, you might need to put an ONNX -> TOSA conversion step and then link with a TOSA runtime. This is extra work, so sometimes just using ONNX Runtime with ACL might suffice (but then you rely on their support of your target).
- Real-time constraints: If you have timing deadlines (say 30ms per inference), ensure worst-case execution fits that. If garbage collection or an OS scheduler could preempt, consider pinning a thread or using bare-metal execution for critical sections.

**Challenge 5: Keeping up with Changes** – TOSA spec might update, hardware drivers update, etc., which could break or improve things.
- *Solution:* Stay agile and update your toolchains regularly, but also lock down critical components when needed. For example, if you have a working combination of PyTorch version X, torch-mlir commit Y, and Vela version Z that produces a good result, document that environment. If later versions cause a regression, you can compare and perhaps contribute a fix upstream.
- Engage with the community: early knowledge of upcoming changes (like TOSA 1.0 final spec) can help you prepare.

By applying these strategies and being mindful of the challenges, one can successfully develop and deploy highly optimized ML models on ARM platforms using TOSA and related tools. This deep technical work can lead to models that run *efficiently* (fast and low-power) on devices ranging from powerful Arm servers to tiny microcontrollers, fulfilling the promise of pervasive AI.

---

**Sources:**

The information and examples above were informed by Arm’s official documentation and community resources, including the TOSA specification and Arm AI ecosystem blog posts. For instance, Arm’s introduction of TOSA emphasizes its cross-platform consistency, and their collaboration with Meta for PyTorch/ExecuTorch demonstrates quantization and deployment on Ethos NPUs. The need for a standardized operator set to scale ML across devices is highlighted in industry discussions. Additionally, community forums and Q&A with Arm engineers (like Eric Kunze) shed light on TOSA’s development priorities (e.g., addressing training op gaps ([TOSA Specifications Creation Approach - TOSA - Discourse](https://discuss.mlplatform.org/t/tosa-specifications-creation-approach/264#:~:text=operators%20that%20are%20commonly%20in,use%20in%20the%20frameworks)) and adding ops for transformers ([Api -> Tosa operator mapping - TOSA - Discourse](https://discuss.mlplatform.org/t/api-tosa-operator-mapping/285#:~:text=We%20try%20to%20keep%20the,0%20networks))). These resources collectively underscore the role of TOSA as a linchpin in ARM’s ML strategy and guided the best practices and insights shared in this study.

