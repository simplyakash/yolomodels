7️⃣ YOLOv8 (2023)

Major change: Anchor-free detection

Backbone → C2f modules

Neck     → PAN-FPN

Head     → Anchor-free head

Architecture:

Image

↓

C2f Backbone

↓

FPN + PAN

↓

Anchor-free Head

Benefits:

Simpler training and Better generalization



Key Features of YOLOv8 (Compared to YOLOv5)

🔹 1. Advanced Backbone and Neck Architectures

YOLOv5:
Uses a standard backbone and neck design
Good feature extraction, but limited in highly complex scenes

YOLOv8:
Introduces more advanced and optimized architectures
Enhances feature extraction across multiple scales
Impact:
Better detection of small and overlapping objects
Improved overall accuracy and robustness

🔹 2. Anchor-Free Split Head

YOLOv5:
Uses an anchor-based detection approach
Requires manual tuning of anchor boxes

YOLOv8:
Uses an anchor-free split Ultralytics head
Eliminates dependency on predefined anchors
Impact:
Simpler training pipeline
Reduced hyperparameter tuning
More accurate object localization

🔹 3. Optimized Accuracy–Speed Tradeoff

YOLOv5:
Strong real-time performance
Balanced speed and accuracy

YOLOv8:
Further improves the balance
Maintains or improves speed while increasing accuracy
Impact:
More efficient real-time detection
Better performance on edge and production systems

🔹 4. Improved Pretrained Model Variants

YOLOv5:
Offers multiple model sizes (e.g., small, medium, large)

YOLOv8:
Provides more refined and better-optimized pretrained models
Improved scaling across tasks and hardware
Impact:
Easier model selection
Better adaptability for different use cases

📊 Summary Comparison

| **Feature**             | **YOLOv5**             | **YOLOv8**                |
| ----------------------- | ---------------------- | ------------------------- |
| **Architecture**        | Standard               | More advanced & optimized |
| **Detection Head**      | Anchor-based           | Anchor-free split head    |
| **Training Complexity** | Higher (anchor tuning) | Lower (no anchors needed) |
| **Speed vs Accuracy**   | Balanced               | More optimized            |
| **Model Variants**      | Multiple               | More refined & scalable   |





✅ Conclusion

YOLOv8 improves upon YOLOv5 by:

Simplifying the detection pipeline

Enhancing accuracy

Maintaining efficient real-time performance

👉 Overall, YOLOv8 is more modern, flexible, and easier to use for real-world object detection tasks.

**YOLOv8 Architecture (Layer-by-Layer Overview with Shapes & Filters)**

⚠️ Note: YOLOv8 uses a dynamic and scalable architecture (n, s, m, l, x variants).
The shapes and filter sizes below are based on a YOLOv8-S (small) model with input size 640×640×3.

📌 High-Level Architecture

YOLOv8 is divided into three main parts:

**Backbone → Feature extraction** ||
**Neck → Feature aggregation (PAN-FPN)** ||
**Head → Detection (anchor-free)**

🔹 1. Input Layer
Input Image Shape:
640×640×3

🔹 2. Backbone (Feature Extraction)

Stage 1: Initial Convolution
Operation: Conv + BN + SiLU
Filters: 32
Kernel: 3×3
Stride: 2
Output Shape:
320×320×32

Stage 2: Downsampling + C2f Block

Downsampling Conv:Filters: 64, Stride: 2,   Output Shape: 160×160×64

C2f Block:(Cross Stage Partial with 2 Convolutions - Fast)

Bottleneck layers (lightweight CSP variant)

**C2f Block (Step-by-Step with Shapes)**

Step 1: Input to C2f  :160×160×64

Step 2: Initial 1×1 Convolution (Channel Adjustment),
Purpose: Prepare features for splitting,
Output channels remain 64
160×160×64

Step 3: Channel Split

Split channels into two parts: 64→32+32

Part A (skip path):

160×160×32

Part B (processed path):

160×160×32

Step 4: Bottleneck Processing (on Part B)

Assume n = 2 bottleneck layers (typical for small models)

Each bottleneck:

1×1 Conv (reduce/transform)

3×3 Conv (feature extraction)

Residual connection

After Bottleneck 1
160×160×32

After Bottleneck 2
160×160×32

👉 Important:

Shape remains the same
Only features are refined

Step 5: Concatenation

Concatenate:

Skip connection (Part A)
Outputs from bottlenecks
32+32+32=96 channels

So:

160×160×96

Step 6: Final 1×1 Convolution (Fusion)
Reduce channels back to 64

160×160×96→160×160×64 Each of the 64 output channels is a weighted combination of all 96 input channels

🔹 Final Output of C2f
160×160×64

✔ Matches expected Stage 2 output

Output Shape:
160×160×64

Stage 3: Downsampling + C2f
Conv:
Filters: 128, Stride: 2
C2f Block
Output Shape:
80×80×128

Stage 4: Downsampling + C2f
Conv:
Filters: 256, Stride: 2
C2f Block
Output Shape:
40×40×256

Stage 5: Downsampling + C2f
Conv:
Filters: 512, Stride: 2
C2f Block
Output Shape:
20×20×512

**SPPF (Spatial Pyramid Pooling - Fast)**

Multiple pooling operations (different receptive fields)
Enhances contextual understanding
Output Shape:
20×20×512

1. Spatial Dimensions (20×20)
These correspond to the height and width of the feature map

The original input image (typically 640×640) is progressively downsampled

Downsampling Process

YOLOv8 reduces spatial size using stride-2 convolutions:
640→320→160→80→40→20

So:

20×20 means the feature map is 32× smaller than the input.Each cell represents a large region of the original image

2. Channel Dimension (512)
This represents the number of feature channels (filters).Each channel captures a different type of learned feature:

**Edges
Textures
Object parts
Semantic patterns**

👉 Think of it as:

**512 different “views” or “detectors” looking at the same 20×20 grid**


**📦 SPPF (Spatial Pyramid Pooling – Fast) in YOLOv8**
🧠 Overview

SPPF (Spatial Pyramid Pooling – Fast) is used in YOLOv8 to:

Increase receptive field
Capture multi-scale spatial context
Maintain computational efficiency

Instead of multiple large pooling kernels, SPPF:

Uses a single $5 \times 5$ max pooling layer repeatedly

⚙️ Input Specification
| **Parameter** | **Description**             |
| ------------- | --------------------------- |
| $B$           | $\text{Batch size}$         |
| $C$           | $\text{Number of channels}$ |
| $H, W$        | $\text{Spatial dimensions}$ |

| **Parameter** | **Description**    |
| ------------- | ------------------ |
| `B`           | Batch size         |
| `C`           | Channels           |
| `H, W`        | Spatial dimensions |


Example Input

(1, 512, 20, 20)

🔁 Step-by-Step Computation

✅ Step 1: Channel Reduction (1×1 Convolution)

(1, 512, 20, 20) → (1, 256, 20, 20)

| Input              | Output             | Operation    |
| ------------------ | ------------------ | ------------ |
| `(1, 512, 20, 20)` | `(1, 256, 20, 20)` | `Conv (1×1)` |



Let:

$x$

✅ Step 2: First Max Pooling

y1 = MaxPool(x)

| **Input Shape**    | **Output Shape**   |
| ------------------ | ------------------ |
| $(1, 256, 20, 20)$ | $(1, 256, 20, 20)$ |

✅ Step 3: Second Max Pooling

y2 = MaxPool(y1)

| **Input Shape**    | **Output Shape**   |
| ------------------ | ------------------ |
| $(1, 256, 20, 20)$ | $(1, 256, 20, 20)$ |

✅ Step 4: Third Max Pooling

y3 = MaxPool(y2)

| **Input Shape**    | **Output Shape**   |
| ------------------ | ------------------ |
| $(1, 256, 20, 20)$ | $(1, 256, 20, 20)$ |


| **Feature Map** | **Receptive Field** |
| --------------- | ------------------- |
| $x$             | $\text{Original}$   |
| $y_1$           | $5 \times 5$        |
| $y_2$           | $9 \times 9$        |
| $y_3$           | $13 \times 13$      |


| Feature | Receptive Field |
| ------- | --------------- |
| `x`     | Original        |
| `y1`    | `5×5`           |
| `y2`    | `9×9`           |
| `y3`    | `13×13`         |


✅ Step 5: Concatenation

[x, y1, y2, y3]

(1, 256×4, 20, 20) = (1, 1024, 20, 20)

| Input Channels | Output Channels | Shape               |
| -------------- | --------------- | ------------------- |
| `256×4`        | `1024`          | `(1, 1024, 20, 20)` |



✅ Step 6: Final 1×1 Convolution

(1, 1024, 20, 20) → (1, 512, 20, 20)

| **Input Shape**     | **Output Shape**   | **Operation**               |
| ------------------- | ------------------ | --------------------------- |
| $(1, 1024, 20, 20)$ | $(1, 512, 20, 20)$ | $\text{Conv } (1 \times 1)$ |


🎯 Final Output

(1, 512, 20, 20)


🔄 Full Pipeline Summary
| **Step**       | **Operation**                  | **Output Shape**    |
| -------------- | ------------------------------ | ------------------- |
| $\text{Input}$ | $-$                            | $(1, 512, 20, 20)$  |
| $1$            | $\text{Conv } (1 \times 1)$    | $(1, 256, 20, 20)$  |
| $2$            | $\text{MaxPool } (5 \times 5)$ | $(1, 256, 20, 20)$  |
| $3$            | $\text{MaxPool } (5 \times 5)$ | $(1, 256, 20, 20)$  |
| $4$            | $\text{MaxPool } (5 \times 5)$ | $(1, 256, 20, 20)$  |
| $5$            | $\text{Concat 1234}$                | $(1, 1024, 20, 20)$ |
| $6$            | $\text{Conv } (1 \times 1)$    | $(1, 512, 20, 20)$  |

| Step  | Operation | Output              |
| ----- | --------- | ------------------- |
| Input | -         | `(1, 512, 20, 20)`  |
| 1     | Conv      | `(1, 256, 20, 20)`  |
| 2     | MaxPool   | `(1, 256, 20, 20)`  |
| 3     | MaxPool   | `(1, 256, 20, 20)`  |
| 4     | MaxPool   | `(1, 256, 20, 20)`  |
| 5     | Concat    | `(1, 1024, 20, 20)` |
| 6     | Conv      | `(1, 512, 20, 20)`  |



| **Stage** | **Operation**         | **Input Shape**    | **Output Shape**   | **Kernel Size** | **Stride** | **Padding** | **Effective Receptive Field** |
| --------- | --------------------- | ------------------ | ------------------ | --------------- | ---------- | ----------- | ----------------------------- |
| $x$       | Input                 | $(1, 256, 20, 20)$ | $(1, 256, 20, 20)$ | $-$             | $-$        | $-$         | $1 \times 1$                  |
| $y_1$     | $\text{MaxPool}(x)$   | $(1, 256, 20, 20)$ | $(1, 256, 20, 20)$ | $5 \times 5$    | $1$        | $2$         | $5 \times 5$                  |
| $y_2$     | $\text{MaxPool}(y_1)$ | $(1, 256, 20, 20)$ | $(1, 256, 20, 20)$ | $5 \times 5$    | $1$        | $2$         | $9 \times 9$                  |
| $y_3$     | $\text{MaxPool}(y_2)$ | $(1, 256, 20, 20)$ | $(1, 256, 20, 20)$ | $5 \times 5$    | $1$        | $2$         | $13 \times 13$                |

| Stage | Operation | Input              | Output             | Kernel | Stride | Padding | RF      |
| ----- | --------- | ------------------ | ------------------ | ------ | ------ | ------- | ------- |
| `x`   | Input     | `(1, 256, 20, 20)` | `(1, 256, 20, 20)` | -      | -      | -       | `1×1`   |
| `y1`  | MaxPool   | `(1, 256, 20, 20)` | `(1, 256, 20, 20)` | `5×5`  | 1      | 2       | `5×5`   |
| `y2`  | MaxPool   | `(1, 256, 20, 20)` | `(1, 256, 20, 20)` | `5×5`  | 1      | 2       | `9×9`   |
| `y3`  | MaxPool   | `(1, 256, 20, 20)` | `(1, 256, 20, 20)` | `5×5`  | 1      | 2       | `13×13` |



🚀 SPPF vs Traditional SPP
| **Feature**               | **SPP**                               | **SPPF**                      |
| ------------------------- | ------------------------------------- | ----------------------------- |
| $\text{Pooling Strategy}$ | $\text{Multiple kernels } (5, 9, 13)$ | $\text{Repeated } 5 \times 5$ |
| $\text{Computation}$      | $\text{Expensive}$                    | $\text{Efficient}$            |
| $\text{Implementation}$   | $\text{Parallel}$                     | $\text{Sequential}$           |
| $\text{Speed}$            | $\text{Slower}$                       | $\text{Faster}$               |

| Feature          | SPP                         | SPPF           |
| ---------------- | --------------------------- | -------------- |
| Pooling Strategy | Multiple kernels `(5,9,13)` | Repeated `5×5` |
| Computation      | Expensive                   | Efficient      |
| Implementation   | Parallel                    | Sequential     |
| Speed            | Slower                      | Faster         |



🧠 Key Takeaways

Spatial dimensions unchanged
Channel expand → compress
Multi-scale context


🔹 3. Neck (Feature Aggregation - PAN-FPN)

🔺 Upsample + Concatenate  with corresponding C2f Output ($P5 \rightarrow P4$)
Upsample:

(20×20 → 40×40)

Concatenate with Stage 4 output

Output Shape:

40×40x512

C2f Block (feature refinement)

Output Shape:

40×40x256

🔺 Upsample + Concatenate ($P4 \rightarrow P3$)
Upsample:

(40x40 → 80x80)

Concatenate with Stage 3 output

Output Shape:

80×80x256

C2f Block

Output Shape:

80×80x128

🔻 Downsample Path (PAN)
Conv (stride $2$):

(80x80 → 40x40)

Concatenate with previous P4

Output Shape:

40x40x256

C2f Block

Output Shape:

40x40x256

🔻 Final Downsample

Conv (stride $2$):

(40x40 → 20x20)

Concatenate with P5

Output Shape:

20x20x512

C2f Block

Output Shape:

20x20x512

🔹 4. Head (Detection Layer)

YOLOv8 uses an anchor-free detection head.

| **Scale**          | **Feature Map Size** | **Channels** |
| ------------------ | -------------------- | ------------ |
| **Small Objects**  | $$80 \times 80$$     | 128          |
| **Medium Objects** | $$40 \times 40$$     | 256          |
| **Large Objects**  | $$20 \times 20$$     | 512          |

| Scale  | Feature Map | Channels |
| ------ | ----------- | -------- |
| Small  | `80×80`     | 128      |
| Medium | `40×40`     | 256      |
| Large  | `20×20`     | 512      |


Detection Output

For each grid cell:

Bounding box: x,y,w,h
Objectness score
Class probabilities

Final output per scale:
S×S×(C+4+1)

Where:

S = grid size
C = number of classes
📊 Summary Table
Stage	Output Shape	Filters
Input	640×640×3	-
Conv1	320×320×32	32
Stage 2	160×160×64	64
Stage 3	80×80×128	128
Stage 4	40×40×256	256
Stage 5	20×20×512	512
Neck Outputs	80,40,20 scales	128–512

| Stage  | Output       | Filters |
| ------ | ------------ | ------- |
| Input  | `640×640×3`  | -       |
| Conv1  | `320×320×32` | 32      |
| Stage2 | `160×160×64` | 64      |
| Stage3 | `80×80×128`  | 128     |
| Stage4 | `40×40×256`  | 256     |
| Stage5 | `20×20×512`  | 512     |
| Neck   | multi-scale  | 128–512 |


✅ Key Architectural Improvements over YOLOv5
C2f modules replace older CSP blocks → better gradient flow
Anchor-free head → simpler and more accurate
Improved feature fusion (PAN-FPN)
SPPF optimization for faster computation
🚀 Conclusion

YOLOv8’s architecture is modular, efficient, and optimized for real-time detection, offering:

Better accuracy
Faster inference
Simpler training pipeline


Abbreviations Used in YOLOv8 Architecture

This section explains all the key abbreviations used in the YOLOv8 architecture in a clear, README-ready format.

🔹 Core Architecture Terms
Backbone
The part of the network responsible for feature extraction from the input image
Converts raw pixels into meaningful feature maps
Neck
Connects backbone and head
Combines features from different scales for better detection
Head
Final part of the model
Produces predictions:
Bounding boxes
Object confidence
Class probabilities
🔹 Layer & Block Abbreviations
Conv (Convolution Layer)
Performs convolution operation on input
Extracts spatial features
BN (Batch Normalization)
Normalizes activations during training
Improves stability and speeds up convergence
SiLU (Sigmoid Linear Unit)
Activation function used in YOLOv8
Also called Swish

Formula:

SiLU(x)=x⋅σ(x)
C2f (Cross Stage Partial with 2 Convolutions - Fast)
Improved version of CSP block
Splits feature maps and processes them efficiently

Purpose:

Better gradient flow
Lower computation cost
Faster training
SPPF (Spatial Pyramid Pooling - Fast)
Applies multiple pooling operations with different receptive fields

Purpose:

Captures multi-scale context
Improves detection of objects of different sizes
🔹 Architecture Concepts
PAN (Path Aggregation Network)
Bottom-up feature fusion

Purpose:

Improves localization by passing low-level features upward
FPN (Feature Pyramid Network)
Top-down feature fusion

Purpose:

Combines high-level semantic features with low-level details
PAN-FPN
Combination of PAN + FPN

Purpose:

Strong multi-scale feature aggregation

🔹 Detection Concepts
Anchor-Based
Uses predefined bounding boxes (anchors)
Requires tuning
Anchor-Free
Predicts bounding boxes directly without anchors

Used in YOLOv8

Grid Cell
Image is divided into grid cells
Each cell predicts objects within its region

🔹 Output Terms
Bounding Box (x,y,w,h)
x,y → center of box
w → width
h → height
Objectness Score
Probability that an object exists in a box
Class Probability (C)
Probability distribution over object classes
Final Output Shape
S×S×(C+4+1)

Where:

S = grid size
C = number of classes
4 = bounding box values
1 = objectness score
🔹 Training & Scaling Terms
n, s, m, l, x

Model size variants:

n → nano (smallest, fastest)
s → small
m → medium
l → large
x → extra large (highest accuracy, slowest)

**SPPF (Spatial Pyramid Pooling – Fast) in YOLOv8**

🔹 What is SPPF?

SPPF (Spatial Pyramid Pooling – Fast) is a module used in YOLOv8’s backbone to capture multi-scale spatial information efficiently.

It allows the network to “see” objects at different receptive fields
Helps detect objects of varying sizes without increasing input resolution

🔹 Why is SPPF Needed?

In object detection:

Small objects require fine details
Large objects require broader context

SPPF solves this by combining features from multiple receptive fields in a computationally efficient way.

🔹 How SPPF Works (Step-by-Step)

1. Input Feature Map

Example shape: 20×20×512

2. Max Pooling (Repeated Sequentially)

Apply MaxPool with kernel: 5×5
Stride = 1 (no downsampling)

Instead of parallel pooling (like older SPP), SPPF applies pooling sequentially:

First pooling → captures local context

Second pooling → larger receptive field

Third pooling → even larger context

3. Feature Concatenation

All outputs are concatenated along the channel dimension:

Output Channels=4×C

Where:

C = input channels

4. Final Convolution
5. 
A 1×1 convolution reduces channels back:
4C→C

🔹 Receptive Field Intuition

Each pooling layer increases the effective receptive field:

Stage	Receptive Field Size
Original	1×1
Pool 1	5×5
Pool 2	9×9
Pool 3	13×13

| Stage    | Receptive Field Size      |
| -------- | ------- |
| Original | `1×1`   |
| Pool1    | `5×5`   |
| Pool2    | `9×9`   |
| Pool3    | `13×13` |


👉 This allows the model to understand both local and global context.

🔹 SPP vs SPPF

Aspect	SPP (YOLOv5)	SPPF (YOLOv8)
Pooling Style	Parallel pooling	Sequential pooling
Speed	Slower	Faster
Memory Usage	Higher	Lower
Output Quality	Good	Similar or better

| Aspect | SPP      | SPPF       |
| ------ | -------- | ---------- |
| Pooling Style  | Parallel | Sequential |
| Speed  | Slow     | Fast       |
| Memory | High     | Low        |



🔹 Advantages of SPPF
✅ Faster computation than traditional SPP
✅ Captures multi-scale features efficiently
✅ Low computational overhead
✅ Improves detection of large objects and context
🔹 Where SPPF is Used in YOLOv8
Located at the end of the backbone
Processes the deepest feature map before passing it to the neck
📊 Example Flow

Input:

20×20×512

After SPPF:

Concatenation → 20×20×2048
After 1×1 Conv → 20×20×512






