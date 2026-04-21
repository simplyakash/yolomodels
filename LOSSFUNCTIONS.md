**Loss Function in YOLOv8**
🔹 Total Loss

$L_{total} = \lambda_1 L_{box} + \lambda_2 L_{cls} + \lambda_3 L_{dfl}$

Where:

$L_{box}$ → Bounding box loss
$L_{cls}$ → Classification loss
$L_{dfl}$ → Distribution Focal Loss
$\lambda_1, \lambda_2, \lambda_3$ → Weighting coefficients

🔹 1. Bounding Box Loss ($L_{box}$)

Uses Complete IoU (CIoU) Loss:

$L_{CIoU} = 1 - IoU + \frac{\rho^2(b, b^{gt})}{c^2} + \alpha v$

Where:

$IoU$ → Intersection over Union
$\rho(b, b^{gt})$ → Distance between box centers
$c$ → Diagonal length of smallest enclosing box
$v$ → Aspect ratio consistency
$\alpha$ → Weighting factor

🔹 2. Classification Loss ($L_{cls}$)

Uses Binary Cross Entropy (BCE):

$L_{cls} = - \left[y \log(p) + (1 - y)\log(1 - p)\right]$

Where:

$y$ → Ground truth label
$p$ → Predicted probability
🔹 3. Distribution Focal Loss ($L_{dfl}$)

$L_{DFL} = - \sum y_i \log(p_i)$

Where:

$y_i$ → Target distribution
$p_i$ → Predicted probabilities
📊 Summary
Loss Component	Type	Purpose
$L_{box}$	CIoU Loss	Bounding box accuracy
$L_{cls}$	Binary Cross Entropy	Classification accuracy
$L_{dfl}$	Distribution Focal Loss	Precise localization





### Loss Functions

**Binary Cross Entropy:**

$L_{BCE} = -[y \log(p) + (1 - y)\log(1 - p)]$

**Dice Loss:**

$L_{dice} = 1 - \frac{2|P \cap G|}{|P| + |G|}$
