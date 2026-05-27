# 🧠 Difference Between Semantic, Instance, and Panoptic Segmentation

| Type | Meaning | Output | Example |
|---|---|---|---|
| Semantic Segmentation | Classifies every pixel into a category | Same class objects share same label | All cars marked as one class |
| Instance Segmentation | Separates individual objects of same class | Each object gets separate mask | Each car gets different mask |
| Panoptic Segmentation | Combines semantic + instance segmentation | Background semantic + object instances | Road segmented + individual cars separated |

---

# 🥇 1️⃣ Semantic Segmentation

Semantic segmentation answers:

```text
“What class does each pixel belong to?”
```

It does NOT distinguish between:
- multiple objects of same class

---

# 📦 Example

Suppose image contains:
- 5 cars

Output:

```text
all car pixels → class “car”
```

All cars get:
- same label/color

---

# 🏗️ Output Example

```text
Road → Gray
Car → Blue
Tree → Green
Sky → Cyan
```

But:
- all cars merged together

---

# 📌 Real-World Example

- road scene understanding
- satellite segmentation
- organ segmentation

---

# 🥈 2️⃣ Instance Segmentation

Instance segmentation answers:

```text
“What class is this AND which object instance is it?”
```

It separates:
- individual objects

even if same class.

---

# 📦 Example

Image contains:
- 5 cars

Output:

```text
Car 1 → Mask A
Car 2 → Mask B
Car 3 → Mask C
```

Each object gets:
- separate mask

---

# 🏗️ Output Example

```text
Car 1 → Red
Car 
