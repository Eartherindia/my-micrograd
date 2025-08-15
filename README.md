# My Micrograd

A Python implementation of Andrej Karpathy’s [micrograd](https://github.com/karpathy/micrograd) — a minimal neural network and autograd engine built from scratch.  
This project follows Karpathy’s YouTube tutorial ["The spelled-out intro to neural networks and backpropagation"](https://youtu.be/VMj-3S1tku0) and serves as my learning exercise in understanding backpropagation, computational graphs, and building models without heavy frameworks.

---

## 📌 Features
- Minimal, clean Python code
- `Value` class for scalar values with automatic differentiation
- Basic operations: addition, multiplication, exponentiation, etc.
- Backpropagation from scratch
- Simple neural network implementation

---

## 📂 Project Structure
my-micrograd/
│
├── micrograd/ # Core engine
│ ├── engine.py # Value class and backprop logic
│ ├── nn.py # Simple neural network components
│
├── examples/ # Example usage scripts
│ └── train.py # Small neural network training example
│
├── README.md # Project documentation
└── requirements.txt # Dependencies (if any)

---

## 🚀 Getting Started

### 1️⃣ Clone the repository
```bash
git clone https://github.com/<your-username>/my-micrograd.git
cd my-micrograd
pip install -r requirements.txt
from micrograd.engine import Value

a = Value(2.0)
b = Value(-3.0)
c = a * b
c.backward()

print(a.grad)  # Gradient of a
print(b.grad)  # Gradient of b
