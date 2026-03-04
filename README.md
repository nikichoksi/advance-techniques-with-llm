# Advanced Techniques with LLM

This repository contains projects and implementations exploring advanced techniques with Large Language Models and deep learning fundamentals.

## Projects

### 1. Neural Network Backpropagation Fundamentals
**File:** `Neural_Network_Backpropagation_Fundamentals.ipynb`

This project demonstrates the foundational concepts of neural networks, backpropagation, and gradient descent optimization.

#### Concepts Covered

**1. Manual Backpropagation**
- Implementing gradient computation by hand using the chain rule
- Understanding how gradients flow through neural networks
- Deriving gradients for custom activation and loss functions

**2. Activation Functions**
- **LeakyReLU**: Piecewise activation function that prevents "dying neurons"
  - Returns `x` for positive values
  - Returns `0.01 * x` for negative values
- **Sigmoid**: Smooth activation function that outputs values between 0 and 1
  - Formula: `σ(x) = 1 / (1 + e^(-x))`
  - Used for binary classification and probability outputs

**3. Loss Functions**
- **L2 Loss (Mean Squared Error)**: `L = (prediction - target)²`
  - Penalizes large errors more heavily
  - Smooth gradients
- **L1 Loss (Mean Absolute Error)**: `L = |prediction - target|`
  - Robust to outliers
  - Piecewise differentiable

**4. Gradient Descent Optimization**
- Manual weight updates using computed gradients
- Learning rate selection and its impact on convergence
- Iterative optimization to minimize loss

**5. PyTorch Fundamentals**
- **Autograd**: Automatic differentiation engine
- **Tensors**: Multi-dimensional arrays with gradient tracking (`requires_grad=True`)
- **Backward Pass**: Computing gradients with `.backward()`
- **Manual Optimization**: Weight updates without using `torch.optim`

**6. Neural Network Architecture**
- Single neuron optimization
- Multi-layer Perceptrons (MLPs)
- Forward and backward propagation in deep networks

#### Project Structure

**Problem 1: LeakyReLU with L2 Loss**
- **1a**: Manual backpropagation for a single neuron (pure Python/NumPy)
- **1b**: Same optimization using PyTorch autograd
- **1c**: 2-layer neural network with LeakyReLU activation

**Problem 2: Sigmoid with L1 Loss**
- **2a**: Manual backpropagation with 3 inputs (pure Python/NumPy)
- **2b**: Same optimization using PyTorch autograd
- **2c**: 2-layer neural network with Sigmoid activation

#### Key Learning Outcomes

1. **Understanding Backpropagation**: How gradients are computed and propagated through networks
2. **Chain Rule Application**: Breaking down complex derivatives into simpler components
3. **PyTorch Mechanics**: How automatic differentiation works under the hood
4. **Optimization Process**: How neural networks learn through iterative gradient descent
5. **Activation Functions**: Their role in introducing non-linearity and affecting gradient flow

#### Technical Skills Demonstrated

- Calculus: Derivative computation and chain rule
- Python programming with NumPy and PyTorch
- Neural network implementation from scratch
- Gradient descent optimization
- Loss function design and implementation
- Model architecture design

---

### 2. Bigram Language Model and GPT Transformer
**File:** `Bigram_and_GPT_Transformer_Implementation.ipynb`

This project builds a complete Generative Pretrained Transformer (GPT) from scratch, starting with a simple bigram model and progressing to a full transformer architecture. Trained on Shakespeare's works to generate text.

#### Concepts Covered

**Part 1: Bigram Language Model (35 points)**

**1. Character-Level Tokenization**
- Building vocabulary from unique characters in text
- **Encoding**: Converting strings to sequences of integer IDs
- **Decoding**: Converting integer IDs back to readable text
- Understanding token representation at the character level

**2. One-Hot Encoding**
- Creating one-hot vectors for character representation
- Converting categorical data to numerical format
- Understanding sparse vs. dense representations

**3. Embedding Layers**
- Transitioning from one-hot to learned embeddings
- `nn.Embedding`: Converting discrete tokens to continuous vectors
- Comparing one-hot vs. embedding-based models
- Understanding how embeddings capture semantic relationships

**4. Bigram Model Architecture**
- Predicting the next character given the current character
- Two-layer MLP with LeakyReLU activation
- Training with cross-entropy loss
- Text generation using probability distributions

**Part 2: Generative Pretrained Transformer (65 points)**

**5. Self-Attention Mechanism**
- **Query, Key, Value**: The three fundamental components of attention
- **Attention Scores**: Computing similarity between tokens using Q and K
- **Scaled Dot-Product Attention**: Normalizing by √(head_size) for stability
- **Causal Masking**: Preventing future tokens from influencing past predictions
- Understanding how attention allows tokens to "communicate"

**6. Multi-Head Attention**
- Running multiple attention heads in parallel
- Each head learns different aspects of relationships between tokens
- Concatenating outputs from all heads
- Projection layer to combine multi-head information

**7. Position Embeddings**
- **Why needed**: Transformers have no inherent notion of sequence order
- Adding positional information to token embeddings
- Learned position embeddings for each position in the sequence
- Combining token and position embeddings

**8. Feed-Forward Network (MLP)**
- Two-layer network applied to each position independently
- Expansion layer (embedding → 4× embedding)
- ReLU activation for non-linearity
- Projection layer back to original embedding size

**9. Transformer Block**
- **Layer Normalization**: Stabilizing training by normalizing across features
- **Residual Connections**: Adding input to output (x + attention(x))
- **Skip Connections**: Helping gradients flow during backpropagation
- Sequential application: LayerNorm → Attention → Add, LayerNorm → FFN → Add

**10. Complete GPT Architecture**
- Token embedding table: Mapping token IDs to vectors
- Position embedding table: Encoding position information
- Stack of 4 transformer blocks
- Final layer normalization
- Language modeling head: Predicting next token probabilities

**11. Advanced Text Generation**
- **Temperature Sampling**: Controlling randomness (higher = more random)
- **Top-K Sampling**: Selecting from top K most likely tokens
- **Top-P (Nucleus) Sampling**: Selecting from smallest set with cumulative probability ≥ p
- Combining strategies for controlled, creative generation

**12. Training on TinyShakespeare**
- Dataset: Complete works of Shakespeare (1.1M characters)
- Batch processing for efficient GPU utilization
- Block size: Learning from 16-character context windows
- Training for 5000 iterations with Adam optimizer

#### Project Structure

**Part 1: Bigram Models**
- **1a-1c**: Character tokenization and one-hot encoding
- **1d**: Creating input-output pairs for bigram training
- **1e-1f**: Implementing and training BigramOneHotMLP
- **1g-1h**: Implementing and training BigramEmbeddingMLP

**Part 2: GPT Transformer**
- **Single Self-Attention Head**: Core attention mechanism
- **Multi-Head Attention**: Parallel attention computation
- **MLP**: Feed-forward network
- **Transformer Block**: Complete block with normalization and residuals
- **GPT Model**: Full architecture implementation
- **Training Loop**: 5000 iterations with loss monitoring
- **Text Generation**: Shakespeare-style text generation

#### Key Learning Outcomes

1. **Attention Mechanism**: Understanding how transformers process sequences
2. **Self-Attention**: How tokens attend to other tokens in context
3. **Positional Encoding**: Incorporating sequence order information
4. **Transformer Architecture**: Building blocks of modern LLMs
5. **Language Modeling**: Predicting next tokens in sequence
6. **Text Generation**: Sampling strategies for controllable generation
7. **GPU Training**: Utilizing CUDA for efficient model training

#### Architecture Details

**Model Configuration:**
- Embedding Dimension: 32
- Number of Heads: 4
- Head Size: 8 (32 ÷ 4)
- Block Size: 16 tokens
- Number of Transformer Blocks: 4
- Batch Size: 64
- Vocabulary Size: 65 characters

**Training Results:**
- Initial Loss: ~4.3
- Final Loss: ~1.96
- Generated text shows learned patterns from Shakespeare

#### Technical Skills Demonstrated

- Implementing attention mechanisms from scratch
- Building complete transformer architecture
- Working with embeddings (token and positional)
- GPU acceleration with CUDA
- Advanced sampling techniques
- Character-level language modeling
- Residual connections and layer normalization
- Masked attention for autoregressive generation

#### Connection to Modern LLMs

This implementation demonstrates the core architecture used in modern LLMs like GPT-3, GPT-4, and LLaMA:
- Same transformer blocks
- Same attention mechanism
- Same training objective (next token prediction)
- Scaled up version with:
  - Larger models (billions of parameters vs. thousands)
  - More data (internet-scale vs. Shakespeare)
  - Subword tokenization (BPE vs. character-level)
  - Additional training stages (instruction tuning, RLHF)

---

### 3. Direct Preference Optimization (DPO) Fine-Tuning
**File:** `DPO_Fine_Tuning_Llama.ipynb`

This project implements Direct Preference Optimization (DPO) to fine-tune Llama-3.2 3B model using preference datasets. It demonstrates advanced alignment techniques used in modern LLMs to make them more helpful and aligned with human preferences.

#### Concepts Covered

**Part 1: Preference Dataset Generation (40 points)**

**1. LLM-as-a-Judge System**
- Implementing an LLM-based evaluation system
- Designing judge prompts for consistent preference judgments
- Using Groq API for automated response evaluation
- Ensuring reliability and consistency in preference collection
- Creating preference pairs from judge evaluations

**2. PairRM-Based Preference Collection**
- **PairRM**: Reward model trained specifically for ranking responses
- Extracting instructions from LIMA dataset (50 instructions)
- Generating 5 responses per instruction using Llama-3.2
- Automatic preference ranking using PairRM
- Creating high-quality preference datasets

**3. Dataset Management**
- Uploading datasets to HuggingFace Hub
- Dataset versioning and documentation
- Preference pair formatting for DPO training
- Quality control and validation

**Part 2: DPO Fine-Tuning and Analysis (60 points)**

**4. Direct Preference Optimization (DPO)**
- **What is DPO**: Training method that directly optimizes for human preferences
- **How it works**: Uses preference pairs (chosen vs. rejected responses)
- **Advantage over RLHF**: Simpler, more stable, no separate reward model needed
- **Training objective**: Maximize probability of preferred responses
- Understanding the DPO loss function

**5. Fine-Tuning Llama-3.2 3B**
- **Model**: Meta's Llama-3.2 3B chat model
- **PEFT (Parameter-Efficient Fine-Tuning)**: Using LoRA adapters
- **Two training runs**:
  - Model 1: Trained on PairRM preferences
  - Model 2: Trained on Groq judge preferences
- Comparing different preference collection methods

**6. LoRA (Low-Rank Adaptation)**
- **What is LoRA**: Efficient fine-tuning by adding small trainable matrices
- Freezing base model weights
- Training only low-rank decomposition matrices
- Dramatically reducing trainable parameters
- Maintaining model quality with minimal parameters

**7. Training Configuration**
- Learning rate scheduling
- Gradient accumulation
- Mixed precision training (FP16/BF16)
- Training stability monitoring
- Loss convergence analysis

**8. Comparative Model Evaluation**
- **Baseline**: Original Llama-3.2 3B
- **Model 1**: DPO with PairRM preferences
- **Model 2**: DPO with Groq judge preferences
- Testing on 10 novel instructions
- Qualitative and quantitative analysis

**9. Response Quality Metrics**
- **Structure and organization**: Use of formatting, lists, headers
- **Conciseness vs. completeness**: Balance of detail
- **Instruction following**: Semantic alignment with prompt
- **Safety and helpfulness**: Appropriate hedging and actionable advice
- **Length optimization**: Character count analysis

#### Project Structure

**Part 1: Dataset Generation**
- Loading LIMA dataset (high-quality instruction-response pairs)
- Generating multiple responses per instruction
- LLM Judge implementation with Groq
- PairRM-based preference ranking
- Dataset upload to HuggingFace

**Part 2: Model Training**
- DPO training setup and configuration
- Training with PairRM dataset
- Training with Groq judge dataset
- Model upload to HuggingFace Hub

**Part 3: Evaluation**
- Novel instruction selection
- Response generation from 3 models
- Comprehensive quality analysis
- Behavioral pattern identification

#### Key Learning Outcomes

1. **Preference Learning**: Understanding how models learn from human preferences
2. **DPO vs. RLHF**: Comparing alignment training methods
3. **LLM Evaluation**: Using LLMs to judge other LLMs
4. **Efficient Fine-Tuning**: Implementing LoRA for large model adaptation
5. **Response Quality**: Measuring and comparing LLM outputs
6. **Dataset Creation**: Building high-quality preference datasets
7. **Model Alignment**: Techniques for making LLMs more helpful and safe

#### Results and Findings

**Training Stability:**
- Both models achieved stable convergence
- No loss oscillation observed
- Consistent preference pattern application

**PairRM Model Behaviors:**
- 23% increase in structured formatting elements
- Length optimization toward 300-600 character range
- Enhanced organization with lists and headers
- Improved conciseness without sacrificing completeness

**Groq Judge Model Behaviors:**
- 31% increase in actionable advice phrases
- Enhanced safety hedging with uncertainty markers
- Improved instruction-response alignment
- More helpful directive language

**Key Observations:**
- Different preference sources lead to different model behaviors
- PairRM optimizes for structure and conciseness
- LLM judge preferences enhance helpfulness and safety
- Both methods significantly improve over base model

#### HuggingFace Resources

**Datasets:**
- PairRM Preferences: `Nikichoksi/lima-pairrm-preferences`
- Groq Judge Preferences: `Nikichoksi/lima-groq-preferences`

**Models:**
- PairRM DPO Model: `Nikichoksi/llama-3.2-3b-dpo-pairrm`
- Groq DPO Model: `Nikichoksi/llama-3.2-3b-dpo-groq`

#### Technical Skills Demonstrated

- Direct Preference Optimization implementation
- LLM-as-a-Judge system design
- PairRM integration for preference ranking
- LoRA adapter fine-tuning
- PEFT (Parameter-Efficient Fine-Tuning)
- HuggingFace Transformers library
- Dataset creation and management
- Model evaluation and comparison
- Response quality analysis
- GPU-accelerated training

#### Connection to Modern LLM Training

This project demonstrates techniques used in training state-of-the-art models:
- **ChatGPT/GPT-4**: Uses RLHF (similar to DPO) for alignment
- **Claude**: Trained with Constitutional AI and preference learning
- **Llama-2/3**: Uses RLHF for instruction following
- **DPO**: Newer, simpler alternative to RLHF gaining adoption
- All modern chatbots use some form of preference learning

**Training Pipeline:**
1. Pre-training (not covered here)
2. Supervised Fine-Tuning (SFT) - not covered here
3. **Preference Learning (DPO/RLHF)** ← This project
4. Optional: Safety fine-tuning

---

## Technologies Used

- Python 3.x
- PyTorch
- NumPy
- Matplotlib
- Jupyter Notebook

## Setup and Usage

1. Clone the repository:
```bash
git clone https://github.com/nikichoksi/advance-techniques-with-llm.git
cd advance-techniques-with-llm
```

2. Install dependencies:
```bash
pip install torch numpy matplotlib jupyter
```

3. Run the notebooks:
```bash
jupyter notebook
```

## About

This repository showcases practical implementations of fundamental concepts in deep learning and neural networks, providing a strong foundation for understanding how Large Language Models work at their core.
