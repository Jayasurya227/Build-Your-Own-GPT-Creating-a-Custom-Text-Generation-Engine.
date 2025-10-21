# Build Your Own GPT: Creating a Custom Text Generation Engine 📝

This project demonstrates the fundamental concepts behind building a Generative Pre-trained Transformer (GPT) model by creating a character-level text generation engine from scratch using PyTorch. The model is trained on the complete works of William Shakespeare to generate Shakespearean-style text.

This notebook serves as an educational tool, walking through the process of data preparation, building the core components of a transformer (like self-attention and feed-forward networks), training the language model, and generating new text.

**Dataset:** `shakespeare.txt` (Plain text file containing Shakespeare's works)
**Focus:** Character-level language modeling, Transformer architecture fundamentals (Self-Attention, Multi-Head Attention, Positional Encoding), model building with PyTorch, training loop implementation, text generation.
**Repository:** [https://github.com/Jayasurya227/Build-Your-Own-GPT-Creating-a-Custom-Text-Generation-Engine](https://github.com/Jayasurya227/Build-Your-Own-GPT-Creating-a-Custom-Text-Generation-Engine)

***

## Key Techniques & Concepts Demonstrated

Based on the implementation within the notebook (`14_Build_Your_Own_GPT_Creating_a_Custom_Text_Generation_Engine (1).ipynb`), the following key concepts are covered:

* **Character-Level Language Modeling:** Predicting the next character in a sequence based on the preceding characters.
* **Data Preprocessing for Text:**
    * Creating a vocabulary of unique characters.
    * Encoding/Decoding text to/from integer representations.
    * Splitting data into training and validation sets.
* **Batch Generation:** Creating input (`x`) and target (`y`) batches for training, where the target sequence is shifted by one position.
* **Token & Positional Embeddings:** Representing characters as vectors and adding positional information.
* **Transformer Architecture Components (Built from Scratch):**
    * **Self-Attention (`Head`):** Implementing the scaled dot-product attention mechanism with masking (causal attention) to allow characters to attend only to previous characters in the sequence.
    * **Multi-Head Attention (`MultiHeadAttention`):** Combining multiple self-attention heads running in parallel to capture different representational subspaces.
    * **Feed-Forward Network (`FeedForward`):** Applying a simple multi-layer perceptron (MLP) to each position independently.
    * **Transformer Block (`Block`):** Combining Multi-Head Attention and Feed-Forward networks with residual connections and Layer Normalization.
* **Model Building (`GPTLanguageModel`):** Stacking multiple Transformer Blocks to create the full language model.
* **Training Loop Implementation (PyTorch):**
    * Using the AdamW optimizer.
    * Calculating Cross-Entropy Loss.
    * Performing forward pass, backward pass (backpropagation), and optimizer steps.
    * Implementing a function (`estimate_loss`) to evaluate model performance on training and validation sets periodically without gradient calculation.
* **Text Generation:** Implementing a `generate` function to sample new text character by character from the trained model, given an initial context.
* **Hyperparameter Tuning:** Defining and using various hyperparameters like `batch_size`, `block_size` (context length), `learning_rate`, `max_iters`, `eval_interval`, `n_embd` (embedding dimension), `n_head`, `n_layer`, `dropout`.

***

## Analysis Workflow

The notebook follows a step-by-step process to build and train the character-level GPT model:

1.  **Setup & Data Loading:** Importing libraries (torch, torch.nn, matplotlib) and loading the `shakespeare.txt` dataset.
2.  **Data Preparation:**
    * Extracting unique characters to build the vocabulary.
    * Creating character-to-integer (`stoi`) and integer-to-character (`itos`) mappings.
    * Encoding the entire dataset using the mappings.
    * Splitting the encoded data into training and validation tensors.
3.  **Batching Logic:** Defining the `get_batch` function to efficiently sample random sequences from the data for training.
4.  **Model Definition (Iterative):**
    * Starting with a basic `BigramLanguageModel` as a baseline.
    * Implementing the core components: `Head` (Self-Attention), `MultiHeadAttention`, `FeedForward`, `Block`.
    * Combining these components into the final `GPTLanguageModel` incorporating token and positional embeddings, multiple transformer blocks, and a final linear layer.
5.  **Training Setup:**
    * Defining hyperparameters.
    * Instantiating the model and moving it to the appropriate device (GPU if available).
    * Initializing the AdamW optimizer.
    * Defining the `estimate_loss` helper function.
6.  **Training Loop:** Iteratively training the model by:
    * Sampling batches using `get_batch`.
    * Performing forward and backward passes.
    * Updating model weights using the optimizer.
    * Evaluating and printing training/validation loss at specified intervals.
7.  **Text Generation:**
    * Defining the `generate` method within the model class (and as a standalone function).
    * Using the trained model (`model.eval()`) to generate new text sequences based on an initial context (e.g., a newline character).
    * Decoding the generated integer sequences back into readable text.

***

## Technologies Used

* **Python**
* **PyTorch:** Core deep learning framework for building tensors, defining neural network modules (`nn.Module`), automatic differentiation, optimization, and training.
* **NumPy:** (Implicitly used) For numerical operations.
* **Matplotlib:** For plotting (e.g., used for visualizing loss curves if added).
* **Jupyter Notebook / Google Colab:** For the interactive development environment.

***

## How to Run the Project

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Jayasurya227/Build-Your-Own-GPT-Creating-a-Custom-Text-Generation-Engine.git](https://github.com/Jayasurya227/Build-Your-Own-GPT-Creating-a-Custom-Text-Generation-Engine.git)
    cd Build-Your-Own-GPT-Creating-a-Custom-Text-Generation-Engine
    ```
2.  **Install dependencies:**
    (It is recommended to use a virtual environment)
    ```bash
    pip install torch torchvision torchaudio matplotlib jupyter numpy 
    # Or install PyTorch following official instructions: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)
    ```
3.  **Ensure Dataset:** Make sure the `shakespeare.txt` file is present in the repository directory. (If not included, you may need to add it or modify the code to download it, e.g., from Andrej Karpathy's char-rnn repository).
4.  **Launch Jupyter Notebook:**
    ```bash
    jupyter notebook "14_Build_Your_Own_GPT_Creating_a_Custom_Text_Generation_Engine (1).ipynb"
    ```
5.  **Run Cells:** Execute the cells sequentially. The training process may take some time depending on your hardware (CPU vs. GPU) and the specified `max_iters`. The final cells will output generated text.

***

## Author & Portfolio Use

* **Author:** Jayasurya227
* **Portfolio:** This project ([https://github.com/Jayasurya227/Build-Your-Own-GPT-Creating-a-Custom-Text-Generation-Engine](https://github.com/Jayasurya227/Build-Your-Own-GPT-Creating-a-Custom-Text-Generation-Engine)) is an excellent educational piece demonstrating a deep understanding of the fundamental building blocks of modern language models like GPT. It showcases proficiency in PyTorch and the core concepts of the Transformer architecture. Highly suitable for GitHub, resumes/CVs, LinkedIn, and technical interviews for AI/ML Engineer or Deep Learning Researcher roles.
* **Notes:** Recruiters can observe the from-scratch implementation of key components (attention, blocks), the training logic, and the text generation capability, indicating strong foundational knowledge in NLP and deep learning. This project is inspired by and follows the structure of Andrej Karpathy's "makemore" and "nanoGPT" educational series.
