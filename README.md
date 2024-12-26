# mixture-of-experts
## A simple design for MOE on Mnist data
=======================================
## MIXTURE OF EXPERTS on MNIST DATA

![Mixture of Experts Architecture](MOE-Arch.drawio.png)

</div>
 This is a minimalistic project to understand how MOE architectures work. The training and evaluation is done on MNIST data. 
 Using this code you can have a good look at how things run under the hood.

## Key Features

- **Simple python code to run.**  
  - Just run one file and you can see outputs for
  - `models`: Trained models
  - `plots`: MOE activation outputs on epoch and batch number for a deeper understanding 
  - `csv`: MOE activation activate as a CSV output

## Activation of MOE
![Gattin Network Activation](Gatting-Network-Decision.drawio.png)

## Setup Instructions

1. **Install Anaconda (Recommended):**  
   [Anaconda Installation Guide](https://docs.anaconda.com/)

2. **Create and Activate the Environment:**
   ```
   conda create -n moe_cnn python=3.10.15
   conda activate moe_cnn
   pip install -r requirements.txt
   ```

## Running the Training

- **Small Dataset (e.g., “Alice in Wonderland”):**
    
    ```
    python main.py
    ```

## MIXTURE OF GUIDED XPERTS on MNIST DATA
=======================================

![Mixture of Guided Experts Architecture](MOE-Arch-Guided.drawio.png)

</div>
 Similarly we have a 'guided' of MoE. In this version we use labels to guide the 
 data to handle particular labels. Enforcing expert to only have a expertise 
 towards particular labels.

### Expert Label Assignments:
### ==================================================

### Expert 0: Labels [0, 1]

### Expert 1: Labels [2, 3]

### Expert 2: Labels [4, 5]

### Expert 3: Labels [6, 7]

### Expert 4: Labels [8, 9]

### ==================================================

## Key Features

- **Simple python code to run.**  
  - Just run one file and you can see outputs for
  - `models`: Trained models
  - `plots`: MOE activation outputs on epoch and batch number for a deeper understanding 
  - `csv`: MOE activation activate as a CSV output


