# Loan Approval Machine Learning Project #

**IN PROGRESS**


### Word2Vec and tf-idf ###
- did not have strong results around 50% accuracy
- need to run neural net on more categories
- reused my code for **tf-idf** parsing to input into nn models

### PyTorch - 99% Accuracy ###
- optimized initial set up by using...
    - sigmoid for text column
    - including all columns, even when they had a near 0 correlation
    - bundling input and target tensors using TensorDataset and DataLoader
        - then iterating through each during each epoch
    - **99% test accuracy**

### TensorFlow ###
- dual-input neural network to process both text (TF-IDF) and numeric features
- once again used sigmoid activations in the text branch and relu for numeric branch
- Included a validation_split to monitor overfitting during training
- **99.3% test accuracy** and **ROC AUC of 1.000**
    





***Stay Tuned!!!***