# Loan Approval Machine Learning Project #

### Word2Vec and tf-idf ###
- did not have strong results around 50% accuracy
- need to run neural net on more categories
- reused my code for **tf-idf** parsing to input into nn models

### PyTorch - 99.4% Accuracy ###
- optimized initial set up by using...
    - sigmoid for text column
    - including all columns, even when they had a near 0 correlation
    - bundling input and target tensors using TensorDataset and DataLoader
        - then iterating through each during each epoch
    - **99.4% test accuracy**

### TensorFlow - 99.6% Accuracy ###
- dual-input neural network to process both text (TF-IDF) and numeric features
- once again used sigmoid activations in the text branch and relu for numeric branch
- Included a validation_split to monitor overfitting during training
- **99.6% test accuracy** and **ROC AUC of 1.000**
    
### Web App Implementation for Full Stack Experience ###
- built an interactive web app using Streamlit to deploy trained TensorFlow model
- applies full text preprocessing pipeline (cleaning, stopword removal, lemmatization) before vectorizing with TF-IDF
- scales numeric inputs using a pre-fit StandardScaler that was used during training
- displays warnings for high-risk inputs and a disclaimer about real-world usage
- shows prediction result and confidence score upon submission