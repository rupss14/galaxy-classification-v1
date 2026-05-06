Galaxy Classification System (Machine Learning Project)

Overview

This project classifies galaxies into Spiral and Elliptical types using Machine Learning.

It is an end-to-end ML project that includes data preprocessing, model comparison, evaluation, and deployment using a Streamlit web application.

Dataset

* Galaxy morphology dataset (Kaggle)
* Contains structural features such as concentration, asymmetry, smoothness, and light distribution

⚠️Note:
Due to file size limitations, the dataset is not included in this repository.
You can download it from Kaggle:
(https://www.kaggle.com/datasets/saurabhshahane/galaxy-classification)

Approach

1. Data Preprocessing

* Converted 'TType' to numeric
* Removed invalid and missing values
* Handled placeholder values ('-9999')
* Created binary classification labels:

  * 0 → Elliptical
  * 1 → Spiral

2. Model Experimentation

As part of my learning process, I experimented with multiple models:

* Logistic Regression
* Decision Tree
* Random Forest

These experiments are documented in experiments.ipynb.

3. Final Model

* Random Forest Classifier
* Selected based on best performance and generalization

Accuracy: 84%

4. Evaluation

The model was evaluated using:

* Accuracy
* Confusion Matrix
* Precision & Recall

The results show balanced performance across both classes.

5. Feature Importance

Feature importance analysis showed that structural properties like:

* H (complexity)
* K (concentration)
* G2 (light distribution)

play a significant role in galaxy classification.

6. Streamlit App

An interactive web application is built using Streamlit where users can:

* Input galaxy features
* Get real-time predictions

# Run Locally:
streamlit run app.py

# Learning & Motivation

This project was built as part of my Machine Learning learning journey.

As a beginner, I focused on:

* Understanding data deeply
* Experimenting with multiple models
* Interpreting results instead of just building models

 # Future Improvements (Version 2)

* Multiclass classification (more galaxy types)
* Image-based classification using CNN (Deep Learning)
* Advanced model tuning and optimization





 # Author
 Rupali Tripathy
