# Exploring Student Performance Data

This project involves an in-depth exploration and analysis of student performance data. The primary goal is to understand the factors influencing students' GPA and overall academic performance. We utilize various machine learning algorithms and visualization techniques to extract insights from the dataset.

## Table of Contents
- [Dataset](#dataset)
- [Installation](#installation)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Visualization](#visualization)
- [Machine Learning Models](#machine-learning-models)
- [Deep Learning Models](#deep-learning-models)
- [Results](#results)
- [Conclusion](#conclusion)


## Dataset
The dataset used in this project is `student_performance.csv`, which includes the following columns:
- `StudentID`
- `Age`
- `Gender`
- `Ethnicity`
- `ParentalEducation`
- `StudyTimeWeekly`
- `Absences`
- `Tutoring`
- `ParentalSupport`
- `Extracurricular`
- `Sports`
- `Music`
- `Volunteering`
- `GPA`
- `GradeClass` 'Target 5 classes' 

## Installation
To run this project, you'll need to install the following dependencies:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow keras
```

## Exploratory Data Analysis (EDA)

In this step, we perform the following tasks:

- **Data Loading**: Load the dataset and display the first few records.
- **Data Cleaning**: Drop unnecessary columns and check for missing values.
- **Descriptive Statistics**: Summarize the dataset using various statistical measures.

## Visualization
We use `matplotlib` and `seaborn` for visualizing the data:

- **Distribution Plots**: Plot the distribution of each feature.
- **Correlation Heatmap**: Visualize the correlation between different features.
- **Scatter Plots**: Explore the relationship between GPA and various factors such as age, study time, ethnicity, parental education, and absences.

## Machine Learning Models
We implement and evaluate the following machine learning algorithms to predict students' GradeClass:

- **RandomForestClassifier**
- **SVC (Support Vector Classifier)**
- **KNeighborsClassifier**
- **LogisticRegression**
- **DecisionTreeClassifier**
- **GaussianNB (Naive Bayes)**

## Deep Learning Models
We implement a deep learning model using TensorFlow and Keras to predict students' GradeClass. The model architecture includes multiple dense layers with ReLU activation functions and dropout layers to prevent overfitting.Applying Softmax activation function on the output layer and 1 hot encoder for the target variable

## Results
The analysis reveals that absences have a strong negative correlation with GPA. Various models were trained and evaluated based on accuracy, precision, recall, F1 score.

### Maximum Accuracy Achieved
- **Machine Learning (ML)**: The highest accuracy achieved by ML models is 70%.
- **Deep Learning (DL)**: The highest accuracy achieved by the DL model is 75%.

## Conclusion
Absences are a significant factor affecting students' GPA. Future interventions could focus on reducing absences to improve academic performance.


