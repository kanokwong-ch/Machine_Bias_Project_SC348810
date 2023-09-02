# Hepatitis C Virus from UCI Machine Learning

This dataset is tabular data with 14 features. The dataset contains 615 cases from patients that can classify which patients are disease or not.

| VARIABLE  |  DESCRIPTION  | DATA TYPE | EXAMPLE |
|:----------|---------------|-----------|---------|
| Patient ID| Index ID to patient identification | NUMBER | 1 |
| Category  | Category of diagnosis (0=Blood Donor, 0s=suspect Blood Donor, 1=Hepatitis, 2=Fibrosis, 3=Cirrhosis) | STRING | 0=Blood Donor |
| Age       | Number of age | NUMBER | 32 |
| Sex       | Sex (f = Female, m = Male) | STRING | m |
| ALB       | An albumin blood test checks your liver and kidney function. Albumin is protein in your blood plasma. Normal Range: 3.5 - 5.5 (grams(g)/deciliter(dL) | NUMBER | 38.5 |
| ALP       | Alkaline phosphatase test measures the amount of ALP in your blood.To check your liver or bone abnormalities.Normal Range: 44 - 147 (international units(IU)/liter(L) | NUMBER | 52.5 |
| ALT       | Alanine transaminase is an enzyme found mostly in the liver. An ALT test measures the amount of ALT in the blood. High levels of ALT in your blood may be a sign of a liver injury or disease.Normal Range: 7 - 56 (units(U)/liter(L) | NUMBER | 7.7 |
| AST       | Aspartate aminotransferase is an enzyme found mostly in the liver.but it's also in muscles and other organs in your body. An AST test measures the amount of AST in the blood.The test is commonly used to help diagnose liver damage or disease. Normal Range: Men 14 - 20 (units(U)/liter(L)), Women 10 - 36 (units(U)/liter(L)| NUMBER | 22.1 |
| BIL       |Bilirubin test measures the levels of bilirubin in your blood.Higher than usual levels of bilirubin may indicate different types of liver or bile duct problems. Sometimes, higher bilirubin levels may be caused by an increased rate of destruction of red blood cells.Normal Range: 0.1 - 1.2 (miligram(mg)/deciliter(dL) | NUMBER | 7.5 |
| CHE       | Cholinesterase is a blood test that looks at levels of 2 substances that help the nervous system work properly. They are called Acetylcholinesterase is found in nerve tissue and red blood cells. Pseudocholinesterase is found primarily in the liver.The test is commonly used to help diagnose liver damage or disease.Normal Range: 8 - 18 (units(U)/milliliter(mL)) | NUMBER | 6.93 |
| CHOL      | Cholesterol (or lipid profile) blood test looks at the levels of cholesterol and other fats in your blood.The test is assess your risk of heart disease and stroke. Normal Range: <= 200 (milligrams(mg)/deciliter(dL)) | NUMBER | 3.23 |
| CREA      | Creatinine test is a measure of how well your kidneys are performing their job of filtering waste from your blood. Normal Range: 0.7 - 1.3 (milligrams(mg)/deciliter(dL)) | NUMBER | 106 |
| GGT       | Gamma-glutamyl transferase (GGT) test measures the amount of GGT in the blood. GGT is an enzyme found throughout the body, but it is mostly found in the liver.High levels of GGT in the blood may be a sign of liver disease or damage to the bile ducts.Normal Range: 5 - 40 (units(U)/liter(L)) | NUMBER | 12.1 |
| PROT      | Total protein test measures your albumin and globulin levels. High levels may mean you have inflammation or infection. Low levels may suggest kidney problems, malnutrition, and several other possible conditions. Normal Range: 6 - 8.3 (gram(g)/deciliter(dL)) | NUMBER | 69 |



# Quality issues in the dataset
- Missing data issue => Some columns in this dataset have missing values and they need to clean up.
- Inconsistent formatting issue => Because this dataset has categorical data and label column is not in the right format to feed machine learning.
- Skewness issue => In laboratory features, most of them are skewed, need to take care about this point.
- Unbalanced data issue => In target class, mostly is Blood Donor when compared to other disease class. 


# Define the goals and a suitable measure for the quality issues. 

For this dataset we have 4 goals to achieve. 
1.	Handle missing value in the dataset. 
In this dataset have 5 columns that contain missing values such as ALB, ALP, ALT, CHOL and PROT.
- For handle missing values, I filled it by using mean value from each class not whole dataset because it has different range from each class. The plot shows the result after filling missing value.

2.	Do the right way to encode the dataset. 
The dataset has categorical data, and the target class needs to encode it.

 Sex column : 
|  Old Value  | New Value |       
|-------------|-----------|
| male | 0 |
| female | 1 |

Category(Target) column : 
|  Old Value  | New Value |       
|-------------|-----------|
| 0=Blood Donor | 0 |
| 0s=suspect Blood Donor | 0 |
| 1=Hepatitis | 1 |
| 2=Fibrosis | 2 |
| 3=Cirrhosis | 3 |

3.	Handle skewness in the dataset. 
- Skewed data can introduce bias into the model prediction because most machine learning assumes that data is roughly normally distributed, with an equal number of data points on both sides of the mean and it also affects feature engineering too. 
- For handle this problem, I used Box-cox transform to make the data close to normal distribution shape in every features. The sample graph shows the result of box-cox transform that apply with AST feature.  

4.	Handle imbalanced data in the dataset. 
- I split the dataset into train and test set then I applied SMOTE-ENN to train data for handle imbalanced data.

# Explore different kinds of machine learning models developed with different modeling techniques. Then, choose the machine learning techniques, implement the models using scikit-learn, and train the models. 

In this experiment we used 9 different classification models, namely K-NN, Logistic Regression, Multi-Layer Perceptron, Support Vector Machine, Gaussian Naïve Bayes, Random Forest, LightGBM, XGBoost and Voting Classifier.  

# Model performance results

|  Model  | Accuracy | Recall | Precision | F1-Score |      
|---------|----------|--------|-----------|----------|
| K-NN  | 65.0407 | 65.0407 | 82.5551 | 71.6134 |
| SVM  | 70.2703 | 70.2703 | 84.6486 | 74.7475 |
| Gaussian Naïve Bayes | 70.7317 | 70.7317 | 89.2287 | 77.3133 |
| Logistic Regression | 86.9919 | 86.9919 | 75.6759 | 80.9403 |
| MLP | 88.1081 | 88.1081 | 77.6304 | 82.5381 |
| Random Forest | 95.1351 | 95.1351 | 96.3449 | 95.5191 |
| XGBoost | 96.2162 | 96.2162 | 97.5646 | 96.5391 |
| LightGBM  | 97.2973 | 97.2973  | 98.3368 | 97.5980 |
| Voting Classifier | 97.2973  | 97.2973 | 98.93368 | 97.5980 |

From the results, an ensemble model is a great choice for imbalanced data problem when we need to select kinds of the model.  

