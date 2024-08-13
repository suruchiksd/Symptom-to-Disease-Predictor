import tkinter as tk
from tkinter import *
from tkinter import ttk
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# List of symptoms and diseases
l1 = [
    'back_pain', 'constipation', 'abdominal_pain', 'diarrhoea', 'mild_fever', 'yellow_urine',
    'yellowing_of_eyes', 'acute_liver_failure', 'fluid_overload', 'swelling_of_stomach',
    'swelled_lymph_nodes', 'malaise', 'blurred_and_distorted_vision', 'phlegm', 'throat_irritation',
    'redness_of_eyes', 'sinus_pressure', 'runny_nose', 'congestion', 'chest_pain', 'weakness_in_limbs',
    'fast_heart_rate', 'pain_during_bowel_movements', 'pain_in_anal_region', 'bloody_stool',
    'irritation_in_anus', 'neck_pain', 'dizziness', 'cramps', 'bruising', 'obesity', 'swollen_legs',
    'swollen_blood_vessels', 'puffy_face_and_eyes', 'enlarged_thyroid', 'brittle_nails',
    'swollen_extremeties', 'excessive_hunger', 'extra_marital_contacts', 'drying_and_tingling_lips',
    'slurred_speech', 'knee_pain', 'hip_joint_pain', 'muscle_weakness', 'stiff_neck', 'swelling_joints',
    'movement_stiffness', 'spinning_movements', 'loss_of_balance', 'unsteadiness',
    'weakness_of_one_body_side', 'loss_of_smell', 'bladder_discomfort', 'foul_smell_of_urine',
    'continuous_feel_of_urine', 'passage_of_gases', 'internal_itching', 'toxic_look_(typhos)',
    'depression', 'irritability', 'muscle_pain', 'altered_sensorium', 'red_spots_over_body', 'belly_pain',
    'abnormal_menstruation', 'dischromic_patches', 'watering_from_eyes', 'increased_appetite', 'polyuria',
    'family_history', 'mucoid_sputum', 'rusty_sputum', 'lack_of_concentration', 'visual_disturbances',
    'receiving_blood_transfusion', 'receiving_unsterile_injections', 'coma', 'stomach_bleeding',
    'distention_of_abdomen', 'history_of_alcohol_consumption', 'blood_in_sputum', 'prominent_veins_on_calf',
    'palpitations', 'painful_walking', 'pus_filled_pimples', 'blackheads', 'scurring', 'skin_peeling',
    'silver_like_dusting', 'small_dents_in_nails', 'inflammatory_nails', 'blister', 'red_sore_around_nose',
    'yellow_crust_ooze'
]

disease = [
    'Fungal infection', 'Allergy', 'GERD', 'Chronic cholestasis', 'Drug Reaction',
    'Peptic ulcer diseae', 'AIDS', 'Diabetes', 'Gastroenteritis', 'Bronchial Asthma', 'Hypertension',
    'Migraine', 'Cervical spondylosis', 'Paralysis (brain hemorrhage)', 'Jaundice', 'Malaria', 'Chicken pox',
    'Dengue', 'Typhoid', 'hepatitis A', 'Hepatitis B', 'Hepatitis C', 'Hepatitis D', 'Hepatitis E',
    'Alcoholic hepatitis', 'Tuberculosis', 'Common Cold', 'Pneumonia', 'Dimorphic hemmorhoids(piles)',
    'Heart attack', 'Varicose veins', 'Hypothyroidism', 'Hyperthyroidism', 'Hypoglycemia', 'Osteoarthristis',
    'Arthritis', '(vertigo) Paroymsal  Positional Vertigo', 'Acne', 'Urinary tract infection', 'Psoriasis',
    'Impetigo'
]

# Read and preprocess data
df = pd.read_csv("Prototype.csv")
print("Columns in Prototype.csv:", df.columns.tolist())

# Check and handle missing columns
missing_cols = [col for col in l1 if col not in df.columns]
if missing_cols:
    print(f"Missing columns in Prototype.csv: {missing_cols}")

# Filter out missing columns
existing_cols = [col for col in l1 if col in df.columns]
X = df[existing_cols]
y = df[['prognosis']].values.ravel()

tr = pd.read_csv("Prototype 1.csv")
print("Columns in Prototype 1.csv:", tr.columns.tolist())

# Check and handle missing columns in testing data
missing_cols_test = [col for col in l1 if col not in tr.columns]
if missing_cols_test:
    print(f"Missing columns in Prototype 1.csv: {missing_cols_test}")

# Filter out missing columns
existing_cols_test = [col for col in l1 if col in tr.columns]
X_test = tr[existing_cols_test]
y_test = tr[['prognosis']].values.ravel()

# Initialize symptom list with zeros
l2 = [0] * len(l1)

# Define functions for machine learning models
def DecisionTree():
    clf = tree.DecisionTreeClassifier()
    clf.fit(X, y)
    
    y_pred = clf.predict(X_test)
    print("DecisionTree Accuracy (Normalized):", accuracy_score(y_test, y_pred))
    print("DecisionTree Accuracy (Raw):", accuracy_score(y_test, y_pred, normalize=False))
    
    predict_and_display(clf)

def randomforest():
    clf = RandomForestClassifier()
    clf.fit(X, y)
    
    y_pred = clf.predict(X_test)
    print("RandomForest Accuracy (Normalized):", accuracy_score(y_test, y_pred))
    print("RandomForest Accuracy (Raw):", accuracy_score(y_test, y_pred, normalize=False))
    
    predict_and_display(clf)

def NaiveBayes():
    clf = GaussianNB()
    clf.fit(X, y)
    
    y_pred = clf.predict(X_test)
    print("NaiveBayes Accuracy (Normalized):", accuracy_score(y_test, y_pred))
    print("NaiveBayes Accuracy (Raw):", accuracy_score(y_test, y_pred, normalize=False))
    
    predict_and_display(clf)

def predict_and_display(clf):
    # Retrieve selected symptoms from the GUI
    psymptoms = [symptom_vars[0].get(), symptom_vars[1].get(), symptom_vars[2].get(), symptom_vars[3].get(), symptom_vars[4].get()]

    # Initialize the symptom list with zeros
    l2 = [0] * len(l1)

    # Update the l2 list based on selected symptoms
    for k in range(len(l1)):
        if l1[k] in psymptoms:
            l2[k] = 1

    # Make prediction
    inputtest = [l2]
    predict = clf.predict(inputtest)
    predicted = predict[0]

    # Map prediction to disease
    result = "Not Found"
    if predicted in range(len(disease)):
        result = disease[predicted]

    # Display the result
    t1.delete("1.0", END)
    t1.insert(END, result)

# GUI setup
root = tk.Tk()
root.title("Disease Predictor")
root.geometry("600x500")
root.configure(bg='#2E2E2E')

PRIMARY_COLOR = '#2E2E2E'
SECONDARY_COLOR = '#4A90E2'
TEXT_COLOR = '#FFFFFF'
BUTTON_COLOR = '#7B8D8F'

# Header frame
header_frame = tk.Frame(root, bg=PRIMARY_COLOR)
header_frame.pack(fill=tk.X, pady=10)

title_label = tk.Label(header_frame, text="Disease Predictor", bg=PRIMARY_COLOR, fg=SECONDARY_COLOR, font=("Arial", 24, "bold"))
title_label.pack()

subtitle_label = tk.Label(header_frame, text="A Project by Shrimad Mishra", bg=PRIMARY_COLOR, fg=TEXT_COLOR, font=("Arial", 14))
subtitle_label.pack()

# Input frame
input_frame = tk.Frame(root, bg=PRIMARY_COLOR)
input_frame.pack(padx=20, pady=10)

name_label = tk.Label(input_frame, text="Name of the Patient:", bg=PRIMARY_COLOR, fg=TEXT_COLOR, font=("Arial", 12))
name_label.grid(row=0, column=0, pady=10, sticky=tk.W)
name_entry = tk.Entry(input_frame, width=40)
name_entry.grid(row=0, column=1, padx=10, pady=10)

symptoms = ["Select Here"] + l1
symptom_vars = [tk.StringVar(value="Select Here") for _ in range(5)]
symptom_labels = ["Symptom 1", "Symptom 2", "Symptom 3", "Symptom 4", "Symptom 5"]

for i, label_text in enumerate(symptom_labels):
    label = tk.Label(input_frame, text=label_text, bg=PRIMARY_COLOR, fg=TEXT_COLOR, font=("Arial", 12))
    label.grid(row=i+1, column=0, pady=5, sticky=tk.W)
    option_menu = ttk.Combobox(input_frame, textvariable=symptom_vars[i], values=symptoms, state='readonly')
    option_menu.grid(row=i+1, column=1, padx=10, pady=5)

# Button frame
button_frame = tk.Frame(root, bg=PRIMARY_COLOR)
button_frame.pack(pady=20)

buttons = [
    ("DecisionTree", DecisionTree, "red"),
    ("RandomForest", randomforest, "orange"),
    ("Naive Bayes", NaiveBayes, "green")
]

for i, (text, command, color) in enumerate(buttons):
    button = tk.Button(button_frame, text=text, bg=color, fg=TEXT_COLOR, font=("Arial", 12), command=command)
    button.grid(row=0, column=i, padx=10)

# Result frame
result_frame = tk.Frame(root, bg=PRIMARY_COLOR)
result_frame.pack(padx=20, pady=10)

result_labels = ["Decision Tree", "Random Forest", "Naive Bayes"]
result_texts = [tk.Text(result_frame, height=1, width=40, bg=BUTTON_COLOR, fg=TEXT_COLOR, font=("Arial", 12)) for _ in result_labels]

for i, label_text in enumerate(result_labels):
    label = tk.Label(result_frame, text=label_text, bg=PRIMARY_COLOR, fg=TEXT_COLOR, font=("Arial", 12))
    label.grid(row=i, column=0, pady=5, sticky=tk.W)
    result_texts[i].grid(row=i, column=1, padx=10)

t1, t2, t3 = result_texts

root.mainloop()
