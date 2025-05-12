import pandas as pd
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
import tkinter as tk
from tkinter import messagebox

# Load dataset
df = pd.read_csv("phishing.csv", encoding='utf-8')
df.columns = df.columns.str.strip()
df.dropna(inplace=True)

# Text preprocessing
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

df['EmailText'] = df['EmailText'].astype(str).apply(clean_text)
X = df['EmailText']
y = df['Label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

vectorizer = TfidfVectorizer()
X_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_vectors, y_train)

y_pred = model.predict(X_test_vectors)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nReport:\n", classification_report(y_test, y_pred))

# GUI Functions
def predict_email():
    email = email_input.get("1.0", "end-1c").strip()
    char_count_label.config(text=f"Characters: {len(email)}")
    
    if email == '':
        messagebox.showwarning("Input Error", "Please enter email text to classify.")
        result_label.config(text="Prediction: —", fg="#2c3e50")
        return

    email_clean = clean_text(email)
    email_vector = vectorizer.transform([email_clean])
    result = model.predict(email_vector)
    prob = model.predict_proba(email_vector).max()

    prediction = "Phishing ⚠️" if result[0] == 1 else "Legitimate ✅"
    result_label.config(text=f"Prediction: {prediction}\nConfidence: {round(prob*100, 2)}%", fg="#2c3e50")

# GUI Setup
root = tk.Tk()
root.title("Phishing Email Detector")
root.geometry("640x520")
root.config(bg="#edf2fb")

# Title
title_label = tk.Label(root, text="📧 Phishing Email Detector", font=("Helvetica", 20, "bold"), bg="#edf2fb", fg="#1c1c1c")
title_label.pack(pady=(20, 10))

# Instruction
instruction = tk.Label(root, text="Paste an email below to check if it's legitimate or phishing.", font=("Helvetica", 12), bg="#edf2fb", fg="#444")
instruction.pack()

# Input Box
email_input = tk.Text(root, height=10, width=60, font=("Helvetica", 12), wrap="word", bd=2, relief="groove")
email_input.pack(pady=(10, 5))

# Character Count Label
char_count_label = tk.Label(root, text="Characters: 0", font=("Helvetica", 10), bg="#edf2fb", fg="#555")
char_count_label.pack()

# Predict Button
predict_button = tk.Button(root, text="🧠 Analyze Email", font=("Helvetica", 14, "bold"), bg="#1d3557", fg="white", width=20, height=2, command=predict_email)
predict_button.pack(pady=20)

# Result Label
result_label = tk.Label(root, text="Prediction: —", font=("Helvetica", 14), bg="#edf2fb", fg="#2c3e50")
result_label.pack(pady=10)

# Footer
footer = tk.Label(root, text="Model: Naive Bayes | Vectorizer: TF-IDF", font=("Helvetica", 9), bg="#edf2fb", fg="#999")
footer.pack(side="bottom", pady=10)

root.mainloop()
