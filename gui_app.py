import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load & train model
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text

df = pd.read_csv("product_reviews.csv")
df['sentiment'] = df['sentiment'].str.lower().str.strip()
df['clean_review'] = df['review'].apply(clean_text)

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['clean_review'])
y = df['sentiment']
model = MultinomialNB()
model.fit(X, y)

# GUI Setup
app = tk.Tk()
app.title("💬 Sentiment Analyzer")
app.geometry("700x500")
app.configure(bg="#e6f2ff")  # Light sky background

# Fonts
TITLE_FONT = ("Segoe UI", 22, "bold")
LABEL_FONT = ("Segoe UI", 14)
ENTRY_FONT = ("Segoe UI", 13)
RESULT_FONT = ("Segoe UI", 16, "bold")

# Header Frame
header = tk.Frame(app, bg="#e6f2ff")
header.pack(pady=30)

tk.Label(header, text="Product Review Sentiment Analyzer", font=TITLE_FONT, bg="#e6f2ff", fg="#003366").pack()

# Body Frame
body = tk.Frame(app, bg="#ffffff", padx=20, pady=20, relief="groove", bd=2)
body.pack(pady=10)

tk.Label(body, text="Enter your review below:", font=LABEL_FONT, bg="#ffffff", fg="#222").pack(anchor="w", pady=(0, 10))

entry = tk.Text(body, width=60, height=5, font=ENTRY_FONT, bd=1, relief="solid", wrap="word", padx=8, pady=5)
entry.pack()

# Prediction Label
output_label = tk.Label(app, text="", font=RESULT_FONT, bg="#e6f2ff", fg="#006699")
output_label.pack(pady=20)

# Prediction Function
def predict_sentiment():
    review = entry.get("1.0", tk.END).strip()
    if not review:
        messagebox.showwarning("Empty Input", "Please enter a product review.")
        return
    cleaned = clean_text(review)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    output_label.config(text=f"Predicted Sentiment: {prediction.capitalize()}")

# Stylish Button
predict_button = tk.Button(app, text="🔍 Analyze Sentiment", font=("Segoe UI", 13, "bold"),
                           bg="#3399ff", fg="white", activebackground="#2673cc",
                           relief="flat", padx=20, pady=10, command=predict_sentiment)
predict_button.pack()

# Footer
tk.Label(app, text="Made with ❤ using Naive Bayes", font=("Segoe UI", 10),
         bg="#e6f2ff", fg="#888").pack(side="bottom", pady=10)

app.mainloop()