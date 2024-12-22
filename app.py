%%writefile app.py
from fastapi import FastAPI, HTTPException, Body
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.responses import HTMLResponse, FileResponse
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from pydantic import BaseModel
from typing import Optional
import uvicorn
import os
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import random
from googletrans import Translator
from difflib import get_close_matches
import matplotlib.pyplot as plt
import seaborn as sns

# Download required NLTK resources
nltk.download('wordnet')
nltk.download('stopwords')

# Initialize FastAPI app
app = FastAPI()

# Initialize tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
translator = Translator()

# Load CSV data
csv_path = "mentalhealth.csv"  # Update this path to your CSV location
df = pd.read_csv(csv_path)

# Preprocess data
def preprocess_text(text):
    text = str(text).lower()
    words = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]
    return " ".join(words)

df['processed_activity'] = df['activity'].apply(preprocess_text)

# Load model and tokenizer
model_id = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id)
pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)

# Emotion Mapping
emotion_mapping = {
    "LABEL_0": "happy",
    "LABEL_1": "stressed",
    "LABEL_2": "sad",
    "LABEL_3": "anxious",
    "LABEL_4": "angry",
    "LABEL_5": "joy",
}

# Request and response models
class MoodRequest(BaseModel):
    activity: str
    user_name: str
    language: Optional[str] = "en"

class MoodResponse(BaseModel):
    mood: str
    recommendation: str

# Match user activity with dataset
def match_activity_with_dataset(user_activity):
    preprocessed_activity = preprocess_text(user_activity)
    activities = df['processed_activity'].tolist()
    matches = get_close_matches(preprocessed_activity, activities, n=1, cutoff=0.6)
    if matches:
        matched_row = df[df['processed_activity'] == matches[0]].iloc[0]
        return matched_row['mood'], matched_row['recommendation']
    return None, None

# Generate recommendation
def generate_recommendation(predicted_mood):
    filtered_df = df[df['mood'].str.lower() == predicted_mood.lower()]
    if not filtered_df.empty:
        recommendation = random.choice(filtered_df['recommendation'].tolist())
        return recommendation
    return "Take care of yourself and try something relaxing!"

# Translate recommendation
def translate_recommendation(recommendation, target_language):
    try:
        translated = translator.translate(recommendation, dest=target_language)
        return translated.text
    except Exception as e:
        return recommendation  # Fallback to the original recommendation if translation fails


# Load user data helper function
def load_user_data():
    user_data_path = "user_data.csv"
    if os.path.exists(user_data_path):
        return pd.read_csv(user_data_path)
    else:
        raise FileNotFoundError(f"File '{user_data_path}' does not exist.")

# Helper function to return plots as base64 images
def get_plot_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    base64_img = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()
    return base64_img



# Serve the HTML directly as a response
@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Mood Analysis Dashboard</title>
        <script>
            // Function to fetch and display mood distribution pie chart
            async function fetchMoodDistribution() {
                const response = await fetch("/visualize-mood-distribution");
                const data = await response.json();
                if (data.image) {
                    document.getElementById("mood-distribution").innerHTML = `
                        <h2>Mood Distribution</h2>
                        <img src="data:image/png;base64,${data.image}" alt="Mood Distribution Pie Chart">
                    `;
                } else {
                    document.getElementById("mood-distribution").innerHTML = "<p>Failed to load chart.</p>";
                }
            }

            // Function to fetch and display activity time distribution
            async function fetchActivityTimeDistribution() {
                const response = await fetch("/visualize-activity-time");
                const data = await response.json();
                if (data.image) {
                    document.getElementById("activity-time-distribution").innerHTML = `
                        <h2>Activity Time Distribution</h2>
                        <img src="data:image/png;base64,${data.image}" alt="Activity Time Distribution">
                    `;
                } else {
                    document.getElementById("activity-time-distribution").innerHTML = "<p>Failed to load chart.</p>";
                }
            }

            // Function to fetch and display AI-generated recommendations
            async function fetchRecommendations() {
                const response = await fetch("/display-ai-recommendations");
                const data = await response.json();
                const recommendationsDiv = document.getElementById("ai-recommendations");

                if (data.message) {
                    recommendationsDiv.innerHTML = `<p>${data.message}</p>`;
                } else {
                    recommendationsDiv.innerHTML = "<h2>AI-generated Recommendations</h2>";
                    data.forEach(rec => {
                        recommendationsDiv.innerHTML += `
                            <p><strong>User ID:</strong> ${rec["User ID"]}</p>
                            <p><strong>Recommendation:</strong> ${rec["Recommendation"]}</p>
                            <hr>
                        `;
                    });
                }
            }

            // Fetch data when the page loads
            document.addEventListener("DOMContentLoaded", () => {
                fetchMoodDistribution();
                fetchActivityTimeDistribution();
                fetchRecommendations();
            });
        </script>
    </head>
    <body>
        <header>
            <h1>Mood Analysis Dashboard</h1>
        </header>

        <main>
            <section id="mood-distribution">
                <h2>Mood Distribution</h2>
                <p>Loading...</p>
            </section>

            <section id="activity-time-distribution">
                <h2>Activity Time Distribution</h2>
                <p>Loading...</p>
            </section>

            <section id="ai-recommendations">
                <h2>AI-generated Recommendations</h2>
                <p>Loading...</p>
            </section>
        </main>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

# Endpoint: Visualize Mood Distribution (Pie Chart)
@app.get("/visualize-mood-distribution")
async def visualize_mood_distribution_pie_chart():
    try:
        df = load_user_data()
        mood_counts = df["Matched Mood"].value_counts()

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.pie(
            mood_counts.values,
            labels=mood_counts.index,
            autopct='%1.1f%%',
            startangle=140,
            colors=sns.color_palette("pastel")[0:len(mood_counts)],
        )
        ax.set_title("Mood Distribution", fontsize=16)

        img_base64 = get_plot_base64(fig)
        plt.close(fig)

        return JSONResponse(content={"image": img_base64})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint: Visualize Activity Time Distribution
@app.get("/visualize-activity-time")
async def visualize_activity_time():
    try:
        df = load_user_data()
        df['Time'] = pd.to_datetime(df['Time'], format='%H:%M').dt.hour

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(df['Time'], bins=24, kde=False, color='blue', ax=ax)
        ax.set_title("Activity Time Distribution", fontsize=16)
        ax.set_xlabel("Hour of the Day", fontsize=12)
        ax.set_ylabel("Number of Activities", fontsize=12)
        ax.set_xticks(range(0, 24, 2))

        img_base64 = get_plot_base64(fig)
        plt.close(fig)

        return JSONResponse(content={"image": img_base64})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint: Display AI-generated Recommendations
@app.get("/display-ai-recommendations")
async def display_ai_generated_recommendations():
    try:
        user_data = load_user_data()
        if user_data.empty:
            return {"message": "No data available."}

        recommendations = []
        for idx, row in user_data.iterrows():
            recommendations.append({
                "User ID": row["User ID"],
                "Recommendation": row["Recommendation"],
            })

        return recommendations
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint: View User Profiles and Mood Patterns
@app.get("/view-user-profile/{user_id}")
async def view_user_profiles_and_mood_patterns(user_id: str):
    try:
        df = load_user_data()
        df["User ID"] = df["User ID"].astype(str)
        user_profile = df[df["User ID"] == user_id]

        if user_profile.empty:
            raise HTTPException(status_code=404, detail=f"No data found for User ID: {user_id}")

        user_name = user_profile["User Name"].iloc[0] if "User Name" in user_profile.columns else "N/A"
        mood_counts = user_profile["Matched Mood"].value_counts().to_dict()

        user_profile["DateTime"] = pd.to_datetime(user_profile["Date"] + " " + user_profile["Time"])
        user_profile.sort_values(by="DateTime", inplace=True)

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(
            data=user_profile,
            x="DateTime",
            y=user_profile["Matched Mood"].rank(method='dense'),
            hue="Matched Mood",
            marker="o",
            ax=ax,
        )
        ax.set_title(f"Mood Trend for User ID: {user_id}", fontsize=16)
        ax.set_xlabel("Date & Time", fontsize=12)
        ax.set_ylabel("Mood Rank (Ordinal Scale)", fontsize=12)
        ax.legend(title="Moods", loc="upper left")
        plt.xticks(rotation=45)
        plt.tight_layout()

        img_base64 = get_plot_base64(fig)
        plt.close(fig)

        return {
            "User ID": user_id,
            "User Name": user_name,
            "Mood Counts": mood_counts,
            "Mood Trend Plot": img_base64,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



# Endpoint: Predict Mood
@app.post("/predict_mood", response_model=MoodResponse)
async def predict_mood(request: MoodRequest):
    matched_mood, matched_recommendation = match_activity_with_dataset(request.activity)

    if matched_mood:
        translated_recommendation = translate_recommendation(matched_recommendation, request.language)
        return MoodResponse(mood=matched_mood, recommendation=translated_recommendation)

    processed_activity = preprocess_text(request.activity)
    result = pipe(processed_activity)
    predicted_label = result[0]['label']
    predicted_mood = emotion_mapping.get(predicted_label, "joy")
    recommendation = generate_recommendation(predicted_mood)
    translated_recommendation = translate_recommendation(recommendation, request.language)

    return MoodResponse(mood=predicted_mood, recommendation=translated_recommendation)

# Run the app
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
