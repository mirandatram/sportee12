import pandas as pd
import json
import streamlit as st
import openai

API_KEY = open('Open_AI_key', 'r').read()

client = openai.Client(api_key=API_KEY)

# Load the JSON file into a DataFrame 
lines = []
with open('dataset.jsonl', 'r') as file: 
    for i, line in enumerate(file):
        lines.append(line.strip())
        if i >= 9999:
            break

# Convert each line from JSON format to Python dictionary
data = [json.loads(line) for line in lines]

# If the JSON file has nested structures, pandas will automatically flatten them
jobtech_dataset = pd.json_normalize(data)

# Select only relevant columns
subset = jobtech_dataset[[
    'id',
    'headline',
    'keywords.extracted.occupation'
]].copy()

# Define function to generate industry using OpenAI API
def generate_industry(headline):
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Du är expert identifiera vilken bransch olika jobb tillhör"},
                {"role": "user", "content": f"Kan du hitta vilka branscher jobbannonsen \"{headline}\" tillhör?"}
            ]
        )
        st.write(response)
        return response.choices[0].message['content']
    except Exception as e:
        print(f"Error occurred: {e}")
        return "N/A"

# Apply function to generate industry for each job ad
subset['industry'] = subset['headline'].apply(generate_industry)

# Display the DataFrame with generated industries
st.title('Jobtech Dataset Clustering with OpenAI')
st.write(subset)
