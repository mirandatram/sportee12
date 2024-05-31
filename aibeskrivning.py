import pandas as pd
import json
import streamlit as st
import openai
from openai import OpenAI

API_KEY = open('Open_AI_key', 'r').read()

client = OpenAI(
    api_key=API_KEY
) 

# Läs in API-nyckeln från filen
with open("Open_AI_key", "r") as file:
    api_key = file.read().strip()

# Ange din API-nyckel
openai.api_key = api_key

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

#select only these columns
subset = jobtech_dataset[[
    'id',
    'original_id',
    'headline',
    'number_of_vacancies',
    'experience_required',
    'driving_license_required',
    'detected_language',
    'description.text',
    'duration.label',
    'working_hours_type.label',
    'employer.name',
    'employer.workplace',
    'workplace_address.municipality',
    'workplace_address.region',
    'workplace_address.region_code',
    'keywords.extracted.occupation'
]]

# Streamlit app
st.title('Jobtech Dataset')
# Display the DataFrame
st.write(subset)

# Loop through the first 10 rows and display description text
st.subheader("Beskrivning för de första 10 annonserna (omformulerade av OpenAI):")
for i in range(10):
    st.write(f"Beskrivning för annons {i+1}:")
    
    # Anropa OpenAI för att omformulera beskrivningstexten
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Du är expert på att skriva snygga jobbannonser"},
            {"role": "user", "content": subset['description.text'].iloc[i]},
        ]
    )

    # Hämta och skriv ut den genererade omformulerade beskrivningen
    for choice in response.choices:
        simplified_description = choice.message.content
        st.write(simplified_description)

