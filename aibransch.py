import pandas as pd
import json
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import streamlit as st
import os
import requests
import io
import zipfile

# Retry fetching data with a maximum of 3 attempts
for attempt in range(3):
    try:
        # Fetch data from URL and write to JSONL file
        if not os.path.exists("dataset.jsonl"):
            url = "https://data.jobtechdev.se/annonser/historiska/2023.jsonl.zip"

            zip_response = requests.get(url)
            zip_response.raise_for_status()

            with zipfile.ZipFile(io.BytesIO(zip_response.content)) as zip_file:
                file_name = zip_file.namelist()[0]
                with zip_file.open(file_name) as jsonl_file:
                    print("File loaded successfully!")

                    # Write to the JSONL file
                    with open("dataset.jsonl", "wb") as output_file:  # Use "wb" mode to write
                        for line in jsonl_file:
                            output_file.write(line)
                    print("JSONL data written successfully!")
        break  # Exit the retry loop if data fetching is successful
    except requests.exceptions.RequestException as e:
        print("Failed to fetch data (attempt {}): {}".format(attempt + 1, e))
    except Exception as e:
        print("Error:", e)


# Check if the CSV file exists
if os.path.isfile('subset.csv'):
    # If CSV exists, read it into DataFrame
    subset = pd.read_csv('subset.csv')
else:
    # Load the JSON file into a DataFrame 
    lines = []
    try:
        url = "https://data.jobtechdev.se/annonser/historiska/2023.jsonl.zip"
        zip_response = requests.get(url)
        zip_response.raise_for_status()

        with zipfile.ZipFile(io.BytesIO(zip_response.content)) as zip_file:
            file_name = zip_file.namelist()[0]
            with zip_file.open(file_name) as jsonl_file:
                print("File loaded successfully!")

                # Convert each line from JSON format to Python dictionary
                for i, line in enumerate(jsonl_file):
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

        # Write the subset DataFrame to CSV
        subset.to_csv('subset.csv', index=False)

        print("Done!")

    except requests.exceptions.RequestException as e:
        print("Failed to fetch data:", e)
    except Exception as e:
        print("Error:", e)

# Ladda in nltk:s stemmingfunktion för svenska
from nltk.stem.snowball import SnowballStemmer
stemmer_sv = SnowballStemmer("swedish")

# Ladda in stoppord för svenska från nltk
from nltk.corpus import stopwords
stop_words_sv = set(stopwords.words('swedish'))

# Ladda in engelska stoppord för att hantera engelska texter
stop_words_en = set(stopwords.words('english'))

# Ladda in punktuation från string
import string
punctuation = set(string.punctuation)

# Select only relevant columns and make a copy to avoid SettingWithCopyWarning
new_subset = subset[[
    'headline',
    'description.text'
]].copy()

# Define a function to preprocess text with stemming for both English and Swedish
def preprocess_text(text, language='english'):
    # Convert list of keywords to a single string and then convert to lowercase
    if isinstance(text, list):
        text = ' '.join(text)
    # Convert text to lowercase
    text = text.lower()
    # Tokenize the text
    tokens = nltk.word_tokenize(text)
    # Remove stop words and punctuation based on language
    if language == 'swedish':
        stop_words = stop_words_sv
        stemmer = stemmer_sv
    else:
        stop_words = stop_words_en
        stemmer = SnowballStemmer("english")
    tokens = [word for word in tokens if word not in stop_words and word not in punctuation]
    # Stem the tokens
    stemmed_tokens = [stemmer.stem(word) for word in tokens]
    # Join the stemmed tokens back into a single string
    text = ' '.join(stemmed_tokens)
    return text

# Apply text preprocessing with stemming
#new_subset['combined_text'] = new_subset.apply(lambda row: preprocess_text(row['headline'] + ' ' + row['description.text'], language='swedish'), axis=1)
new_subset['combined_text'] = new_subset.apply(lambda row: preprocess_text((row['headline'] if pd.notnull(row['headline']) else '') + ' ' + (row['description.text'] if pd.notnull(row['description.text']) else ''), language='swedish'), axis=1)


# Justera vektoriseringen för att använda olika parametrar
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(new_subset['combined_text'])

# Optimera antalet kluster med elbow method eller silhouette score
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

max_clusters = 10  # Justera detta beroende på dina data
silhouette_scores = []
for num_clusters in range(2, max_clusters + 1):
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(X)
    silhouette_scores.append(silhouette_score(X, kmeans.labels_))

# Plotta silhouette score för olika antal kluster
plt.plot(range(2, max_clusters + 1), silhouette_scores)
plt.xlabel('Antal kluster')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score för olika antal kluster')
plt.show()

# Antal kluster, inklusive "Lagerarbete och logistik" och "Övrigt"
optimal_num_clusters = 7  # Justera detta baserat på din analys

# Använd det optimala antalet kluster för att utföra klustringen
kmeans = KMeans(n_clusters=optimal_num_clusters)
kmeans.fit(X)

# Manuellt tilldela namn till varje kluster baserat på de framträdande orden eller nyckelorden
cluster_names = [
    "Teknologi och IT",
    "Hälsovård och medicin",
    "Utbildning och pedagogik",
    "Ekonomi och finans",
    "Försäljning och marknadsföring",
    "Lagerarbete och logistik",
    "Övrigt"
]

# Lägg till en ny kolumn i DataFrame för branschnamn
subset['industry'] = [cluster_names[label] for label in kmeans.labels_]

# Streamlit app
st.title('Jobtech Dataset Clustering')
# Display the DataFrame with cluster labels and industry names
st.write(subset)

