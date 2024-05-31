import pandas as pd
import streamlit as st
import openai
from openai import OpenAI

import nltk
import string
import pickle
import os
from openai.types import Completion, CompletionChoice, CompletionUsage

from openai.types import ChatModel
from openai.types.chat import (
    ChatCompletion,
)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from nltk.stem.snowball import SnowballStemmer


print("Running...")

#----Den gråa sidopanelen----------------------------------------------------------------------------------------------------------------------#

#Vår logga
st.image('logo2.jpg', width=330)  
st.markdown("Det ska vara lätt att hitta jobb för just dig!")
st.markdown("---")
st.markdown('<br>', unsafe_allow_html=True)
st.markdown('<br>', unsafe_allow_html=True)


om_oss = """Vi på SPORTEE är idrottsentusiaster med ett intresse i entreprenöskap och teknikinovasion där vi delar tron på idrottens kraft att 
forma både individer och samhällen. Vår passion för idrotten driver oss att utveckla verktyg och resurser som syftar till att hjälpa idrottare 
att blomstra och skapa inkluderande miljöer inom idrottsföreningar.


Vi är övertygade om att varje idrottare förtjänar stöd och möjligheter att nå sin fulla potential, oavsett bakgrund eller nivå.
"""

vidare_lasning = """
Rapporten Swedish Elite Sport handlar om de svenska idrottarnas ekonomiska utmaningar, i jämförelse med våra grannar Norge och Danmark. 
Texten pekar på ett bristande svenskt idrottsstöd under utvecklingsfasen som har resulterat i den nuvarande ekonomiska osäkerheten 
hos våra svenska idrottare.
[Läs mer](https://www.idan.dk/media/stgjthhj/swedish-elite-sport.pdf)

How 5 Athletes Afford to Stay in the Game and Still Make Rent är en amerikansk artikel som handlar om hur idrottare, 
särskilt kvinnor och i de mindre populära idrottsgrenarna, globalt sett lever i en ekonomisk kamp och osäkerhet.
[Läs mer](https://www.thecut.com/2024/01/pro-athletes-working-second-jobs-careers.html)"""

kontakt_uppgifter = """
Python Consulant 
Vera Hertzman
Vera@outlook.com


Head of AI 
Thea Håkansson
Thea@gmail.se


Head of Streamlit 
Frida Eriksson
Royal@yahoo.com


Project Coordinator 
Miranda Tham
Miranda@hotmail.com


Agenda and Report Administrator Tove Lennertsson
Tove@gmail.com
"""

bakgrund = """SPORTEE föddes ur vår gemensamma passion för entreprenörskap, teknikinnovation och idrott. 
Som studenter med en stark drivkraft för att göra skillnad insåg vi behovet av att förnya stödet till 
idrottsföreningar i Sverige. Vår erfarenhet och intresse för entreprenörskap och teknik har varit centrala 
i vår resa. 

Vi har aktivt strävat efter att inte bara erbjuda extrajobb för idrottare, utan även att 
främja deras övergripande utveckling inom och utanför idrotten.
Genom att lyssna på och samarbeta med idrottsföreningar har vi fått värdefulla insikter som har varit 
avgörande för att forma vår plattform. 

Genom noggrann forskning och samråd har vi skapat SPORTEE - 
en banbrytande applikation som syftar till att revolutionera stödet för idrottsföreningar i Sverige 
och främja en mer hållbar och inkluderande idrottsmiljö.
 """

left_column = st.sidebar.container()

left_column.write("""
<style>
.left-column {
    background-color: #FF7F7F;
    width: 30%;
    padding: 20px;
    border-radius: 5px;
}
</style>
""", unsafe_allow_html=True)

                    #Texten i sidopanelen: annan text som vi kan lägga till
left_column.markdown("### Vi på <span style='color: #4a90e2;'>SPORTEE</span>", unsafe_allow_html=True)

#Om oss                  
with left_column.expander("💼 Om oss"):
    st.info(om_oss)

#Vidare läsning
with left_column.expander("📖   Vidare läsning"):
    st.info(vidare_lasning)

#Kontaktuppgifter
with left_column.expander("📫   Kontaktuppgifter"):
    st.info(kontakt_uppgifter)

# Bakgrund
with left_column.expander("📋   Projektets bakgrund"):
    st.info(bakgrund) 

#--API nyckel------------------------------------------------------------------------------------------------------------------------#

# API_KEY = 'sk-proj-mw9Jp4DlNLGyWNRS5g3xT3BlbkFJYypQktNpsa6iETrOqIHt'

# client = OpenAI(api_key=API_KEY)
# # Ange din API-nyckel
# client.api_key = API_KEY

# Läs in API-nyckeln från filen


#-----Read the CSV file into a DataFrame-----------------------------------------------------------------------------------------------------------#

@st.cache_data
def read_csv_file():
    subset = pd.read_csv('subset.csv')
    return subset

# Load data using @st.cache
subset = read_csv_file()
print("Almost done!")

#-----CLUSTERING---------------------------------------------------------------------------------------------------------------------#

# Ensure nltk resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')

# Paths to save the preprocessed data and models
STOP_WORDS_SV_PATH = 'stop_words_sv.pkl'
STOP_WORDS_EN_PATH = 'stop_words_en.pkl'
STEMMER_SV_PATH = 'stemmer_sv.pkl'
VECTORIZER_PATH = 'vectorizer.pkl'
KMEANS_PATH = 'kmeans.pkl'
PREPROCESSED_TEXT_PATH = 'preprocessed_text.csv'

# Load Swedish and English stemmers and stop words if not already saved
if not os.path.exists(STOP_WORDS_SV_PATH):
    stop_words_sv = set(stopwords.words('swedish'))
    with open(STOP_WORDS_SV_PATH, 'wb') as f:
        pickle.dump(stop_words_sv, f)
else:
    with open(STOP_WORDS_SV_PATH, 'rb') as f:
        stop_words_sv = pickle.load(f)

if not os.path.exists(STOP_WORDS_EN_PATH):
    stop_words_en = set(stopwords.words('english'))
    with open(STOP_WORDS_EN_PATH, 'wb') as f:
        pickle.dump(stop_words_en, f)
else:
    with open(STOP_WORDS_EN_PATH, 'rb') as f:
        stop_words_en = pickle.load(f)

if not os.path.exists(STEMMER_SV_PATH):
    stemmer_sv = SnowballStemmer("swedish")
    with open(STEMMER_SV_PATH, 'wb') as f:
        pickle.dump(stemmer_sv, f)
else:
    with open(STEMMER_SV_PATH, 'rb') as f:
        stemmer_sv = pickle.load(f)

punctuation = set(string.punctuation)

#En funktion som hanterar språket
def preprocess_text(text, language='english'):
    if isinstance(text, list):
        text = ' '.join(text)
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    if language == 'swedish':
        stop_words = stop_words_sv
        stemmer = stemmer_sv
    else:
        stop_words = stop_words_en
        stemmer = SnowballStemmer("english")
    tokens = [word for word in tokens if word not in stop_words and word not in punctuation]
    stemmed_tokens = [stemmer.stem(word) for word in tokens]
    text = ' '.join(stemmed_tokens)
    return text

#Funktion för klustringen och använd cache Function
@st.cache_data
def perform_clustering(subset, max_clusters=10):
    silhouette_scores = []
    if not os.path.exists(PREPROCESSED_TEXT_PATH):
        new_subset = subset[['headline', 'description.text']].copy()
        new_subset['combined_text'] = new_subset.apply(lambda row: preprocess_text(
            (row['headline'] if pd.notnull(row['headline']) else '') + ' ' + 
            (row['description.text'] if pd.notnull(row['description.text']) else ''), language='swedish'), axis=1)
        new_subset.to_csv(PREPROCESSED_TEXT_PATH, index=False)
    else:
        new_subset = pd.read_csv(PREPROCESSED_TEXT_PATH)
    
    if not os.path.exists(VECTORIZER_PATH) or not os.path.exists(KMEANS_PATH):
        vectorizer = TfidfVectorizer(max_features=5000)
        X = vectorizer.fit_transform(new_subset['combined_text'])

        for num_clusters in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=num_clusters)
            kmeans.fit(X)
            silhouette_scores.append(silhouette_score(X, kmeans.labels_))
        
        plt.plot(range(2, max_clusters + 1), silhouette_scores)
        plt.xlabel('Antal kluster')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Score för olika antal kluster')
        plt.show()

        optimal_num_clusters = 7  # Based on the analysis
        kmeans = KMeans(n_clusters=optimal_num_clusters)
        kmeans.fit(X)

        # Save the models
        with open(VECTORIZER_PATH, 'wb') as f:
            pickle.dump(vectorizer, f)
        with open(KMEANS_PATH, 'wb') as f:
            pickle.dump(kmeans, f)
    else:
        with open(VECTORIZER_PATH, 'rb') as f:
            vectorizer = pickle.load(f)
        with open(KMEANS_PATH, 'rb') as f:
            kmeans = pickle.load(f)

    X = vectorizer.transform(new_subset['combined_text'])
    
    cluster_names = [
        "Teknologi och IT",
        "Hälsovård och medicin",
        "Utbildning och pedagogik",
        "Ekonomi och finans",
        "Försäljning och marknadsföring",
        "Lagerarbete och logistik",
        "Övrigt"
    ]

    subset['industry'] = [cluster_names[label] for label in kmeans.labels_]
    return subset, silhouette_scores

# Perform clustering
subset, silhouette_scores = perform_clustering(subset)


# Your Streamlit code for displaying the bubble
st.markdown(
    """
    <div style="position: fixed; bottom: 20px; right: 20px; width: 90px; height: 40px; background-color: rgba(240, 240, 240, 0.8); border-radius: 10px; padding: 10px; display: flex; justify-content: center; align-items: center;">
        <div style="position: absolute; top: 50%; left: 100%; margin-top: -10px; width: 0; height: 0; border-top: 10px solid transparent; border-bottom: 10px solid transparent; border-left: 10px solid rgba(240, 240, 240, 0.8);"></div>
        <p style="margin: 0; color: #333;">Fråga oss!</p>
    </div> 
    """,
    unsafe_allow_html=True
)
print("clustering done!")

#--------------------------------------------------------------------------------------------------------------------------#
cluster_names = [
        "Teknologi och IT",
        "Hälsovård och medicin",
        "Utbildning och pedagogik",
        "Ekonomi och finans",
        "Försäljning och marknadsföring",
        "Lagerarbete och logistik",
        "Övrigt"
    ]


#Tabell där man kan filtrera med båda rullistorna
column_aliases = {
    'headline': 'headline',
    'employer.workplace': 'employer.workplace',
    'number_of_vacancies': 'number_of_vacancies',
    'description.text': 'description.text',
    'working_hours_type.label': 'working_hours_type.label',
    'workplace_address.region': 'workplace_address.region',
    'workplace_address.municipality': 'workplace_address.municipality',
    'duration.label': 'duration.label'
}

df = pd.read_csv("subset.csv")

places_list = subset['workplace_address.region'].dropna().unique().tolist()
places_list.insert(0, 'Visa alla')

time_of_work = subset['working_hours_type.label'].dropna().unique().tolist()
time_of_work.insert(0, 'Visa alla')

duration_time = subset['duration.label'].dropna().unique().tolist()
duration_time.insert(0, 'Visa alla')

# Visa titel 
st.subheader('Lediga jobb')

#SKAPA SÖKBOXAR
search_query = st.text_input('Sök efter specifika ord:', value="", help="Jobbtitel, nyckelord eller företag etc",)

region, form, time, branch = st.columns(4)
with region:
   selected_place = st.selectbox(f'Välj region:', places_list)

with form:
   selected_time_of_work = st.selectbox(f'Välj anställningsform:', time_of_work)

with time:
   selected_duration_time = st.selectbox(f'Välj tidsomfattning', duration_time)

with branch:
    selected_industry = st.selectbox("Välj bransch:", ['Visa alla'] + cluster_names)


if selected_place == 'Visa alla':
    region_condition = subset['workplace_address.region'].notna()
else:
    region_condition = subset['workplace_address.region'] == selected_place

if selected_time_of_work == 'Visa alla':
    time_of_work_condition = subset['working_hours_type.label'].notna()
else:
    time_of_work_condition = subset['working_hours_type.label'] == selected_time_of_work

if selected_duration_time == 'Visa alla':
    duration_condition = subset['duration.label'].notna()
else:
    duration_condition = subset['duration.label'] == selected_duration_time

if search_query:
    text_condition = df['description.text'].str.contains(search_query, case=False, na=False)
else:
    text_condition = pd.Series(True, index=df.index)  # Default condition to select all rows

if selected_industry == 'Visa alla':
    industry_condition = subset['industry'].notna()
else:
    industry_condition = subset['industry'] == selected_industry

# Filtered subset based on all conditions
filtered_subset = subset[(region_condition) & (time_of_work_condition) & (duration_condition) & (text_condition) & (industry_condition)]
filtered_subset = filtered_subset.rename(columns=column_aliases) 

job_count = filtered_subset.shape[0]

#---BESKRIVNINGAR MED HJÄLP AV AI-------------------------------------------------------------------------------------------------------------#

#Visar hur många lediga jobba som finns
st.markdown(f"<h1 style='font-weight: bold; color: #4a90e2'>{job_count} st </h1>", unsafe_allow_html=True)
st.markdown(f"<h6 style='font-weight: bold; color: black'>Jobb som matchar sökningen:</h6>", unsafe_allow_html=True)


# Välj endast dessa tre kolumner
ny_subset = filtered_subset[[
    'headline',
    'employer.workplace',  
    'description.text'
]]

print("starting with AI answers")


client = OpenAI(api_key=st.secrets['OPEN_API_KEY'])

# if:

#     # Använd API-nyckeln för att skapa OpenAI-klienten
#     client = openai.Client(api_key=)

#     # Resten av din kod
#     number = 5 
#     temp = st.empty()

#     with temp.container():
#         st.write("Laddar GPT...")
#         for i in range(min(len(ny_subset), number)):
#             st.write(f'#{i}')
#             with st.expander(f"Jobbannons {i+1} - {ny_subset['headline'].iloc[i]}"):
#                 st.write("-------------------------------------------------")
#                 response = client.chat.completions.create(
#                     model="gpt-3.5-turbo",
#                     messages=[
#                         {"role": "system", "content": """Du är expert på att skriva effektiva och snygga jobbannonser. 
#                         Alla annonser ska vara kortfattade, ha enhetliga rubriker och innehåll. 
#                          Skriv varje jobbannons på detta sätt.
#                          """},
#                         {"role": "user", "content": f"Sammanfatta denna annons till max 500 ord: {filtered_subset['description.text'].iloc[i]}"},
#                     ]
#                 )

#                 for choice in response.choices:
#                     simplified_description = choice.message.content
#                     st.write(f"{simplified_description}")

number = 5 

required_columns = ['headline', 'description.text']
missing_columns = [col for col in required_columns if col not in ny_subset.columns]

if missing_columns:
    st.error(f"Missing columns in the data: {', '.join(missing_columns)}")
else:
    number = 5
    temp = st.empty()
    with temp.container():
        print(f"Loading GPT")
        for i in range(min(len(ny_subset), number)):
            print(f'#{i}')
            headline = ny_subset['headline'].iloc[i]
            description_text = ny_subset['description.text'].iloc[i]

            with st.expander(f"Job Listing {i+1} - {headline}"):
                st.write("-------------------------------------------------")
                # Call OpenAI API to rephrase the job description
                try:
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": """You are an expert in writing effective and attractive job listings. 
                            All listings should be concise, have uniform headlines and content. 
                            Write each job listing in this way."""},
                            {"role": "user", "content": f"Summarize this job listing to a maximum of 200 words: {description_text}"},
                        ]
                    )
                    # Extract and display the generated rephrased description
                    simplified_description = response.choices[0].message.content
                    st.write(simplified_description)
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")


# #antalet jobb
# number = 5 
# temp = st.empty()

# #resultaten som visas
# with temp.container():
#     print("Laddar gpt")
#     for i in range(min(len(ny_subset), number)):
#         print(f'#{i}')
#         with st.expander(f"Jobbannons {i+1} - {ny_subset['headline'].iloc[i]}"):
#             st.write("-------------------------------------------------")
#             # Anropa OpenAI för att omformulera beskrivningstexten
#             response = client.chat.completions.create(
#                 model="gpt-3.5-turbo",
#                 messages=[
#                     {"role": "system", "content": """Du är expert på att skriva effektiva och snygga jobbannonser. 
#                     Alla annonser ska vara kortfattade, ha enhetliga rubriker och innehåll. 
#                      Skriv varje jobbannons på detta sätt.
#                      """},
#                     {"role": "user", "content": f"Sammanfatta denna annons till max 500 ord: {filtered_subset['description.text'].iloc[i]}"},
#                 ]
#             )

#             #Hämta och skriv ut den genererade omformulerade beskrivningen
#             for choice in response.choices:
#                 simplified_description = choice.message.content
#                 st.write(f"{simplified_description}")


#visa fler alternativ
if len(ny_subset) > number:
    if st.button('Visa fler'):
        temp.empty()
        number += 5
        temp = st.empty()
        with temp.container():
            for i in range(number - 5, min(len(ny_subset), number)):
                with st.expander(f"Jobbannons {i+1} - {ny_subset['headline'].iloc[i]}"):
                    st.write("-------------------------------------------------")
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": """Du är expert på att skriva effektiva och snygga jobbannonser. 
                    Alla annonser ska vara kortfattade, ha enhetliga rubriker och innehåll. 
                     Skriv varje jobbannons på detta sätt."""},
                            {"role": "user", "content": f"Sammanfatta denna annons till max 500 ord: {filtered_subset['description.text'].iloc[i]}"},
                        ]
                    )

                    #Hämta och skriv ut den genererade omformulerade beskrivningen
                    print('GPT HAR SVARAT')
                    for choice in response.choices:
                        simplified_description = choice.message.content
                        st.write(f"{simplified_description}")




#---SUPERVISED LEARNING----------------------------------------------------------------------------------------------------------------#

# Läs in data
df = pd.read_csv('subset.csv').head(206)
# Läs in 'Headline' från CSV-filen
pd.read_csv('subset.csv')['headline'].head(206)


# Skapa en kopia av den ursprungliga kolumnen
df['stemmed_text'] = df['description.text']

# Definiera stoppord
swedish_stop_words = set(stopwords.words('swedish'))
# Skapa en instans av SnowballStemmer för svenska
stemmer = SnowballStemmer('swedish')

# Funktion för textpreprocessering för en specifik kolumn
def preprocess_text_column(column):
    # Tokenisera texten i kolumnen
    column_tokens = [word_tokenize(text.lower(), language='swedish') for text in column]
    # Ta bort stoppord och ord med en längd mindre än 3, samt stamma ord
    preprocessed_column = []
    for tokens in column_tokens:
        filtered_tokens = [stemmer.stem(token) for token in tokens if token not in swedish_stop_words and len(token) > 2 and token.isalpha()]
        preprocessed_column.append(' '.join(filtered_tokens))  
    return preprocessed_column

# Preprocessa texten i kolumnen 'description.text'
df['stemmed_text'] = preprocess_text_column(df['stemmed_text'])

# Funktion för att extrahera viktiga ord från jobbannonser
def extract_manual_keywords():
    # Lista över manuellt valda viktiga ord
    manual_keywords = ["tävlingsinriktad", "35 timmar", "flexibelt arbete", "deltid", "extra personal"]
    
    return manual_keywords
# Extrahera de manuellt valda viktiga orden
manual_keywords = extract_manual_keywords()
# Lägg till de manuellt valda viktiga orden i vokabulären för TF-IDF-vektoriseringen
vectorizer = TfidfVectorizer(vocabulary=manual_keywords)


# Manuellt märkta etiketter för de första 200 raderna
labels = ["NEJ", "NEJ", "NEJ", "NEJ", "NEJ", "NEJ", "NEJ", "NEJ", "NEJ", "NEJ",
          "NEJ", "NEJ", "JA", "NEJ", "NEJ", "NEJ", "NEJ", "NEJ", "NEJ", "JA",
          "NEJ", "JA", "NEJ", "NEJ", "JA", "NEJ", "NEJ", "NEJ", "JA", "JA",
          "NEJ", "NEJ", "NEJ", "NEJ", "NEJ", "NEJ", "JA", "NEJ", "NEJ", "NEJ", 
          "NEJ", "JA", "JA", "JA", "NEJ", "NEJ", "NEJ", "NEJ", "NEJ",
          "NEJ", "NEJ", "JA", "NEJ", "JA", "NEJ", "JA", "NEJ", "NEJ", "NEJ",
          "NEJ", "NEJ", "NEJ", "NEJ", "NEJ", "NEJ", "NEJ", "NEJ", "NEJ", "NEJ",
          "NEJ", "NEJ", "NEJ", "NEJ", "NEJ", "NEJ", "NEJ", "NEJ", "NEJ", "NEJ",
          "NEJ", "NEJ", "NEJ", "NEJ", "NEJ", "NEJ", "NEJ", "NEJ", "NEJ", "NEJ",
          "NEJ", "NEJ", "NEJ", "NEJ", "NEJ", "NEJ", "NEJ", "NEJ", "NEJ", "JA",
          "NEJ", "NEJ", "NEJ", "NEJ", "NEJ", "NEJ", "NEJ", "NEJ", "NEJ", "NEJ",
          "JA", "NEJ", "NEJ", "NEJ", "NEJ", "NEJ", "NEJ", "JA", "JA", "JA", "NEJ",
          "NEJ", "NEJ", "NEJ", "NEJ", "NEJ", "NEJ", "JA", "JA", "JA", "JA", "JA",
          "JA", "JA", "JA", "JA", "JA", "JA", "JA", "JA", "JA", "JA", "JA", "JA",
          "JA", "NEJ", "NEJ", "JA", "JA", "NEJ", "NEJ", "NEJ", "NEJ", "JA", "NEJ",
          "NEJ", "NEJ", "NEJ", "NEJ", "NEJ", "JA", "NEJ", "JA", "NEJ", "JA", "NEJ", 
          "NEJ", "JA", "NEJ", "JA", "NEJ", "NEJ", "NEJ", "JA", "JA", "NEJ", "NEJ", 
          "NEJ", "NEJ", "NEJ", "NEJ", "JA", "NEJ", "JA", "JA", "NEJ", "NEJ", "NEJ", 
          "NEJ", "NEJ", "NEJ", "NEJ", "NEJ", "NEJ", "NEJ", "JA", "NEJ", "NEJ", "NEJ", 
          "NEJ", "NEJ", "JA", "JA", "NEJ", "JA", "JA", "NEJ"]

# Skapa en ny kolumn med namnet "label" och tilldela den dina manuellt märkta etiketter
df['label'] = labels[:len(df)]
# Ta bara de första 200 raderna som har en etikett
df_with_labels = df.dropna(subset=['label']).head(200)

# Förutsatt att ditt dataset finns i en DataFrame df med kolumnen "description.text" för jobbannonserna och "label" för etiketten
X = df_with_labels['stemmed_text']
y = df_with_labels['label']

# Dela upp data i träningsdata och testdata
# Dela upp data i träningsdata (150) och testdata (50) slumpmässigt
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=120, test_size=80, random_state=42)
# Skapa TF-IDF-vektorer från text med samma vokabulär som användes för träning
X_train_vectorized = vectorizer.fit_transform(X_train)
# Använd samma vektoriseringsinstans för att transformera testdatan
X_test_vectorized = vectorizer.transform(X_test)

# Välj modell (Logistisk regression) och ange viktade klasser
model = LogisticRegression(max_iter=1000, class_weight={'NEJ': 1, 'JA': 10})

# Träna modellen
model.fit(X_train_vectorized, y_train)
# Förutsäg på testdata
y_pred = model.predict(X_test_vectorized)
# Utvärdera modellens prestanda
print(classification_report(y_test, y_pred))
# Förutsäg lämpligheten för varje jobbannons i ditt dataset
df['prediction'] = model.predict(vectorizer.transform(df['stemmed_text']))
# Sortera DataFrame baserat på förutsägelserna för att få jobbannonserna i kronologisk ordning för vad som passar bäst med idrott
sorted_df = df.sort_values(by='prediction', ascending=False)

st.markdown("---")
st.subheader("AI-generator", help="Detta är endast en prototyp och inte en färdigt utvecklad modell")
info = """Nedan listar en AI de tre bäst lämpade arbeten för elitidrottare. 
Dessa förslag har utvecklats utifrån en supervised model 
som tränats för att ge bästa möjliga rekommendation."""

st.write(info)
st.markdown('<br>', unsafe_allow_html=True)
st.markdown("<h6 style='text-align:left;'>Top tre resultat:</h6>", unsafe_allow_html=True)


top_predictions = sorted_df[['headline','description.text', 'prediction']].head(3)


#Gpt genererade förslag utifrån filter
for i in range(len(top_predictions)):
        with st.expander(f"{top_predictions['headline'].iloc[i]}"):
            st.write("-------------------------------------------------")
            # Anropa OpenAI för att omformulera beskrivningstexten
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": """Du är expert på att skriva snygga jobbannonser 
                     och alla jobbanonser ska vara skrivna på samma sätt det vill säga med liknande rubriker och innehåll utan listor.
                     """},
                    {"role": "user", "content": top_predictions['description.text'].iloc[i]},
                ]
            )

            #Hämta och skriv ut den genererade omformulerade beskrivningen
            for choice in response.choices:
                simplified_description = choice.message.content
                st.write(f"{simplified_description}")

#--------------------------------------------------------------------------------------------------------------------------#

#Panelen längst ner
st.markdown('<br>', unsafe_allow_html=True)
st.markdown('<br>', unsafe_allow_html=True)


#Tjock linje innan
st.markdown(
    """
    <style>
        .line {
            width: 100%;
            height: 2px;
            background-color: black; /* Navy färg */
            margin-bottom: 20px;
        }
    </style>
    <div class="line"></div>
    """,
    unsafe_allow_html=True
)

#Info längst ner i kolumner
safety, ass, terms, sportee = st.columns(4)

with safety:
    st.markdown("<h6 style='text-align:left;'>Säkerhet</h6>", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align:left; font-weight: 500;'>Kundsäkerhet</h6>", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align:left; font-weight: 500;'>Hantering av kunduppgifter</h6>", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align:left; font-weight: 500;'>Falska mail</h6>", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align:left; font-weight: 500;'>Anmäl ett fel</h6>", unsafe_allow_html=True)
    

with ass:
    st.markdown("<h6 style='text-align:left;'>För föreingen</h6>", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align:left; font-weight: 500;'>Lägg till egen annons</h6>", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align:left; font-weight: 500;'>Ändra layout</h6>", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align:left; font-weight: 500;'>Visa alla jobb</h6>", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align:left; font-weight: 500;'>Inloggning för förenigar</h6>", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align:left; font-weight: 500;'>Administrera föreningsannonser</h6>", unsafe_allow_html=True)


with terms:
    st.markdown("<h6 style='text-align:left;'>Villkor</h6>", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align:left; font-weight: 500;'>Användarvillkor</h6>", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align:left; font-weight: 500;'>Personuppgiftshantering</h6>", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align:left; font-weight: 500;'>Cookies</h6>", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align:left; font-weight: 500;'>Cookiesinställningar</h6>", unsafe_allow_html=True)


with sportee:
    st.markdown("<h6 style='text-align:left;'>SPORTEE</h6>", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align:left; font-weight: 500;'>Om SPORTEE</h6>", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align:left; font-weight: 500;'>Press</h6>", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align:left; font-weight: 500;'>Jobba på SPORTEE</h6>", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align:left; font-weight: 500;'>Kontakta oss</h6>", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align:left; font-weight: 500;'>Inspiration</h6>", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align:left; font-weight: 500;'>Tips och guider</h6>", unsafe_allow_html=True)

# import pandas as pd
# import streamlit as st
# import openai
# from openai import OpenAI

# import nltk
# import string
# import pickle
# import os
# from openai.types import Completion, CompletionChoice, CompletionUsage

# from openai.types import ChatModel
# from openai.types.chat import (
#     ChatCompletion,
# )

# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.cluster import KMeans
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report
# from sklearn.metrics import silhouette_score
# import matplotlib.pyplot as plt

# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# from nltk.stem import SnowballStemmer
# from nltk.stem.snowball import SnowballStemmer


# print("Running...")

# #----Den gråa sidopanelen----------------------------------------------------------------------------------------------------------------------#

# #Vår logga
# st.image('logo2.jpg', width=330)  
# st.markdown("Det ska vara lätt att hitta jobb för just dig!")
# st.markdown("---")
# st.markdown('<br>', unsafe_allow_html=True)
# st.markdown('<br>', unsafe_allow_html=True)


# om_oss = """Vi på SPORTEE är idrottsentusiaster med ett intresse i entreprenöskap och teknikinovasion där vi delar tron på idrottens kraft att 
# forma både individer och samhällen. Vår passion för idrotten driver oss att utveckla verktyg och resurser som syftar till att hjälpa idrottare 
# att blomstra och skapa inkluderande miljöer inom idrottsföreningar.


# Vi är övertygade om att varje idrottare förtjänar stöd och möjligheter att nå sin fulla potential, oavsett bakgrund eller nivå.
# """

# vidare_lasning = """
# Rapporten Swedish Elite Sport handlar om de svenska idrottarnas ekonomiska utmaningar, i jämförelse med våra grannar Norge och Danmark. 
# Texten pekar på ett bristande svenskt idrottsstöd under utvecklingsfasen som har resulterat i den nuvarande ekonomiska osäkerheten 
# hos våra svenska idrottare.
# [Läs mer](https://www.idan.dk/media/stgjthhj/swedish-elite-sport.pdf)

# How 5 Athletes Afford to Stay in the Game and Still Make Rent är en amerikansk artikel som handlar om hur idrottare, 
# särskilt kvinnor och i de mindre populära idrottsgrenarna, globalt sett lever i en ekonomisk kamp och osäkerhet.
# [Läs mer](https://www.thecut.com/2024/01/pro-athletes-working-second-jobs-careers.html)"""

# kontakt_uppgifter = """
# Python Consulant 
# Vera Hertzman
# Vera@outlook.com


# Head of AI 
# Thea Håkansson
# Thea@gmail.se


# Head of Streamlit 
# Frida Eriksson
# Royal@yahoo.com


# Project Coordinator 
# Miranda Tham
# Miranda@hotmail.com


# Agenda and Report Administrator Tove Lennertsson
# Tove@gmail.com
# """

# bakgrund = """SPORTEE föddes ur vår gemensamma passion för entreprenörskap, teknikinnovation och idrott. 
# Som studenter med en stark drivkraft för att göra skillnad insåg vi behovet av att förnya stödet till 
# idrottsföreningar i Sverige. Vår erfarenhet och intresse för entreprenörskap och teknik har varit centrala 
# i vår resa. 

# Vi har aktivt strävat efter att inte bara erbjuda extrajobb för idrottare, utan även att 
# främja deras övergripande utveckling inom och utanför idrotten.
# Genom att lyssna på och samarbeta med idrottsföreningar har vi fått värdefulla insikter som har varit 
# avgörande för att forma vår plattform. 

# Genom noggrann forskning och samråd har vi skapat SPORTEE - 
# en banbrytande applikation som syftar till att revolutionera stödet för idrottsföreningar i Sverige 
# och främja en mer hållbar och inkluderande idrottsmiljö.
#  """

# left_column = st.sidebar.container()

# left_column.write("""
# <style>
# .left-column {
#     background-color: #FF7F7F;
#     width: 30%;
#     padding: 20px;
#     border-radius: 5px;
# }
# </style>
# """, unsafe_allow_html=True)

#                     #Texten i sidopanelen: annan text som vi kan lägga till
# left_column.markdown("### Vi på <span style='color: #4a90e2;'>SPORTEE</span>", unsafe_allow_html=True)

# #Om oss                  
# with left_column.expander("💼 Om oss"):
#     st.info(om_oss)

# #Vidare läsning
# with left_column.expander("📖   Vidare läsning"):
#     st.info(vidare_lasning)

# #Kontaktuppgifter
# with left_column.expander("📫   Kontaktuppgifter"):
#     st.info(kontakt_uppgifter)

# # Bakgrund
# with left_column.expander("📋   Projektets bakgrund"):
#     st.info(bakgrund) 

# #-----Read the CSV file into a DataFrame-----------------------------------------------------------------------------------------------------------#

# @st.cache_data
# def read_csv_file():
#     subset = pd.read_csv('subset.csv')
#     return subset

# # Load data using @st.cache
# subset = read_csv_file()
# print("Almost done!")

# #-----CLUSTERING---------------------------------------------------------------------------------------------------------------------#

# # Ensure nltk resources are downloaded
# nltk.download('punkt')
# nltk.download('stopwords')

# # Paths to save the preprocessed data and models
# STOP_WORDS_SV_PATH = 'stop_words_sv.pkl'
# STOP_WORDS_EN_PATH = 'stop_words_en.pkl'
# STEMMER_SV_PATH = 'stemmer_sv.pkl'
# VECTORIZER_PATH = 'vectorizer.pkl'
# KMEANS_PATH = 'kmeans.pkl'
# PREPROCESSED_TEXT_PATH = 'preprocessed_text.csv'

# # Load Swedish and English stemmers and stop words if not already saved
# if not os.path.exists(STOP_WORDS_SV_PATH):
#     stop_words_sv = set(stopwords.words('swedish'))
#     with open(STOP_WORDS_SV_PATH, 'wb') as f:
#         pickle.dump(stop_words_sv, f)
# else:
#     with open(STOP_WORDS_SV_PATH, 'rb') as f:
#         stop_words_sv = pickle.load(f)

# if not os.path.exists(STOP_WORDS_EN_PATH):
#     stop_words_en = set(stopwords.words('english'))
#     with open(STOP_WORDS_EN_PATH, 'wb') as f:
#         pickle.dump(stop_words_en, f)
# else:
#     with open(STOP_WORDS_EN_PATH, 'rb') as f:
#         stop_words_en = pickle.load(f)

# if not os.path.exists(STEMMER_SV_PATH):
#     stemmer_sv = SnowballStemmer("swedish")
#     with open(STEMMER_SV_PATH, 'wb') as f:
#         pickle.dump(stemmer_sv, f)
# else:
#     with open(STEMMER_SV_PATH, 'rb') as f:
#         stemmer_sv = pickle.load(f)

# punctuation = set(string.punctuation)

# #En funktion som hanterar språket
# def preprocess_text(text, language='english'):
#     if isinstance(text, list):
#         text = ' '.join(text)
#     text = text.lower()
#     tokens = nltk.word_tokenize(text)
#     if language == 'swedish':
#         stop_words = stop_words_sv
#         stemmer = stemmer_sv
#     else:
#         stop_words = stop_words_en
#         stemmer = SnowballStemmer("english")
#     tokens = [word for word in tokens if word not in stop_words and word not in punctuation]
#     stemmed_tokens = [stemmer.stem(word) for word in tokens]
#     text = ' '.join(stemmed_tokens)
#     return text

# #Funktion för klustringen och använd cache Function
# @st.cache_data
# def perform_clustering(subset, max_clusters=10):
#     silhouette_scores = []
#     if not os.path.exists(PREPROCESSED_TEXT_PATH):
#         new_subset = subset[['headline', 'description.text']].copy()
#         new_subset['combined_text'] = new_subset.apply(lambda row: preprocess_text(
#             (row['headline'] if pd.notnull(row['headline']) else '') + ' ' + 
#             (row['description.text'] if pd.notnull(row['description.text']) else ''), language='swedish'), axis=1)
#         new_subset.to_csv(PREPROCESSED_TEXT_PATH, index=False)
#     else:
#         new_subset = pd.read_csv(PREPROCESSED_TEXT_PATH)
    
#     if not os.path.exists(VECTORIZER_PATH) or not os.path.exists(KMEANS_PATH):
#         vectorizer = TfidfVectorizer(max_features=5000)
#         X = vectorizer.fit_transform(new_subset['combined_text'])

#         for num_clusters in range(2, max_clusters + 1):
#             kmeans = KMeans(n_clusters=num_clusters)
#             kmeans.fit(X)
#             silhouette_scores.append(silhouette_score(X, kmeans.labels_))
        
#         plt.plot(range(2, max_clusters + 1), silhouette_scores)
#         plt.xlabel('Antal kluster')
#         plt.ylabel('Silhouette Score')
#         plt.title('Silhouette Score för olika antal kluster')
#         plt.show()

#         optimal_num_clusters = 7  # Based on the analysis
#         kmeans = KMeans(n_clusters=optimal_num_clusters)
#         kmeans.fit(X)

#         # Save the models
#         with open(VECTORIZER_PATH, 'wb') as f:
#             pickle.dump(vectorizer, f)
#         with open(KMEANS_PATH, 'wb') as f:
#             pickle.dump(kmeans, f)
#     else:
#         with open(VECTORIZER_PATH, 'rb') as f:
#             vectorizer = pickle.load(f)
#         with open(KMEANS_PATH, 'rb') as f:
#             kmeans = pickle.load(f)

#     X = vectorizer.transform(new_subset['combined_text'])
    
#     cluster_names = [
#         "Teknologi och IT",
#         "Hälsovård och medicin",
#         "Utbildning och pedagogik",
#         "Ekonomi och finans",
#         "Försäljning och marknadsföring",
#         "Lagerarbete och logistik",
#         "Övrigt"
#     ]

#     subset['industry'] = [cluster_names[label] for label in kmeans.labels_]
#     return subset, silhouette_scores

# # Perform clustering
# subset, silhouette_scores = perform_clustering(subset)


# # Your Streamlit code for displaying the bubble
# st.markdown(
#     """
#     <div style="position: fixed; bottom: 20px; right: 20px; width: 90px; height: 40px; background-color: rgba(240, 240, 240, 0.8); border-radius: 10px; padding: 10px; display: flex; justify-content: center; align-items: center;">
#         <div style="position: absolute; top: 50%; left: 100%; margin-top: -10px; width: 0; height: 0; border-top: 10px solid transparent; border-bottom: 10px solid transparent; border-left: 10px solid rgba(240, 240, 240, 0.8);"></div>
#         <p style="margin: 0; color: #333;">Fråga oss!</p>
#     </div> 
#     """,
#     unsafe_allow_html=True
# )
# print("clustering done!")

# #--------------------------------------------------------------------------------------------------------------------------#
# cluster_names = [
#         "Teknologi och IT",
#         "Hälsovård och medicin",
#         "Utbildning och pedagogik",
#         "Ekonomi och finans",
#         "Försäljning och marknadsföring",
#         "Lagerarbete och logistik",
#         "Övrigt"
#     ]


# #Tabell där man kan filtrera med båda rullistorna
# column_aliases = {
#     'headline': 'headline',
#     'employer.workplace': 'employer.workplace',
#     'number_of_vacancies': 'number_of_vacancies',
#     'description.text': 'description.text',
#     'working_hours_type.label': 'working_hours_type.label',
#     'workplace_address.region': 'workplace_address.region',
#     'workplace_address.municipality': 'workplace_address.municipality',
#     'duration.label': 'duration.label'
# }

# df = pd.read_csv("subset.csv")

# places_list = subset['workplace_address.region'].dropna().unique().tolist()
# places_list.insert(0, 'Visa alla')

# time_of_work = subset['working_hours_type.label'].dropna().unique().tolist()
# time_of_work.insert(0, 'Visa alla')

# duration_time = subset['duration.label'].dropna().unique().tolist()
# duration_time.insert(0, 'Visa alla')

# # Visa titel 
# st.subheader('Lediga jobb')

# #SKAPA SÖKBOXAR
# search_query = st.text_input('Sök efter specifika ord:', value="", help="Jobbtitel, nyckelord eller företag etc",)

# region, form, time, branch = st.columns(4)
# with region:
#    selected_place = st.selectbox(f'Välj region:', places_list)

# with form:
#    selected_time_of_work = st.selectbox(f'Välj anställningsform:', time_of_work)

# with time:
#    selected_duration_time = st.selectbox(f'Välj tidsomfattning', duration_time)

# with branch:
#     selected_industry = st.selectbox("Välj bransch:", ['Visa alla'] + cluster_names)


# if selected_place == 'Visa alla':
#     region_condition = subset['workplace_address.region'].notna()
# else:
#     region_condition = subset['workplace_address.region'] == selected_place

# if selected_time_of_work == 'Visa alla':
#     time_of_work_condition = subset['working_hours_type.label'].notna()
# else:
#     time_of_work_condition = subset['working_hours_type.label'] == selected_time_of_work

# if selected_duration_time == 'Visa alla':
#     duration_condition = subset['duration.label'].notna()
# else:
#     duration_condition = subset['duration.label'] == selected_duration_time

# if search_query:
#     text_condition = df['description.text'].str.contains(search_query, case=False, na=False)
# else:
#     text_condition = pd.Series(True, index=df.index)  # Default condition to select all rows

# if selected_industry == 'Visa alla':
#     industry_condition = subset['industry'].notna()
# else:
#     industry_condition = subset['industry'] == selected_industry

# # Filtered subset based on all conditions
# filtered_subset = subset[(region_condition) & (time_of_work_condition) & (duration_condition) & (text_condition) & (industry_condition)]
# filtered_subset = filtered_subset.rename(columns=column_aliases) 

# job_count = filtered_subset.shape[0]

# #---BESKRIVNINGAR MED HJÄLP AV AI-------------------------------------------------------------------------------------------------------------#

# #Visar hur många lediga jobba som finns
# st.markdown(f"<h1 style='font-weight: bold; color: #4a90e2'>{job_count} st </h1>", unsafe_allow_html=True)
# st.markdown(f"<h6 style='font-weight: bold; color: black'>Jobb som matchar sökningen:</h6>", unsafe_allow_html=True)


# # Välj endast dessa tre kolumner
# ny_subset = filtered_subset[[
#     'headline',
#     'employer.workplace',  
#     'description.text'
# ]]

# print("starting with AI answers")

# from openai import OpenAI

# client = openai.Client(api_key='sk-proj-POdkU8Rnpa9RUOgaMMLYT3BlbkFJ3cALEFAkUFuvVNgeqQBj')
# number = 5 

# required_columns = ['headline', 'description.text']
# missing_columns = [col for col in required_columns if col not in ny_subset.columns]

# if missing_columns:
#     st.error(f"Missing columns in the data: {', '.join(missing_columns)}")
# else:
#     number = 5
#     temp = st.empty()
#     with temp.container():
#         print(f"Loading GPT")
#         for i in range(min(len(ny_subset), number)):
#             print(f'#{i}')
#             headline = ny_subset['headline'].iloc[i]
#             description_text = ny_subset['description.text'].iloc[i]

#             with st.expander(f"Job Listing {i+1} - {headline}"):
#                 st.write("-------------------------------------------------")
#                 # Call OpenAI API to rephrase the job description
#                 try:
#                     response = client.chat.completions.create(
#                         model="gpt-3.5-turbo",
#                         messages=[
#                             {"role": "system", "content": """You are an expert in writing effective and attractive job listings. 
#                             All listings should be concise, have uniform headlines and content. 
#                             Write each job listing in this way."""},
#                             {"role": "user", "content": f"Summarize this job listing to a maximum of 200 words: {description_text}"},
#                         ]
#                     )

#                     # Extract and display the generated rephrased description
#                     simplified_description = response.choices[0].message.content
#                     st.write(simplified_description)
#                 except Exception as e:
#                     st.error(f"An error occurred: {str(e)}")

                    
   
# #visa fler alternativ
# if len(ny_subset) > number:
#     if st.button('Visa fler'):
#         temp.empty()
#         number += 5
#         temp = st.empty()
#         with temp.container():
#             for i in range(number - 5, min(len(ny_subset), number)):
#                 with st.expander(f"Jobbannons {i+1} - {ny_subset['headline'].iloc[i]}"):
#                     st.write("-------------------------------------------------")
#                     response = client.chat.completions.create(
#                         model="gpt-3.5-turbo",
#                         messages=[
#                             {"role": "system", "content": """Du är expert på att skriva effektiva och snygga jobbannonser. 
#                     Alla annonser ska vara kortfattade, ha enhetliga rubriker och innehåll. 
#                      Skriv varje jobbannons på detta sätt."""},
#                             {"role": "user", "content": f"Sammanfatta denna annons till max 200 ord: {filtered_subset['description.text'].iloc[i]}"},
#                         ]
#                     )

#                     #Hämta och skriv ut den genererade omformulerade beskrivningen
#                     print('GPT HAR SVARAT')
#                     for choice in response.choices:
#                         simplified_description = choice.message.content
#                         st.write(f"{simplified_description}")


# #---SUPERVISED LEARNING----------------------------------------------------------------------------------------------------------------#

# # Läs in data
# df = pd.read_csv('subset.csv').head(206)
# # Läs in 'Headline' från CSV-filen
# pd.read_csv('subset.csv')['headline'].head(206)


# # Skapa en kopia av den ursprungliga kolumnen
# df['stemmed_text'] = df['description.text']

# # Definiera stoppord
# swedish_stop_words = set(stopwords.words('swedish'))
# # Skapa en instans av SnowballStemmer för svenska
# stemmer = SnowballStemmer('swedish')

# # Funktion för textpreprocessering för en specifik kolumn
# def preprocess_text_column(column):
#     # Tokenisera texten i kolumnen
#     column_tokens = [word_tokenize(text.lower(), language='swedish') for text in column]
#     # Ta bort stoppord och ord med en längd mindre än 3, samt stamma ord
#     preprocessed_column = []
#     for tokens in column_tokens:
#         filtered_tokens = [stemmer.stem(token) for token in tokens if token not in swedish_stop_words and len(token) > 2 and token.isalpha()]
#         preprocessed_column.append(' '.join(filtered_tokens))  
#     return preprocessed_column

# # Preprocessa texten i kolumnen 'description.text'
# df['stemmed_text'] = preprocess_text_column(df['stemmed_text'])

# # Funktion för att extrahera viktiga ord från jobbannonser
# def extract_manual_keywords():
#     # Lista över manuellt valda viktiga ord
#     manual_keywords = ["tävlingsinriktad", "35 timmar", "flexibelt arbete", "deltid", "extra personal"]
    
#     return manual_keywords
# # Extrahera de manuellt valda viktiga orden
# manual_keywords = extract_manual_keywords()
# # Lägg till de manuellt valda viktiga orden i vokabulären för TF-IDF-vektoriseringen
# vectorizer = TfidfVectorizer(vocabulary=manual_keywords)


# # Manuellt märkta etiketter för de första 200 raderna
# labels = ["NEJ", "NEJ", "NEJ", "NEJ", "NEJ", "NEJ", "NEJ", "NEJ", "NEJ", "NEJ",
#           "NEJ", "NEJ", "JA", "NEJ", "NEJ", "NEJ", "NEJ", "NEJ", "NEJ", "JA",
#           "NEJ", "JA", "NEJ", "NEJ", "JA", "NEJ", "NEJ", "NEJ", "JA", "JA",
#           "NEJ", "NEJ", "NEJ", "NEJ", "NEJ", "NEJ", "JA", "NEJ", "NEJ", "NEJ", 
#           "NEJ", "JA", "JA", "JA", "NEJ", "NEJ", "NEJ", "NEJ", "NEJ",
#           "NEJ", "NEJ", "JA", "NEJ", "JA", "NEJ", "JA", "NEJ", "NEJ", "NEJ",
#           "NEJ", "NEJ", "NEJ", "NEJ", "NEJ", "NEJ", "NEJ", "NEJ", "NEJ", "NEJ",
#           "NEJ", "NEJ", "NEJ", "NEJ", "NEJ", "NEJ", "NEJ", "NEJ", "NEJ", "NEJ",
#           "NEJ", "NEJ", "NEJ", "NEJ", "NEJ", "NEJ", "NEJ", "NEJ", "NEJ", "NEJ",
#           "NEJ", "NEJ", "NEJ", "NEJ", "NEJ", "NEJ", "NEJ", "NEJ", "NEJ", "JA",
#           "NEJ", "NEJ", "NEJ", "NEJ", "NEJ", "NEJ", "NEJ", "NEJ", "NEJ", "NEJ",
#           "JA", "NEJ", "NEJ", "NEJ", "NEJ", "NEJ", "NEJ", "JA", "JA", "JA", "NEJ",
#           "NEJ", "NEJ", "NEJ", "NEJ", "NEJ", "NEJ", "JA", "JA", "JA", "JA", "JA",
#           "JA", "JA", "JA", "JA", "JA", "JA", "JA", "JA", "JA", "JA", "JA", "JA",
#           "JA", "NEJ", "NEJ", "JA", "JA", "NEJ", "NEJ", "NEJ", "NEJ", "JA", "NEJ",
#           "NEJ", "NEJ", "NEJ", "NEJ", "NEJ", "JA", "NEJ", "JA", "NEJ", "JA", "NEJ", 
#           "NEJ", "JA", "NEJ", "JA", "NEJ", "NEJ", "NEJ", "JA", "JA", "NEJ", "NEJ", 
#           "NEJ", "NEJ", "NEJ", "NEJ", "JA", "NEJ", "JA", "JA", "NEJ", "NEJ", "NEJ", 
#           "NEJ", "NEJ", "NEJ", "NEJ", "NEJ", "NEJ", "NEJ", "JA", "NEJ", "NEJ", "NEJ", 
#           "NEJ", "NEJ", "JA", "JA", "NEJ", "JA", "JA", "NEJ"]

# # Skapa en ny kolumn med namnet "label" och tilldela den dina manuellt märkta etiketter
# df['label'] = labels[:len(df)]
# # Ta bara de första 200 raderna som har en etikett
# df_with_labels = df.dropna(subset=['label']).head(200)

# # Förutsatt att ditt dataset finns i en DataFrame df med kolumnen "description.text" för jobbannonserna och "label" för etiketten
# X = df_with_labels['stemmed_text']
# y = df_with_labels['label']

# # Dela upp data i träningsdata och testdata
# # Dela upp data i träningsdata (150) och testdata (50) slumpmässigt
# X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=120, test_size=80, random_state=42)
# # Skapa TF-IDF-vektorer från text med samma vokabulär som användes för träning
# X_train_vectorized = vectorizer.fit_transform(X_train)
# # Använd samma vektoriseringsinstans för att transformera testdatan
# X_test_vectorized = vectorizer.transform(X_test)

# # Välj modell (Logistisk regression) och ange viktade klasser
# model = LogisticRegression(max_iter=1000, class_weight={'NEJ': 1, 'JA': 10})

# # Träna modellen
# model.fit(X_train_vectorized, y_train)
# # Förutsäg på testdata
# y_pred = model.predict(X_test_vectorized)
# # Utvärdera modellens prestanda
# print(classification_report(y_test, y_pred))
# # Förutsäg lämpligheten för varje jobbannons i ditt dataset
# df['prediction'] = model.predict(vectorizer.transform(df['stemmed_text']))
# # Sortera DataFrame baserat på förutsägelserna för att få jobbannonserna i kronologisk ordning för vad som passar bäst med idrott
# sorted_df = df.sort_values(by='prediction', ascending=False)

# st.markdown("---")
# st.subheader("AI-generator", help="Detta är endast en prototyp och inte en färdigt utvecklad modell")
# info = """Nedan listar en AI de tre bäst lämpade arbeten för elitidrottare. 
# Dessa förslag har utvecklats utifrån en supervised model 
# som tränats för att ge bästa möjliga rekommendation."""

# st.write(info)
# st.markdown('<br>', unsafe_allow_html=True)
# st.markdown("<h6 style='text-align:left;'>Top tre resultat:</h6>", unsafe_allow_html=True)


# top_predictions = sorted_df[['headline','description.text', 'prediction']].head(3)

# #Gpt genererade förslag utifrån filter
# for i in range(len(top_predictions)):
#         with st.expander(f"{top_predictions['headline'].iloc[i]}"):
#             st.write("-------------------------------------------------")
#             # Anropa OpenAI för att omformulera beskrivningstexten
#             response = client.chat.completions.create(
#                 model="gpt-3.5-turbo",
#                 messages=[
#                     {"role": "system", "content": """Du är expert på att skriva snygga jobbannonser 
#                      och alla jobbanonser ska vara skrivna på samma sätt det vill säga med liknande rubriker och innehåll utan listor.
#                      """},
#                     {"role": "user", "content": top_predictions['description.text'].iloc[i]},
#                 ]
#             )

#             #Hämta och skriv ut den genererade omformulerade beskrivningen
#             for choice in response.choices:
#                 simplified_description = choice.message.content
#                 st.write(f"{simplified_description}")

# #--------------------------------------------------------------------------------------------------------------------------#

# #Panelen längst ner
# st.markdown('<br>', unsafe_allow_html=True)
# st.markdown('<br>', unsafe_allow_html=True)


# #Tjock linje innan
# st.markdown(
#     """
#     <style>
#         .line {
#             width: 100%;
#             height: 2px;
#             background-color: black; /* Navy färg */
#             margin-bottom: 20px;
#         }
#     </style>
#     <div class="line"></div>
#     """,
#     unsafe_allow_html=True
# )

# #Info längst ner i kolumner
# safety, ass, terms, sportee = st.columns(4)

# with safety:
#     st.markdown("<h6 style='text-align:left;'>Säkerhet</h6>", unsafe_allow_html=True)
#     st.markdown("<h6 style='text-align:left; font-weight: 500;'>Kundsäkerhet</h6>", unsafe_allow_html=True)
#     st.markdown("<h6 style='text-align:left; font-weight: 500;'>Hantering av kunduppgifter</h6>", unsafe_allow_html=True)
#     st.markdown("<h6 style='text-align:left; font-weight: 500;'>Falska mail</h6>", unsafe_allow_html=True)
#     st.markdown("<h6 style='text-align:left; font-weight: 500;'>Anmäl ett fel</h6>", unsafe_allow_html=True)
    

# with ass:
#     st.markdown("<h6 style='text-align:left;'>För föreingen</h6>", unsafe_allow_html=True)
#     st.markdown("<h6 style='text-align:left; font-weight: 500;'>Lägg till egen annons</h6>", unsafe_allow_html=True)
#     st.markdown("<h6 style='text-align:left; font-weight: 500;'>Ändra layout</h6>", unsafe_allow_html=True)
#     st.markdown("<h6 style='text-align:left; font-weight: 500;'>Visa alla jobb</h6>", unsafe_allow_html=True)
#     st.markdown("<h6 style='text-align:left; font-weight: 500;'>Inloggning för förenigar</h6>", unsafe_allow_html=True)
#     st.markdown("<h6 style='text-align:left; font-weight: 500;'>Administrera föreningsannonser</h6>", unsafe_allow_html=True)


# with terms:
#     st.markdown("<h6 style='text-align:left;'>Villkor</h6>", unsafe_allow_html=True)
#     st.markdown("<h6 style='text-align:left; font-weight: 500;'>Användarvillkor</h6>", unsafe_allow_html=True)
#     st.markdown("<h6 style='text-align:left; font-weight: 500;'>Personuppgiftshantering</h6>", unsafe_allow_html=True)
#     st.markdown("<h6 style='text-align:left; font-weight: 500;'>Cookies</h6>", unsafe_allow_html=True)
#     st.markdown("<h6 style='text-align:left; font-weight: 500;'>Cookiesinställningar</h6>", unsafe_allow_html=True)


# with sportee:
#     st.markdown("<h6 style='text-align:left;'>SPORTEE</h6>", unsafe_allow_html=True)
#     st.markdown("<h6 style='text-align:left; font-weight: 500;'>Om SPORTEE</h6>", unsafe_allow_html=True)
#     st.markdown("<h6 style='text-align:left; font-weight: 500;'>Press</h6>", unsafe_allow_html=True)
#     st.markdown("<h6 style='text-align:left; font-weight: 500;'>Jobba på SPORTEE</h6>", unsafe_allow_html=True)
#     st.markdown("<h6 style='text-align:left; font-weight: 500;'>Kontakta oss</h6>", unsafe_allow_html=True)
#     st.markdown("<h6 style='text-align:left; font-weight: 500;'>Inspiration</h6>", unsafe_allow_html=True)
#     st.markdown("<h6 style='text-align:left; font-weight: 500;'>Tips och guider</h6>", unsafe_allow_html=True)

