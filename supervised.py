import streamlit as st
import pandas as pd

# Läs in datasetet
subset = pd.read_csv('subset.csv')





st.write("First few rows of the dataset:")
st.write(subset.head(200))





#subset['Label'] = labels

#st.write("First rows with labels:")
#st.write(subset.head(206))

# Visa beskrivning för de första 100 raderna
for i in range(200):
    st.title("Row " + str(i+1))
    st.subheader("Headline:")
    st.write(subset['headline'].iloc[i])
    st.write(subset['working_hours_type.label'].iloc[i])
    st.subheader("Description:")
    st.write(subset['description.text'].iloc[i])




# Skapa en tom lista för att lagra etiketterna
#labels = []

#st.write("Data with labels:")
# Loopa igenom varje rad i datasetet
#for index, row in subset.head(100).iterrows():
    # Skriv ut raden
    #st.write(row)
     # Generera en unik nyckel för selectbox baserat på radens index
    #selectbox_key = f"label_selectbox_{index}"
    # Lägg till ett select-element för användaren att välja etikett
    #label = st.selectbox(f"Label for this row {index}:", options=["Ja", "Nej"], key=selectbox_key)
    # Lägg till etiketten i listan
    #labels.append(label)

# Lägg till den nya kolumnen med etiketterna i datasetet
#subset['Label'] = labels

# Skriv ut det uppdaterade datasetet
#st.write(subset)

#st.write("First few rows of the dataset:")
#st.write(subset.head(100))



