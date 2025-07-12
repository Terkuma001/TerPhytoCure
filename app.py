import streamlit as st
from bs4 import BeautifulSoup
import requests
import pandas as pd
import pdfkit
import tempfile
from streamlit_option_menu import option_menu
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="🌿 PhytoCure", layout="centered")

# --- Developer Info ---
st.markdown("""
<div style="background-color:#f0f8ff; padding:10px; border-radius:10px;">
<h6>Developed by: <strong>Terkuma Saando</strong></h6>
<p>A bioinformatics enthusiast focused on plant-based drug discovery and AI-powered health insights.</p>
</div>
""", unsafe_allow_html=True)

# --- Language Translations ---
translations = {
    "title": {
        "English": "🌿 PhytoCure - Plant-Based Drug Discovery",
        "Español": "🌿 PhytoCure - Descubrimiento de Medicamentos Basados en Plantas",
        "हिंदी": "🌿 फाइटोक्योर - पौधे आधारित औषधि खोज",
        "Français": "🌿 PhytoCure - Découverte de médicaments à base de plantes"
    },
    "search_placeholder": {
        "English": "Enter botanical name...",
        "Español": "Ingrese el nombre botánico...",
        "हिंदी": "वनस्पति नाम दर्ज करें...",
        "Français": "Entrez le nom botanique..."
    }
}

# --- Sidebar Menu ---
lang = option_menu(
    menu_title=None,
    options=["English", "Español", "हिंदी", "Français"],
    icons=["globe", "flag", "flag", "flag"],
    default_index=0,
    orientation="horizontal"
)

st.title(translations["title"][lang])

# --- Utility Functions ---

def get_knapsack_compounds(plant_name):
    try:
        base_url = "https://kanaya.nuap.jp/servlet/SearchServlet "
        params = {"action": "search", "query": plant_name, "type": "plant"}
        response = requests.get(base_url, params=params)

        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            table = soup.find("table", {"class": "list_table"})
            compounds = []

            if table:
                rows = table.find_all("tr")[1:]  # Skip header row
                for row in rows:
                    cols = row.find_all("td")
                    if len(cols) >= 2:
                        compound = cols[0].get_text(strip=True)
                        percentage = cols[1].get_text(strip=True)
                        compounds.append({"Compound": compound, "Percentage Range": percentage})
                return compounds
        return []
    except Exception as e:
        st.error(f"Error fetching from KNApSAcK: {e}")
        return []

def scrape_dr_duke(plant_name):
    try:
        search_url = f"https://pfaf.org/user/Search.aspx?LatinName={plant_name}"
        response = requests.get(search_url)
        soup = BeautifulSoup(response.content, 'html.parser')

        common_name_tag = soup.find("span", id="ctl00_ContentPlaceHolder1_lblCommon")
        uses_section = soup.find("div", id="uses")

        result = {}
        if common_name_tag:
            result["Common Name"] = common_name_tag.text.strip()
        if uses_section:
            result["Uses"] = uses_section.text.strip()
        return result
    except Exception as e:
        return {}

def generate_pdf(data):
    html_content = f"""
    <h1 style='color:#2E8B57;'>🌿 PhytoCure Report</h1>
    <p><strong>Plant:</strong> {data['plant']}</p>
    <h3>Bioactive Compounds</h3>
    <ul>
    {''.join(f'<li>{c["Compound"]}: {c["Percentage Range"]}</li>' for c in data['compounds'])}
    </ul>
    """
    options = {
        'page-size': 'Letter',
        'margin-top': '0.75in',
        'margin-right': '0.75in',
        'margin-bottom': '0.75in',
        'margin-left': '0.75in',
    }
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmpfile:
        pdfkit.from_string(html_content, tmpfile.name, options=options)
        return tmpfile.name

# --- Disease Predictor AI Model ---
compound_disease_map = {
    "Curcumin": ["Arthritis", "Diabetes", "Cancer"],
    "Epigallocatechin gallate": ["Heart disease", "Obesity"],
    "Quercetin": ["Allergies", "Inflammation"],
    "Azadirachtin": ["Malaria", "Skin infection"]
}

vectorizer = TfidfVectorizer()
disease_descriptions = [
    "Anti-inflammatory and pain relief",
    "Metabolic regulation",
    "Immune system boost",
    "Antimicrobial properties",
    "Neuroprotective effects"
]

X = vectorizer.fit_transform(disease_descriptions)

def predict_diseases(compounds):
    unique_diseases = set()
    for compound in compounds:
        for disease in compound_disease_map.get(compound, []):
            unique_diseases.add(disease)

    if not unique_diseases:
        return ["No strong association found."]
    return list(unique_diseases)[:5]

# --- Main App Logic ---
plant_name = st.text_input("", placeholder=translations["search_placeholder"][lang])

if st.button("🔍 Search"):
    if plant_name:
        with st.spinner("Fetching data from global databases..."):

            knapsack_data = get_knapsack_compounds(plant_name)
            duke_data = scrape_dr_duke(plant_name)

            if knapsack_
                df = pd.DataFrame(knapsack_data)
                st.subheader("🧬 Bioactive Compounds")
                st.dataframe(df, use_container_width=True)
            else:
                st.info("⚠️ No compound data found in KNApSAcK database.")

            if duke_
                st.subheader("🌿 Uses & Properties")
                for key, value in duke_data.items():
                    st.markdown(f"**{key}:** {value[:500]}...")
            else:
                st.info("ℹ️ No additional information from Dr. Duke’s database.")

            bioactive_names = [c["Compound"] for c in knapsack_data]
            predicted_diseases = predict_diseases(bioactive_names)
            st.subheader("🧠 AI-Predicted Diseases")
            st.write(", ".join(predicted_diseases))

            report_data = {
                "plant": plant_name,
                "compounds": knapsack_data
            }

            pdf_path = generate_pdf(report_data)
            with open(pdf_path, "rb") as f:
                st.download_button("📄 Export as PDF", f, file_name=f"{plant_name}_report.pdf")

    else:
        st.warning("Please enter a plant name.")