import streamlit as st
import pandas as pd
st.set_page_config(
    page_title="Glossary of Football Attributes",
    page_icon="📘"
)


st.title("📘 Glossary of Football Attributes")

# Load glossary CSV
glossary = pd.read_csv("glossary.csv")  # adjust path if needed

# Optional search bar
search = st.text_input("🔍 Search for a term")

# Filter results
if search:
    # Search across multiple columns: Metric Name, Short Name, Abbreviation, and Definition
    mask = (
        glossary["Metric Name"].str.contains(search, case=False, na=False) |
        glossary["Short Name"].str.contains(search, case=False, na=False) |
        glossary["Abbreviation"].str.contains(search, case=False, na=False) |
        glossary["Definition"].str.contains(search, case=False, na=False)
    )
    df = glossary[mask]
else:
    df = glossary

# Display each term + definition
for _, row in df.iterrows():
    st.subheader(row["Metric Name"])
    if pd.notna(row.get("Short Name")):
        st.caption(f"Short Name: {row['Short Name']}")
    if pd.notna(row.get("Abbreviation")):
        st.caption(f"Abbreviation: {row['Abbreviation']}")
    if pd.notna(row.get("Category")):
        st.caption(f"Category: {row['Category']}")
    st.write(row["Definition"])
    st.markdown("---")