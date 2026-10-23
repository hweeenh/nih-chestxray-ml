import sqlite3
import pandas as pd

#1. Load CSV file into SQLite
#Point this at wherever CSV file is located
csv_path = "Data_Entry_2017.csv"

#Create a local SQLite database file (save to disk as a .db file)
conn = sqlite3.connect("chest_xray.db")

#Load CSV into pandas DataFrame first, then write to SQLite
df = pd.read_csv(csv_path)
df.to_sql("chest_xray", conn, if_exists="replace", index=False)
print("Database loaded. Rows:", len(df))

#2. Helper Function
#Makes running queries clean and readable throughout script
def query(sql):
    return pd.read_sql_query(sql, conn)

#3. Explore the Schema
#Start by understanding columns and data
print(query("PRAGMA table_info(chest_xray)"))
print(query("Select * FROM chest_xray LIMIT 5"))

#4. Query 1: Basic shape of data
print("\n--- Total records and unique patients ---")
print(query("""
    SELECT
        COUNT(*) AS total_records,
        COUNT(DISTINCT "Patient ID") AS unique_patients
    FROM chest_xray
"""))

#5. Query 2: Age distribution
print("\n--- Average patient age by view position ---")
print(query("""
    SELECT
        "View Position",
        ROUND(AVG("Patient Age"), 2) AS avg_age,
        COUNT(*) AS scan_count
    FROM chest_xray
    GROUP BY "View Position"
    ORDER BY scan_count DESC
"""))

#6. Query 3: Gender distribution
print("\n--- Gender distribution ---")
print(query("""
    SELECT
        "Patient Gender",
        COUNT(*) AS count,
        ROUND((COUNT(*) * 100.0 / (SELECT COUNT(*) FROM chest_xray)), 2) AS percentage
    FROM chest_xray
    GROUP BY "Patient Gender"
"""))

#7. Query 4: Disease prevalence
print("\n--- Disease prevalence ---")
print(query("""
    SELECT
        "Finding Labels",
        COUNT(*) AS count
    FROM chest_xray
    GROUP BY "Finding Labels"
    ORDER BY count DESC
    LIMIT 15
"""))

#8. Query 5: Follow-up patterns
print("\n--- Follow-up patterns ---")
print(query("""
    SELECT
        "Patient ID",
        MAX("Follow-up #") AS max_followup,
        COUNT(*) AS total_scans
    FROM chest_xray
    GROUP BY "Patient ID"
    HAVING MAX("Follow-up #") > 10
    ORDER BY max_followup DESC
    LIMIT 10
"""))

#9. Query 6: Findings by view position
#Does the view position affect findings?
print("\n--- Findings by view position ---")
print(query("""
    SELECT
        "View Position",
        "Finding Labels",
        COUNT(*) AS count
    FROM chest_xray
    GROUP BY "View Position", "Finding Labels"
    ORDER BY count DESC
    LIMIT 15
"""))

#10. Query 7: Age distribution by disease
#Does age affect disease prevalence?
print("\n--- Age distribution by disease ---")
print(query("""
    SELECT
        "Finding Labels",
        ROUND(AVG("Patient Age"), 2) AS avg_age,
        COUNT(*) AS scan_count
    FROM chest_xray
    GROUP BY "Finding Labels"
    ORDER BY scan_count DESC
    LIMIT 15
"""))

#11. Query 8: Findings by follow-up number
#Does disease affect follow-up frequency?
print("\n--- Findings by follow-up number ---")
print(query("""
    SELECT
        "Follow-up #",
        "Finding Labels",
        COUNT(*) AS count
    FROM chest_xray
    GROUP BY "Follow-up #", "Finding Labels"
    ORDER BY count DESC
    LIMIT 15
"""))

#12. Query 9: Disease prevalence by disease
print("\n--- Disease prevalence per disease ---")
print(query("""
WITH RECURSIVE split(disease, rest) AS (
    SELECT
        '',
        "Finding Labels" || '|'
    FROM chest_xray

    UNION ALL

    SELECT
        substr(rest, 0, instr(rest, '|')),
        substr(rest, instr(rest, '|') + 1)
    FROM split
    WHERE rest <> ''
)

SELECT
    disease AS "Disease",
    COUNT(*) AS prevalence
FROM split
WHERE disease <> ''
GROUP BY disease
ORDER BY prevalence DESC;
"""))

#13. Query 10: Which diseases commonly co-occur?
#Disease cooccurrence can be clinically significant. 
#Eg. Effusion with infiltration could suggest pneumonia with pleural involvement.
print("\n--- Commonly co-occurring diseases ---")
print(query("""
    SELECT
        "Finding Labels" AS "Disease Combination",
        COUNT(*) AS co_occurrence
    FROM chest_xray
    WHERE "Finding Labels" LIKE '%|%'
    GROUP BY "Finding Labels"
    ORDER BY co_occurrence DESC
    LIMIT 15
"""))


#Close connection to database
conn.close()