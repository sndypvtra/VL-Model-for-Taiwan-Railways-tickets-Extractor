# parse_ticket.py
import re
import pandas as pd
import json

STATIONS_CSV = "stations_database_modified.csv"
df_stations = pd.read_csv(STATIONS_CSV)

def is_chinese(text: str) -> bool:
    """True jika text mengandung huruf China."""
    return bool(re.search(r'[\u4e00-\u9fff]+', text))

def parse_single_ticket_text(text: str) -> dict:
    """
    Parsing khusus single ticket (hasil OCR raw text).
    Return dict: { date, departure_station, arrival_station, departure_time, arrival_time, price }
    """
    data = {
        "date": "",
        "departure_station": "",
        "arrival_station": "",
        "departure_time": "",
        "arrival_time": "",
        "price": ""
    }

    # 1) Tanggal
    match_date = re.search(r"\d{4}[./]\d{2}[./]\d{2}", text)
    if match_date:
        data["date"] = match_date.group(0)

    # 2) Stasiun (cari pattern "xxx -> yyy" dengan kemungkinan separator yang berbeda)
    match_stations = re.search(r"(.+?)\s*(?:\+|->|to|→|至)\s*(.+)", text, re.IGNORECASE)
    if match_stations:
        departure_raw = match_stations.group(1).strip()
        arrival_raw = match_stations.group(2).strip()

        # Hapus digit
        dep_clean = re.sub(r'\d+', '', departure_raw).strip()
        arr_clean = re.sub(r'\d+', '', arrival_raw).strip()

        # Ambil hanya huruf Latin + spasi
        dep_alphabet = re.sub(r'[^A-Za-z ]+', '', dep_clean).strip()
        arr_alphabet = re.sub(r'[^A-Za-z ]+', '', arr_clean).strip()

        if dep_alphabet:
            data["departure_station"] = dep_alphabet
        if arr_alphabet:
            data["arrival_station"] = arr_alphabet

    # Fallback: jika masih kosong, gunakan ekstraksi dari kata yang diawali huruf kapital
    if not data["departure_station"] or not data["arrival_station"]:
        stations_capital = re.findall(r"\b[A-Z][a-z]{4,}\b", text)
        excluded_words = {"Train", "Car", "Seat", "PSGR", "Ticket", "Time", "None"}
        stations_capital = [w for w in stations_capital if w not in excluded_words]

        if stations_capital:
            if not data["departure_station"]:
                data["departure_station"] = stations_capital[0]
            if len(stations_capital) > 1 and not data["arrival_station"]:
                data["arrival_station"] = stations_capital[1]

    # Logika tambahan: ekstrak stasiun bertuliskan Taiwan (karakter Cina) jika salah satu masih kosong
    if not data["departure_station"] or not data["arrival_station"]:
        # Cari pola dengan karakter Cina dan separator yang sama
        match_tw = re.search(r"([\u4e00-\u9fff]+)\s*(?:\+|->|to|→|至)\s*([\u4e00-\u9fff]+)", text)
        if match_tw:
            if not data["departure_station"]:
                data["departure_station"] = match_tw.group(1).strip()
            if not data["arrival_station"]:
                data["arrival_station"] = match_tw.group(2).strip()

    # 3) Waktu
    times = re.findall(r"(\d{1,2}[:：]\d{2})", text)
    times = [t.replace("：", ":") for t in times]
    if len(times) > 0:
        data["departure_time"] = times[0]
    if len(times) > 1:
        data["arrival_time"] = times[1]

    # 4) Harga
    match_price = re.search(r"(?:NT\$?\s?(\d+)|(\d+)\s*元)", text)
    if match_price:
        data["price"] = match_price.group(1) if match_price.group(1) else match_price.group(2)

    return data

def parse_multi_ticket_json(raw_json: str):
    """
    1) Bersihkan markdown
    2) Parse JSON jadi list of dict
    3) Jika departure_station adalah Chinese & cocok di startStaName, ganti dengan startStaEName
       Jika arrival_station adalah Chinese & cocok di endStaName, ganti dengan endStaEName
       Kalau tidak cocok, biarkan apa adanya.
    """

    import re
    import json

    # (a) Bersihkan format markdown
    cleaned = raw_json.replace("```", "").replace("json\n", "").strip()

    # Hapus teks yang tidak diinginkan (jika ada).
    # Contoh, kalau Qwen memunculkan tambahan keterangan di luar JSON
    # Anda bisa pakai regex atau .split() dsb.
    cleaned = re.sub(r"(?m)^\s*[-—]*\s*票據文本如下：\s*\n", "", cleaned).strip()

    # (b) Coba parse JSON dengan beberapa fallback
    parsed = None
    try:
        # Jika langsung berupa array JSON
        if cleaned.startswith("["):
            parsed = json.loads(cleaned)

        # Jika langsung berupa object JSON
        elif cleaned.startswith("{"):
            # Periksa apakah ada beberapa objek
            json_objects = re.findall(r'\{.*?\}', cleaned, re.DOTALL)

            if len(json_objects) == 1:
                parsed = json.loads(json_objects[0])
            else:
                # Jika ada beberapa object JSON, parse semuanya lalu jadikan list
                parsed_list = []
                for obj in json_objects:
                    try:
                        parsed_list.append(json.loads(obj))
                    except:
                        pass
                parsed = parsed_list if parsed_list else []
        else:
            # Terakhir, coba parse langsung
            parsed = json.loads(cleaned)

    except json.JSONDecodeError:
        # Kalau tetap gagal parse, Anda bisa memutuskan mau return apa
        # supaya aplikasi tidak langsung crash
        print("Warning: Gagal decode JSON dari output model.")
        return []  # fallback: kosong saja

    # (c) Normalisasi => pastikan `parsed` berbentuk list
    if isinstance(parsed, dict):
        parsed = [parsed]
    elif not isinstance(parsed, list):
        # Kalau aneh, ya fallback jadi list kosong
        parsed = []

    # (d) Cocokkan departure_station & arrival_station ke data stasiun
    for ticket in parsed:
        dep = ticket.get("departure_station", "")
        arr = ticket.get("arrival_station", "")

        matching_dep = df_stations[df_stations["startStaName"] == dep]
        if not matching_dep.empty:
            ticket["departure_station"] = matching_dep.iloc[0]["startStaEName"]

        matching_arr = df_stations[df_stations["endStaName"] == arr]
        if not matching_arr.empty:
            ticket["arrival_station"] = matching_arr.iloc[0]["endStaEName"]

    return parsed



def add_mileage_to_ticket(ticket: dict) -> dict:
    dep = ticket.get("departure_station", "").strip()
    arr = ticket.get("arrival_station", "").strip()

    # Jika masih kosong
    if not dep or not arr:
        ticket["mileage"] = None
        return ticket

    # Pilih kolom (is_chinese akan False jika dep Latin)
    departure_col = "startStaName" if is_chinese(dep) else "startStaEName"
    arrival_col   = "endStaName"   if is_chinese(arr) else "endStaEName"

    # Samakan ke format lower() agar match
    dep_lower = dep.lower()
    arr_lower = arr.lower()

    # Bandingkan df
    matching_rows = df_stations[
        (df_stations[departure_col].str.strip().str.lower() == dep_lower) &
        (df_stations[arrival_col].str.strip().str.lower() == arr_lower)
    ]
    if not matching_rows.empty:
        ticket["mileage"] = matching_rows.iloc[0]["mileage"]
    else:
        ticket["mileage"] = None

    return ticket
