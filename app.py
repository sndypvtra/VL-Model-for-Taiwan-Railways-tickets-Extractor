# app.py
import os
import uvicorn
import json
import pandas as pd
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

# Pastikan folder temp_uploads sudah ada sebelum dipasang sebagai static files
os.makedirs("temp_uploads", exist_ok=True)

# Fungsi inference
from single_inference import process_single_ticket
from multi_inference import process_multi_ticket

# Muat CSV stasiun untuk validasi
STATIONS_CSV = "stations_database_modified.csv"
df_stations = pd.read_csv(STATIONS_CSV)

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/temp_uploads", StaticFiles(directory="temp_uploads"), name="temp_uploads")


def is_valid_ticket(ticket: dict) -> bool:
    """
    Memeriksa apakah ticket memiliki informasi yang lengkap dan valid.
    Validasi:
      - departure_station harus cocok dengan salah satu nilai di startStaName atau startStaEName.
      - arrival_station harus cocok dengan salah satu nilai di endStaName atau endStaEName.
      - Juga harus memiliki nilai untuk date.
    """
    if not ticket or not isinstance(ticket, dict):
        return False

    dep = ticket.get("departure_station", "").strip()
    arr = ticket.get("arrival_station", "").strip()
    date = ticket.get("date", "").strip()

    if not dep or not arr or not date:
        return False

    dep_lower = dep.lower()
    arr_lower = arr.lower()

    # Validasi departure_station dari dua kolom: startStaName atau startStaEName
    dep_valid = (
        df_stations["startStaName"].str.strip().str.lower().eq(dep_lower).any() or
        df_stations["startStaEName"].str.strip().str.lower().eq(dep_lower).any()
    )
    # Validasi arrival_station dari dua kolom: endStaName atau endStaEName
    arr_valid = (
        df_stations["endStaName"].str.strip().str.lower().eq(arr_lower).any() or
        df_stations["endStaEName"].str.strip().str.lower().eq(arr_lower).any()
    )

    return dep_valid and arr_valid


def cleanup_stations_inplace(ticket: dict):
    """
    (Opsional) Membersihkan stasiun palsu.
    Misalnya, jika stasiun mengandung 'Exp', kosongkan nilainya.
    """
    if not ticket or not isinstance(ticket, dict):
        return

    dep = ticket.get("departure_station", "")
    arr = ticket.get("arrival_station", "")

    if "Exp" in dep:
        ticket["departure_station"] = ""
    if "Exp" in arr:
        ticket["arrival_station"] = ""


@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})


@app.post("/upload/")
async def upload_file(request: Request, file: UploadFile = File(...)):
    """
    1) Terima file dan simpan.
    2) Jalankan SINGLE inference dan MULTI inference secara independen.
    3) Tentukan hasil final:
         - Jika multi_inference menghasilkan > 1 tiket, gunakan multi_data.
         - Jika hanya 1 tiket dan ticket valid (berdasarkan CSV), gunakan single_data.
         - Jika ticket tidak valid (departure atau arrival tidak match CSV), paksa gunakan hasil single_inference.
    4) (Opsional) Bersihkan stasiun palsu.
    5) Simpan JSON tanpa duplikasi.
    6) Tampilkan di output.html.
    """
    # 1) Simpan file upload
    upload_dir = "temp_uploads"
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # 2) SINGLE inference
    parsed_single = process_single_ticket(file_path)
    if isinstance(parsed_single, dict):
        single_data = [parsed_single]
    else:
        single_data = parsed_single

    # 3) MULTI inference
    parsed_multi = process_multi_ticket(file_path)
    if isinstance(parsed_multi, dict):
        multi_data = [parsed_multi]
    else:
        multi_data = parsed_multi

    # 4) Tentukan hasil final
    if len(multi_data) > 1:
        final_data = multi_data
        print("Menggunakan hasil dari multi_inference (lebih dari satu tiket terdeteksi).")
    else:
        if len(single_data) == 1 and is_valid_ticket(single_data[0]):
            final_data = single_data
            print("Menggunakan hasil dari single_inference (ticket valid).")
        else:
            final_data = multi_data
            print("Menggunakan hasil dari multi_inference sebagai fallback (single_inference tidak valid).")

    # 5) Jika salah satu dari departure_station atau arrival_station tidak match CSV,
    #    paksa gunakan hasil single_inference.
    if final_data:
        ticket0 = final_data[0]
        if not is_valid_ticket(ticket0):
            print("Ticket tidak valid menurut CSV, memaksa penggunaan hasil single_inference.")
            forced_single = process_single_ticket(file_path)
            if isinstance(forced_single, dict):
                final_data = [forced_single]
            else:
                final_data = forced_single

    # 6) (Opsional) Bersihkan stasiun palsu
    for t in final_data:
        cleanup_stations_inplace(t)

    # 7) Simpan hasil ke JSON tanpa duplikasi
    if isinstance(final_data, dict):
        final_data = [final_data]
    output_dir = "json_outputs"
    os.makedirs(output_dir, exist_ok=True)
    data_file = os.path.join(output_dir, "tickets_data.json")
    if os.path.exists(data_file):
        with open(data_file, "r", encoding="utf-8") as f:
            try:
                existing_data = json.load(f)
            except json.JSONDecodeError:
                existing_data = []
    else:
        existing_data = []
    for ticket in final_data:
        if ticket not in existing_data:
            existing_data.append(ticket)
    with open(data_file, "w", encoding="utf-8") as f:
        json.dump(existing_data, f, ensure_ascii=False, indent=2)

    # 8) Tampilkan di output.html
    image_url = f"/temp_uploads/{file.filename}"
    return templates.TemplateResponse("output.html", {
        "request": request,
        "tickets": final_data,
        "image_url": image_url
    })


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
