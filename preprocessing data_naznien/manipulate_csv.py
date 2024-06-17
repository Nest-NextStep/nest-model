import pandas as pd

# Membaca file spreadsheet
df = pd.read_excel("your_file.xlsx")  # Ganti "nama_file.xlsx" dengan nama file spreadsheet Anda

# Menerapkan fungsi untuk mengubah nilai kolom A berdasarkan nilai di kolom BM
def change_value(row):
    if "Political Science" in str(row['major_final']):  # Memeriksa apakah "medicine" ada di nilai kolom BM
        return "Ilmu Politik"
    else:
        return row['major']  # Mempertahankan nilai kolom A jika tidak ditemukan "medicine" di kolom BM

# Mengaplikasikan fungsi ke setiap baris di dataframe
df['major'] = df.apply(change_value, axis=1)

