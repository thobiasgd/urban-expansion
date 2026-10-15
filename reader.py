import pandas as pd

path = r"dados\INPE 2024\INMET_S_SC_A867_ARARANGUA_01-01-2024_A_31-12-2024.CSV"

df = pd.read_csv(
    path,
    skiprows=8,
    sep=";",
    decimal=",",
    encoding="latin-1",
    usecols=[
        "Data",
        "Hora UTC",
        "PRECIPITAÇÃO TOTAL, HORÁRIO (mm)"
    ],
)

df["data_hora"] = pd.to_datetime(
    df["Data"] + " " + df["Hora UTC"],
    format="%Y/%m/%d %H%M UTC",
    utc=True,
)

horarios = df["data_hora"].sort_values()

intervalos = horarios.diff().dropna()

chuva = df["PRECIPITAÇÃO TOTAL, HORÁRIO (mm)"]

print("Resumo da precipitação horária, em mm:")

print("\nValores ausentes:", chuva.isna().sum())
print("Valores negativos:", (chuva < 0 ).sum())
print("Registros iguais a zero:", (chuva == 0).sum())
print("Registros maiores que zero:", (chuva > 0).sum())