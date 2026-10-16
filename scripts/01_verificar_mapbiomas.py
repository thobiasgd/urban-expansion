import ee

# Configurações
PROJECT_ID = "aerobic-library-511720-u9"

MAPBIOMAS_ASSET = (
    "projects/mapbiomas-public/assets/brazil/lulc/"
    "collection11/mapbiomas_brazil_collection11_coverage_v3"
)

ANO_INICIAL = 1985
ANO_FINAL = 2025
CLASSE_URBANA = 24

def main():

    # 1. Autenticar e conectar ao Earth Engine
    print("Conectando ao Google Earth Engine")

    ee.Authenticate()
    ee.Initialize(project=PROJECT_ID)

    print("Conexão estabelecida!")

    # 2. Carregar a imagem do MapBiomas
    print("\nCarregando MapBiomas")

    imagem = ee.Image(MAPBIOMAS_ASSET)

    # 3. Consultar as bandas disponíveis
    bandas = imagem.bandNames(). getInfo()

    print(f"Total de bandas encontradas: {len(bandas)}")

    # 4. Verificar a série histórica
    anos = range(ANO_INICIAL, ANO_FINAL + 1)

    bandas_esperadas = [
        f"classification_{ano}" for ano in anos
    ]

    bandas_faltantes = [
        banda for banda in bandas_esperadas
        if banda not in bandas
    ]

    if bandas_faltantes:
        raise ValueError(
            f"Bandas ausentes: {bandas_faltantes}"
        )

    print("Série histórica de 1985 a 2025 confirmada")

    # 5. Selecionar a classificação de 2025
    mapa_2025 = imagem.select("classification_2025")

    # 6. Criar a máscara de áreas urbanizadas
    urbano_2025 = (
        mapa_2025
        .eq(CLASSE_URBANA)
        .rename("urbano_2025")
    )

    print(
        "Máscara urbana criada:",
        urbano_2025.bandNames().getInfo()
    )

    print("\nPrimeira etapa concluída!")

if __name__ == "__main__":
    main()