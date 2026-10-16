
import json
from pathlib import Path
from zipfile import ZipFile

import ee
import geopandas as gpd
import requests


# 1. Configurações do projeto
PROJECT_ID = "aerobic-library-511720-u9"

CODIGO_MUNICIPIO = "4201406"

URL_IBGE = (
    "https://geoftp.ibge.gov.br/"
    "organizacao_do_territorio/malhas_territoriais/"
    "malhas_municipais/municipio_2025/UFs/SC/"
    "SC_Municipios_2025.zip"
)


# 2. Caminhos dos arquivos
RAIZ = Path(__file__).resolve().parent.parent

PASTA_IBGE = RAIZ / "data" / "raw" / "ibge"
PASTA_LIMITES = RAIZ / "data" / "boundaries"

ARQUIVO_ZIP = PASTA_IBGE / "SC_Municipios_2025.zip"
PASTA_EXTRAIDA = PASTA_IBGE / "SC_Municipios_2025"

ARQUIVO_GEOJSON = (
    PASTA_LIMITES / "ararangua_2025.geojson"
)


def main():

    # 3. Criar diretórios
    PASTA_IBGE.mkdir(parents=True, exist_ok=True)
    PASTA_LIMITES.mkdir(parents=True, exist_ok=True)

    # 4. Baixar os dados oficiais do IBGE
    if not ARQUIVO_ZIP.exists():

        print("Baixando malha municipal do IBGE...")

        resposta = requests.get(URL_IBGE, timeout=60)
        resposta.raise_for_status()

        ARQUIVO_ZIP.write_bytes(resposta.content)

    print("Arquivo do IBGE disponível.")

    # 5. Extrair o arquivo ZIP
    if not PASTA_EXTRAIDA.exists():

        with ZipFile(ARQUIVO_ZIP) as arquivo:
            arquivo.extractall(PASTA_EXTRAIDA)

    # 6. Encontrar o Shapefile
    shapefiles = list(
        PASTA_EXTRAIDA.rglob("*.shp")
    )

    if len(shapefiles) != 1:
        raise ValueError(
            "Esperávamos encontrar um único Shapefile."
        )

    # 7. Ler os municípios de Santa Catarina
    municipios = gpd.read_file(shapefiles[0])

    print(f"Municípios carregados: {len(municipios)}")

    # 8. Filtrar Araranguá
    ararangua = municipios[
        municipios["CD_MUN"].astype(str) == CODIGO_MUNICIPIO
    ].copy()

    if len(ararangua) != 1:
        raise ValueError(
            "Araranguá não foi identificado corretamente."
        )

    if ararangua.crs is None:
        raise ValueError("Sistema de coordenadas ausente.")

    if not ararangua.geometry.is_valid.all():
        raise ValueError("Geometria municipal inválida.")

    print("Araranguá encontrado!")

    # 9. Converter para coordenadas geográficas WGS84
    ararangua = ararangua.to_crs(epsg=4326)

    # 10. Salvar o GeoJSON
    ararangua.to_file(
        ARQUIVO_GEOJSON,
        driver="GeoJSON"
    )

    print(f"GeoJSON salvo em: {ARQUIVO_GEOJSON}")

    # 11. Carregar a geometria no Earth Engine
    with open(ARQUIVO_GEOJSON, encoding="utf-8") as arquivo:
        geojson = json.load(arquivo)

    geometria = geojson["features"][0]["geometry"]

    ee.Initialize(project=PROJECT_ID)

    municipio_ee = ee.Geometry(geometria)

    # 12. Calcular a área do município
    area_m2 = municipio_ee.area(maxError=1).getInfo()
    area_km2 = area_m2 / 1_000_000

    print(f"Área calculada: {area_km2:.2f} km²")
    print("Limite municipal validado no Earth Engine!")
    print("\nEtapa 2 concluída!")


if __name__ == "__main__":
    main()
