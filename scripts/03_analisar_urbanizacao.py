
import csv
import json
from pathlib import Path

import ee
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import requests

from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from rasterio.plot import plotting_extent


# 1. Configurações
PROJECT_ID = "aerobic-library-511720-u9"

MAPBIOMAS_ASSET = (
    "projects/mapbiomas-public/assets/brazil/lulc/"
    "collection11/mapbiomas_brazil_collection11_coverage_v3"
)

ANOS = [1985, 2025]
CLASSE_URBANA = 24
RESOLUCAO = 30
CRS_SAIDA = "EPSG:31982"


# 2. Caminhos dos arquivos
RAIZ = Path(__file__).resolve().parent.parent

LIMITE = (
    RAIZ / "data" / "boundaries" / "ararangua_2025.geojson"
)

PASTA_DADOS = RAIZ / "data" / "processed"
PASTA_MAPAS = RAIZ / "outputs" / "maps"

ARQUIVO_TIF = PASTA_DADOS / "urbano_1985_2025.tif"
ARQUIVO_CSV = PASTA_DADOS / "areas_urbanas.csv"
ARQUIVO_PNG = PASTA_MAPAS / "expansao_urbana.png"


# 3. Calcular a área urbana em km²
def calcular_area_urbana(imagem_urbana, municipio):

    imagem_area = (
        ee.Image.pixelArea()
        .updateMask(imagem_urbana)
    )

    resultado = imagem_area.reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=municipio,
        scale=RESOLUCAO,
        crs=CRS_SAIDA,
        maxPixels=10_000_000,
        tileScale=4
    )

    area_m2 = resultado.get("area").getInfo()

    return (area_m2 or 0) / 1_000_000


# 4. Baixar o GeoTIFF com os dois anos
def baixar_raster(imagem, municipio):

    if ARQUIVO_TIF.exists():
        print("GeoTIFF já existe. Reutilizando arquivo.")
        return

    imagem_exportacao = (
        imagem
        .clip(municipio)
        .unmask(255, sameFootprint=False)
        .toUint8()
    )

    url = imagem_exportacao.getDownloadURL({
        "region": municipio.bounds(maxError=1),
        "scale": RESOLUCAO,
        "crs": CRS_SAIDA,
        "format": "GEO_TIFF"
    })

    print("Baixando GeoTIFF...")

    resposta = requests.get(url, timeout=180)
    resposta.raise_for_status()

    ARQUIVO_TIF.write_bytes(resposta.content)

    print("GeoTIFF salvo com sucesso!")


# 5. Gerar o mapa comparativo
def gerar_mapa():

    with rasterio.open(ARQUIVO_TIF) as raster:

        urbano_1985 = raster.read(1)
        urbano_2025 = raster.read(2)

        extensao = plotting_extent(raster)
        crs_raster = raster.crs

    valido = (
        np.isin(urbano_1985, [0, 1])
        & np.isin(urbano_2025, [0, 1])
    )

    classes = np.full(
        urbano_1985.shape,
        np.nan,
        dtype=np.float32
    )

    classes[valido] = 0

    classes[valido & (urbano_1985 == 1) & (urbano_2025 == 1)] = 1
    classes[valido & (urbano_1985 == 0) & (urbano_2025 == 1)] = 2
    classes[valido & (urbano_1985 == 1) & (urbano_2025 == 0)] = 3

    if not valido.any():
        raise ValueError("Nenhum pixel válido encontrado.")

    cores = [
        "#E7EAE7",
        "#73483E",
        "#EF8354",
        "#5186B5"
    ]

    fig, ax = plt.subplots(figsize=(9, 9))

    ax.imshow(
        np.ma.masked_invalid(classes),
        extent=extensao,
        origin="upper",
        cmap=ListedColormap(cores),
        vmin=-0.5,
        vmax=3.5,
        interpolation="nearest"
    )

    limite = gpd.read_file(LIMITE).to_crs(crs_raster)

    limite.boundary.plot(
        ax=ax,
        color="#25313D",
        linewidth=1
    )

    legenda = [
        Patch(facecolor=cores[0], label="Não urbano nos dois anos"),
        Patch(facecolor=cores[1], label="Urbano em 1985 e 2025"),
        Patch(facecolor=cores[2], label="Tornou-se urbano até 2025"),
        Patch(facecolor=cores[3], label="Deixou de ser urbano até 2025")
    ]

    ax.legend(handles=legenda, loc="lower left")

    ax.set_title(
        "Araranguá (SC) — Mudança da área urbana\n1985–2025",
        fontsize=14
    )

    ax.set_aspect("equal")
    ax.set_axis_off()

    fig.savefig(
        ARQUIVO_PNG,
        dpi=220,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"Mapa salvo em: {ARQUIVO_PNG}")


# 6. Fluxo principal

def main():

    # 1. Criar os diretórios necessários
    PASTA_DADOS.mkdir(parents=True, exist_ok=True)
    PASTA_MAPAS.mkdir(parents=True, exist_ok=True)

    # 2. Inicializar o Google Earth Engine
    print("Conectando ao Google Earth Engine...")

    ee.Initialize(project=PROJECT_ID)

    print("Conexão estabelecida!")

    # 3. Carregar o GeoJSON de Araranguá
    with open(LIMITE, encoding="utf-8") as arquivo:
        geojson = json.load(arquivo)

    # 4. Converter a geometria para Earth Engine
    municipio = ee.Geometry(
        geojson["features"][0]["geometry"]
    )

    # 5. Carregar o MapBiomas
    mapbiomas = ee.Image(MAPBIOMAS_ASSET)

    # 6. Preparar listas para armazenar os resultados
    imagens_urbanas = []
    areas = []

    # 7. Processar cada ano
    for ano in ANOS:

        print(f"\nProcessando {ano}...")

        # Selecionar a classificação do ano e
        # identificar os pixels classificados como urbanos
        urbano = (
            mapbiomas
            .select(f"classification_{ano}")
            .eq(CLASSE_URBANA)
            .rename(f"urbano_{ano}")
        )

        # Calcular a área urbana em km²
        area = calcular_area_urbana(
            urbano,
            municipio
        )

        print(f"Área urbana em {ano}: {area:.2f} km²")

        # Armazenar os resultados
        imagens_urbanas.append(urbano)
        areas.append([ano, area])

    # 8. Salvar as estatísticas em CSV
    with open(
        ARQUIVO_CSV,
        "w",
        newline="",
        encoding="utf-8"
    ) as arquivo:

        escritor = csv.writer(arquivo)

        escritor.writerow([
            "ano",
            "area_urbana_km2"
        ])

        escritor.writerows(areas)

    print(f"\nEstatísticas salvas em: {ARQUIVO_CSV}")

    # 9. Calcular a variação líquida da área urbana
    variacao = areas[1][1] - areas[0][1]

    print(
        f"\nVariação líquida da área urbana: "
        f"{variacao:+.2f} km²"
    )

    # 10. Combinar as imagens de 1985 e 2025
    imagem_combinada = imagens_urbanas[0].addBands(
        imagens_urbanas[1]
    )

    # 11. Baixar o raster para o computador
    baixar_raster(
        imagem_combinada,
        municipio
    )

    # 12. Gerar o mapa da expansão urbana
    gerar_mapa()

    print("\nEtapa 3 concluída!")


if __name__ == "__main__":
    main()
