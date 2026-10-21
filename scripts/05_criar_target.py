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
from numpy.typing import NDArray
from rasterio.crs import CRS
from rasterio.plot import plotting_extent


# 1. Configurações
PROJECT_ID = "aerobic-library-511720-u9"

MAPBIOMAS_ASSET = (
    "projects/mapbiomas-public/assets/brazil/lulc/"
    "collection11/mapbiomas_brazil_collection11_coverage_v3"
)

ANO_BASE = 2020
ANO_FINAL = 2025

CLASSE_URBANA = 24
SEM_DADOS = 255

RESOLUCAO = 30
CRS_SAIDA = "EPSG:31982"


# 2. Caminhos
RAIZ = Path(__file__).resolve().parent.parent

LIMITE = (
    RAIZ / "data" / "boundaries" / "ararangua_2025.geojson"
)

PASTA_RAW = RAIZ / "data" / "raw" / "mapbiomas"
PASTA_PROCESSED = RAIZ / "data" / "processed"
PASTA_MAPAS = RAIZ / "outputs" / "maps"

ARQUIVO_ORIGEM = PASTA_RAW / "uso_2020_2025.tif"
ARQUIVO_TARGET = PASTA_PROCESSED / "target_2020_2025.tif"
ARQUIVO_MAPA = PASTA_MAPAS / "target_2020_2025.png"


# 3. Carregar o limite municipal
def carregar_municipio() -> ee.Geometry:
    """Lê o GeoJSON local e retorna a geometria no Earth Engine."""
    with open(LIMITE, encoding="utf-8") as arquivo:
        geojson = json.load(arquivo)

    geometria = geojson["features"][0]["geometry"]

    return ee.Geometry(geometria)


# 4. Baixar as classificações dos dois anos
def baixar_classificacoes(
    mapbiomas: ee.Image,
    municipio: ee.Geometry,
) -> None:
    """Salva o GeoTIFF de duas bandas, ou reutiliza o arquivo existente."""
    if ARQUIVO_ORIGEM.exists():
        print("Classificações já baixadas. Reutilizando.")
        return

    print("Preparando classificações de 2020 e 2025...")

    uso_2020 = (
        mapbiomas
        .select("classification_2020")
        .rename("uso_2020")
    )

    uso_2025 = (
        mapbiomas
        .select("classification_2025")
        .rename("uso_2025")
    )

    imagem = (
        uso_2020
        .addBands(uso_2025)
        .clip(municipio)
        .unmask(SEM_DADOS, sameFootprint=False)
        .toUint8()
    )

    url = imagem.getDownloadURL({
        "region": municipio.bounds(maxError=1),
        "scale": RESOLUCAO,
        "crs": CRS_SAIDA,
        "format": "GEO_TIFF",
    })

    print("Baixando as duas bandas...")

    resposta = requests.get(url, timeout=180)
    resposta.raise_for_status()

    temporario = ARQUIVO_ORIGEM.with_suffix(".tmp")
    temporario.write_bytes(resposta.content)
    temporario.replace(ARQUIVO_ORIGEM)

    print("Classificações baixadas!")


# 5. Criar o mapa de conferência
def gerar_mapa(
    uso_base: NDArray[np.uint8],
    validos: NDArray[np.bool_],
    target: NDArray[np.uint8],
    extensao: tuple[float, float, float, float],
    crs: CRS,
) -> None:
    """Salva o mapa de conferência; os três arrays devem ser 2D e alinhados.

    A extensão segue a ordem (esquerda, direita, inferior, superior).
    """
    mapa = np.full(
        target.shape,
        np.nan,
        dtype=np.float32,
    )

    mapa[validos] = 0

    mapa[
        validos & (uso_base == CLASSE_URBANA)
    ] = 1

    mapa[target == 1] = 2

    cores = [
        "#E7EAE7",
        "#73483E",
        "#EF8354",
    ]

    fig, ax = plt.subplots(figsize=(9, 9))

    ax.imshow(
        np.ma.masked_invalid(mapa),
        extent=extensao,
        origin="upper",
        cmap=ListedColormap(cores),
        vmin=-0.5,
        vmax=2.5,
        interpolation="nearest",
    )

    limite = gpd.read_file(LIMITE).to_crs(crs)

    limite.boundary.plot(
        ax=ax,
        color="#25313D",
        linewidth=1,
    )

    legenda = [
        Patch(
            facecolor=cores[0],
            label="Não urbanizou entre 2020 e 2025",
        ),
        Patch(
            facecolor=cores[1],
            label="Já era urbano em 2020",
        ),
        Patch(
            facecolor=cores[2],
            label="Urbanizou até 2025 (target = 1)",
        ),
    ]

    ax.legend(
        handles=legenda,
        loc="lower left",
        fontsize=9,
    )

    ax.set_title(
        "Araranguá (SC) — Target de Machine Learning\n"
        "Urbanização de 2020 a 2025"
    )

    ax.set_aspect("equal")
    ax.set_axis_off()

    fig.savefig(
        ARQUIVO_MAPA,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Mapa salvo em: {ARQUIVO_MAPA}")


# 6. Construir e validar os rótulos
def construir_target() -> None:
    """Lê as classificações, constrói os rótulos e salva o target e o mapa."""
    with rasterio.open(ARQUIVO_ORIGEM) as raster:
        if raster.count != 2:
            raise ValueError(
                "Esperávamos duas bandas no GeoTIFF."
            )

        if raster.crs != CRS.from_string(CRS_SAIDA):
            raise ValueError("CRS do raster diferente do esperado.")

        uso_2020 = raster.read(1)
        uso_2025 = raster.read(2)

        perfil = raster.profile.copy()
        extensao = plotting_extent(raster)
        crs = raster.crs

        transformacao = raster.transform

        area_pixel_m2 = abs(
            transformacao.a * transformacao.e
            - transformacao.b * transformacao.d
        )

    # Pixels com informações válidas nos dois anos
    validos = (
        (uso_2020 != SEM_DADOS)
        & (uso_2025 != SEM_DADOS)
        & (uso_2020 != 0)
        & (uso_2025 != 0)
    )

    # Pixels que ainda não eram urbanos em 2020
    candidatos = (
        validos
        & (uso_2020 != CLASSE_URBANA)
    )

    # Inicializar todos os pixels como excluídos
    target = np.full(
        uso_2020.shape,
        SEM_DADOS,
        dtype=np.uint8,
    )

    # Atribuir 1 se tornou-se urbano; caso contrário 0
    target[candidatos] = (
        uso_2025[candidatos] == CLASSE_URBANA
    ).astype(np.uint8)

    # Contagens para validar a distribuição
    positivos = int(np.count_nonzero(target == 1))
    negativos = int(np.count_nonzero(target == 0))

    total_candidatos = positivos + negativos

    if total_candidatos == 0:
        raise ValueError(
            "Nenhum pixel elegível foi encontrado."
        )

    urbanos_2020 = int(np.count_nonzero(
        validos & (uso_2020 == CLASSE_URBANA)
    ))

    taxa_positivos = (
        positivos / total_candidatos
    ) * 100

    area_convertida_km2 = (
        positivos * area_pixel_m2 / 1_000_000
    )

    # Salvar a variável-alvo como GeoTIFF
    perfil.update(
        count=1,
        dtype="uint8",
        nodata=SEM_DADOS,
        compress="lzw",
    )

    with rasterio.open(
        ARQUIVO_TARGET,
        "w",
        **perfil,
    ) as destino:
        destino.write(target, 1)
        destino.set_band_description(
            1,
            "urbanizou_2020_2025",
        )

    print("\nDISTRIBUIÇÃO DO TARGET")
    print(f"Pixels elegíveis: {total_candidatos:,}")
    print(f"Target 0: {negativos:,}")
    print(f"Target 1: {positivos:,}")
    print(f"Já urbanos em 2020 (excluídos): {urbanos_2020:,}")
    print(f"Taxa de positivos: {taxa_positivos:.3f}%")
    print(
        f"Área dos pixels convertidos: "
        f"{area_convertida_km2:.2f} km²"
    )
    print(f"Target salvo em: {ARQUIVO_TARGET}")

    gerar_mapa(
        uso_2020,
        validos,
        target,
        extensao,
        crs,
    )


# 7. Fluxo principal
def main() -> None:
    """Executa a etapa de download e construção da variável-alvo."""
    PASTA_RAW.mkdir(parents=True, exist_ok=True)
    PASTA_PROCESSED.mkdir(parents=True, exist_ok=True)
    PASTA_MAPAS.mkdir(parents=True, exist_ok=True)

    print("Conectando ao Google Earth Engine...")

    ee.Initialize(project=PROJECT_ID)

    municipio = carregar_municipio()
    mapbiomas = ee.Image(MAPBIOMAS_ASSET)

    baixar_classificacoes(mapbiomas, municipio)
    construir_target()

    print("\nEtapa 5 concluída!")


if __name__ == "__main__":
    main()
