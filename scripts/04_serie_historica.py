
import csv
import json
from pathlib import Path

import ee
import matplotlib.pyplot as plt


# 1. Configurações
PROJECT_ID = "aerobic-library-511720-u9"

MAPBIOMAS_ASSET = (
    "projects/mapbiomas-public/assets/brazil/lulc/"
    "collection11/mapbiomas_brazil_collection11_coverage_v3"
)

ANO_INICIAL = 1985
ANO_FINAL = 2025
CLASSE_URBANA = 24
RESOLUCAO = 30
CRS_SAIDA = "EPSG:31982"


# 2. Caminhos
RAIZ = Path(__file__).resolve().parent.parent

LIMITE = (
    RAIZ / "data" / "boundaries" / "ararangua_2025.geojson"
)

PASTA_DADOS = RAIZ / "data" / "processed"
PASTA_MAPAS = RAIZ / "outputs" / "maps"

ARQUIVO_CSV = (
    PASTA_DADOS / "serie_historica_urbanizacao.csv"
)

ARQUIVO_GRAFICO = (
    PASTA_MAPAS / "serie_historica_urbanizacao.png"
)


# 3. Carregar o limite municipal
def carregar_municipio():

    with open(LIMITE, encoding="utf-8") as arquivo:
        geojson = json.load(arquivo)

    geometria = geojson["features"][0]["geometry"]

    return ee.Geometry(geometria)


# 4. Construir as bandas de área urbana
def construir_imagem_areas(mapbiomas):

    bandas = []

    area_pixel = ee.Image.pixelArea()

    for ano in range(ANO_INICIAL, ANO_FINAL + 1):

        urbano = (
            mapbiomas
            .select(f"classification_{ano}")
            .eq(CLASSE_URBANA)
        )

        area_urbana = (
            area_pixel
            .updateMask(urbano)
            .rename(f"area_{ano}")
        )

        bandas.append(area_urbana)

    return ee.Image.cat(bandas)


# 5. Calcular a série temporal
def calcular_serie(imagem_areas, municipio):

    print("Calculando série histórica no Earth Engine...")

    resultado = imagem_areas.reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=municipio,
        scale=RESOLUCAO,
        crs=CRS_SAIDA,
        maxPixels=100_000_000,
        tileScale=4
    ).getInfo()

    registros = []
    area_anterior = None

    for ano in range(ANO_INICIAL, ANO_FINAL + 1):

        chave = f"area_{ano}"

        area_m2 = resultado.get(chave)

        if area_m2 is None:
            raise ValueError(
                f"Área urbana não calculada para {ano}."
            )

        area_km2 = area_m2 / 1_000_000

        variacao = None

        if area_anterior is not None:
            variacao = area_km2 - area_anterior

        registros.append({
            "ano": ano,
            "area_urbana_km2": area_km2,
            "variacao_anual_km2": variacao
        })

        area_anterior = area_km2

    return registros


# 6. Salvar CSV
def salvar_csv(registros):

    colunas = [
        "ano",
        "area_urbana_km2",
        "variacao_anual_km2"
    ]

    with open(
        ARQUIVO_CSV,
        "w",
        newline="",
        encoding="utf-8"
    ) as arquivo:

        escritor = csv.DictWriter(
            arquivo,
            fieldnames=colunas
        )

        escritor.writeheader()
        escritor.writerows(registros)

    print(f"CSV salvo em: {ARQUIVO_CSV}")


# 7. Gerar gráfico
def gerar_grafico(registros):

    anos = [r["ano"] for r in registros]

    areas = [
        r["area_urbana_km2"]
        for r in registros
    ]

    fig, ax = plt.subplots(figsize=(11, 5))

    ax.plot(
        anos,
        areas,
        linewidth=2,
        color="#D97745"
    )

    ax.scatter(
        [anos[0], anos[-1]],
        [areas[0], areas[-1]],
        color="#73483E",
        zorder=3
    )

    for indice in [0, -1]:

        ax.annotate(
            f"{areas[indice]:.2f} km²",
            xy=(anos[indice], areas[indice]),
            xytext=(0, 12),
            textcoords="offset points",
            ha="center"
        )

    ax.set_title(
        "Araranguá (SC) — Evolução da área urbana\n"
        "1985–2025"
    )

    ax.set_xlabel("Ano")
    ax.set_ylabel("Área urbana (km²)")

    ax.set_xlim(ANO_INICIAL - 1, ANO_FINAL + 1)
    ax.grid(alpha=0.25)

    fig.tight_layout()

    fig.savefig(
        ARQUIVO_GRAFICO,
        dpi=220,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"Gráfico salvo em: {ARQUIVO_GRAFICO}")


# 8. Fluxo principal
def main():

    PASTA_DADOS.mkdir(parents=True, exist_ok=True)
    PASTA_MAPAS.mkdir(parents=True, exist_ok=True)

    print("Conectando ao Earth Engine...")

    ee.Initialize(project=PROJECT_ID)

    municipio = carregar_municipio()

    mapbiomas = ee.Image(MAPBIOMAS_ASSET)

    imagem_areas = construir_imagem_areas(mapbiomas)

    registros = calcular_serie(
        imagem_areas,
        municipio
    )

    salvar_csv(registros)
    gerar_grafico(registros)

    # Resumo dos resultados
    primeira_area = registros[0]["area_urbana_km2"]
    ultima_area = registros[-1]["area_urbana_km2"]

    crescimento_km2 = ultima_area - primeira_area

    crescimento_percentual = (
        crescimento_km2 / primeira_area
    ) * 100 if primeira_area > 0 else None

    print("\nRESUMO DA SÉRIE HISTÓRICA")

    print(f"Anos analisados: {len(registros)}")

    print(f"Área em {ANO_INICIAL}: {primeira_area:.2f} km²")
    print(f"Área em {ANO_FINAL}: {ultima_area:.2f} km²")

    print(f"Crescimento líquido: {crescimento_km2:+.2f} km²")

    if crescimento_percentual is not None:
        print(
            f"Crescimento relativo: "
            f"{crescimento_percentual:+.2f}%"
        )

    # Identificar os anos com maior e menor variação
    variacoes = registros[1:]

    maior_aumento = max(
        variacoes,
        key=lambda r: r["variacao_anual_km2"]
    )

    menor_variacao = min(
        variacoes,
        key=lambda r: r["variacao_anual_km2"]
    )

    print(
        f"Maior aumento anual: {maior_aumento['ano']} "
        f"({maior_aumento['variacao_anual_km2']:+.2f} km²)"
    )

    print(
        f"Menor variação anual: {menor_variacao['ano']} "
        f"({menor_variacao['variacao_anual_km2']:+.2f} km²)"
    )

    print("\nEtapa 4 concluída!")


if __name__ == "__main__":
    main()
