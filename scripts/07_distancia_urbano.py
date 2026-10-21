from dataclasses import dataclass
from math import ceil
from pathlib import Path

import ee
import numpy as np
import rasterio
import requests
from affine import Affine
from numpy.typing import NDArray
from rasterio.crs import CRS
from scipy.ndimage import distance_transform_edt


# 1. Configurações e caminhos
PROJECT_ID: str = "aerobic-library-511720-u9"
MAPBIOMAS_ASSET: str = (
    "projects/mapbiomas-public/assets/brazil/lulc/"
    "collection11/mapbiomas_brazil_collection11_coverage_v3"
)

ANO_BASE: int = 2020
CLASSE_URBANA: int = 24
SEM_DADOS_USO: int = 255
SEM_DADOS_DISTANCIA: float = -9999.0
RESOLUCAO_M: float = 30.0
LIMITE_DISTANCIA_M: float = 3000.0
MARGEM_PIXELS: int = ceil(LIMITE_DISTANCIA_M / RESOLUCAO_M) + 1

CODIGOS_COBERTURA: tuple[int, ...] = (
    1, 3, 4, 5, 6, 7, 9, 10, 11, 12, 14, 15,
    18, 19, 20, 21, 22, 23, 24, 25, 26, 29, 30, 31,
    32, 33, 35, 36, 39, 40, 41, 46, 47, 48, 49, 50,
    62, 75, 77, 84, 91,
)

RAIZ: Path = Path(__file__).resolve().parent.parent

# Neste arquivo de duas bandas, leremos SOMENTE a banda 1: 2020.
ARQUIVO_BASE: Path = (
    RAIZ / "data" / "raw" / "mapbiomas" / "uso_2020_2025.tif"
)
ARQUIVO_CONTEXTO: Path = (
    RAIZ / "data" / "raw" / "mapbiomas"
    / f"uso_{ANO_BASE}_contexto_{MARGEM_PIXELS}px.tif"
)
ARQUIVO_FEATURE: Path = (
    RAIZ / "data" / "processed" / "features"
    / f"distancia_urbano_{ANO_BASE}_lim{int(LIMITE_DISTANCIA_M)}m.tif"
)


# 2. Estrutura que reúne as informações da grade espacial
@dataclass(frozen=True)
class GradeRaster:
    linhas: int
    colunas: int
    transformacao: Affine
    crs: CRS


# 3. Ler apenas a classificação do ano-base

def ler_base(caminho: Path) -> tuple[NDArray[np.uint8], GradeRaster]:
    """Lê a banda 1 do arquivo auditado e guarda sua grade."""
    if not caminho.is_file():
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho}")

    with rasterio.open(caminho) as raster:
        if raster.count != 2 or raster.dtypes[0] != "uint8":
            raise ValueError("Esperávamos o arquivo de duas bandas do script 05.")

        if raster.crs != CRS.from_epsg(31982):
            raise ValueError("CRS diferente de EPSG:31982.")

        t = raster.transform
        if not np.allclose(
            (t.a, t.b, t.d, t.e),
            (RESOLUCAO_M, 0.0, 0.0, -RESOLUCAO_M),
            rtol=0,
            atol=1e-9,
        ):
            raise ValueError("Esperávamos pixels de 30 m, sem rotação.")

        uso: NDArray[np.uint8] = raster.read(1)
        uso[raster.read_masks(1) == 0] = SEM_DADOS_USO

        grade = GradeRaster(
            raster.height, raster.width, t, raster.crs
        )

    return uso, grade


# 4. Expandir a grade sem mudar o alinhamento dos pixels

def ampliar_grade(grade: GradeRaster, margem: int) -> GradeRaster:
    """Adiciona uma margem, em pixels, nos quatro lados."""
    return GradeRaster(
        linhas=grade.linhas + 2 * margem,
        colunas=grade.colunas + 2 * margem,
        transformacao=(
            grade.transformacao * Affine.translation(-margem, -margem)
        ),
        crs=grade.crs,
    )


# 5. Obter a classificação de 2020 com o entorno

def carregar_contexto(
    grade: GradeRaster,
    destino: Path,
) -> NDArray[np.uint8]:
    """Baixa ou reutiliza o entorno e verifica sua grade e procedência."""
    destino.parent.mkdir(parents=True, exist_ok=True)
    temporario = destino.with_suffix(".tmp.tif")
    novo = not destino.exists()
    caminho_leitura = temporario if novo else destino

    try:
        if novo:
            print("Conectando ao Earth Engine para baixar o entorno...")
            ee.Initialize(project=PROJECT_ID)

            imagem = (
                ee.Image(MAPBIOMAS_ASSET)
                .select(f"classification_{ANO_BASE}")
                .unmask(SEM_DADOS_USO, sameFootprint=False)
                .toUint8()
            )

            t = grade.transformacao
            url = imagem.getDownloadURL({
                "crs": grade.crs.to_string(),
                "crs_transform": [t.a, t.b, t.c, t.d, t.e, t.f],
                "dimensions": [grade.colunas, grade.linhas],
                "format": "GEO_TIFF",
            })

            with requests.get(url, timeout=(30, 180)) as resposta:
                resposta.raise_for_status()
                temporario.write_bytes(resposta.content)

            with rasterio.open(temporario, "r+") as raster:
                raster.update_tags(
                    ano_base=str(ANO_BASE),
                    asset=MAPBIOMAS_ASSET,
                )
        else:
            print("Reutilizando classificação com entorno.")

        with rasterio.open(caminho_leitura) as raster:
            compativel = (
                raster.count == 1
                and raster.dtypes == ("uint8",)
                and raster.crs == grade.crs
                and raster.shape == (grade.linhas, grade.colunas)
                and np.allclose(
                    tuple(raster.transform)[:6],
                    tuple(grade.transformacao)[:6],
                    rtol=0,
                    atol=1e-6,
                )
                and raster.tags().get("ano_base") == str(ANO_BASE)
                and raster.tags().get("asset") == MAPBIOMAS_ASSET
            )
            if not compativel:
                raise ValueError(
                    f"Entorno incompatível com a configuração: {caminho_leitura}"
                )

            uso: NDArray[np.uint8] = raster.read(1)
            uso[raster.read_masks(1) == 0] = SEM_DADOS_USO

        if novo:
            temporario.replace(destino)

        return uso
    finally:
        if novo:
            temporario.unlink(missing_ok=True)


# 6. Calcular a distância e tratar lacunas de classificação

def calcular_distancia(
    uso: NDArray[np.uint8],
    resolucao_m: float,
    limite_m: float,
) -> NDArray[np.float32]:
    """Distância euclidiana truncada; lacunas potencialmente relevantes viram NoData."""
    if uso.ndim != 2 or resolucao_m <= 0 or limite_m <= 0:
        raise ValueError("Matriz 2D, resolução e limite positivos são necessários.")

    permitidos = (*CODIGOS_COBERTURA, 0, SEM_DADOS_USO)
    if not np.isin(uso, permitidos).all():
        raise ValueError("Código inesperado na classificação do entorno.")

    validos = np.isin(uso, CODIGOS_COBERTURA)
    urbanos = validos & (uso == CLASSE_URBANA)

    if urbanos.any():
        distancia = distance_transform_edt(
            ~urbanos,
            sampling=(resolucao_m, resolucao_m),
        )
        distancia = np.minimum(distancia, limite_m)
    else:
        distancia = np.full(uso.shape, limite_m, dtype=np.float64)

    confiaveis = validos.copy()

    if not validos.all():
        distancia_lacuna = distance_transform_edt(
            validos,
            sampling=(resolucao_m, resolucao_m),
        )
        confiaveis &= distancia_lacuna >= distancia

    return np.where(
        confiaveis,
        distancia,
        SEM_DADOS_DISTANCIA,
    ).astype(np.float32)


# 7. Salvar a feature na grade original

def salvar_feature(
    distancia: NDArray[np.float32],
    grade: GradeRaster,
    destino: Path,
) -> None:
    """Grava uma banda contínua com NoData explícito e metadados."""
    if distancia.shape != (grade.linhas, grade.colunas):
        raise ValueError("A feature não tem o tamanho da grade de saída.")

    destino.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(
        destino,
        "w",
        driver="GTiff",
        height=grade.linhas,
        width=grade.colunas,
        count=1,
        dtype="float32",
        crs=grade.crs,
        transform=grade.transformacao,
        nodata=SEM_DADOS_DISTANCIA,
        compress="lzw",
    ) as raster:
        raster.write(distancia, 1)
        raster.set_band_description(1, "distancia_urbano_limitada_m")
        raster.update_tags(
            ano_base=str(ANO_BASE),
            limite_m=str(LIMITE_DISTANCIA_M),
            asset=MAPBIOMAS_ASSET,
            definicao="Distancia entre centros de pixels; limitada superiormente",
        )

    print(f"\nFeature salva em: {destino}")


# 8. Fluxo principal

def main() -> None:
    print(f"FEATURE DE DISTÂNCIA — ANO-BASE {ANO_BASE}")

    uso_base, grade = ler_base(ARQUIVO_BASE)
    grade_contexto = ampliar_grade(grade, MARGEM_PIXELS)

    print(f"Grade original: {(grade.linhas, grade.colunas)}")
    print(f"Margem em cada lado: {MARGEM_PIXELS * RESOLUCAO_M:.0f} m")

    contexto = carregar_contexto(grade_contexto, ARQUIVO_CONTEXTO)

    p = MARGEM_PIXELS
    recorte = (
        slice(p, p + grade.linhas),
        slice(p, p + grade.colunas),
    )

    validos_base = np.isin(uso_base, CODIGOS_COBERTURA)
    if not validos_base.any():
        raise ValueError("Nenhuma classificação válida no ano-base.")

    divergencias = int(np.count_nonzero(
        validos_base & (contexto[recorte] != uso_base)
    ))
    print(f"Divergências na classificação de 2020: {divergencias:,}")
    if divergencias:
        raise ValueError("O entorno não reproduz a classificação já auditada.")

    distancia_contexto = calcular_distancia(
        contexto,
        RESOLUCAO_M,
        LIMITE_DISTANCIA_M,
    )
    distancia = distancia_contexto[recorte].copy()
    distancia[~validos_base] = SEM_DADOS_DISTANCIA

    urbanos_base = validos_base & (uso_base == CLASSE_URBANA)
    if not np.all(distancia[urbanos_base] == 0):
        raise ValueError("A distância deveria ser zero nos pixels urbanos.")

    candidatos = validos_base & ~urbanos_base
    disponiveis = candidatos & (distancia != SEM_DADOS_DISTANCIA)
    valores = distancia[disponiveis]

    if valores.size == 0:
        raise ValueError("Nenhum candidato possui distância utilizável.")

    salvar_feature(distancia, grade, ARQUIVO_FEATURE)

    print("\nRESUMO — APENAS PIXELS NÃO URBANOS EM 2020")
    print(f"Candidatos pelo ano-base: {np.count_nonzero(candidatos):,}")
    print(f"Com distância utilizável: {valores.size:,}")
    print(
        "Sem distância por lacunas no entorno: "
        f"{np.count_nonzero(candidatos & ~disponiveis):,}"
    )
    print(f"Distância mínima: {valores.min():.2f} m")
    print(f"Mediana: {np.median(valores):.2f} m")
    print(f"Distância máxima armazenada: {valores.max():.2f} m")
    print(
        f"No limite de {LIMITE_DISTANCIA_M:.0f} m: "
        f"{np.count_nonzero(valores == LIMITE_DISTANCIA_M):,}"
    )
    print("Verificação: distância zero nos pixels urbanos.")
    print("\nEtapa 7 concluída!")


if __name__ == "__main__":
    main()