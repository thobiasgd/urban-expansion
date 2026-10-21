import csv
from pathlib import Path

import numpy as np
import rasterio
from numpy.typing import NDArray
from rasterio.crs import CRS
from rasterio.io import DatasetReader


# 1. Configurações e caminhos
ANO_BASE: int = 2020
ANO_FINAL: int = 2025
CLASSE_URBANA: int = 24
SEM_DADOS: int = 255
RESOLUCAO: float = 30.0
CRS_ESPERADO: CRS = CRS.from_epsg(31982)

# Códigos presentes na legenda oficial da Coleção 11,
# incluindo classes agregadas da hierarquia.
CODIGOS_COBERTURA: tuple[int, ...] = (
    1, 3, 4, 5, 6, 7, 9, 10, 11, 12, 14, 15,
    18, 19, 20, 21, 22, 23, 24, 25, 26, 29, 30, 31,
    32, 33, 35, 36, 39, 40, 41, 46, 47, 48, 49, 50,
    62, 75, 77, 84, 91,
)

RAIZ: Path = Path(__file__).resolve().parent.parent

ARQUIVO_ORIGEM: Path = (
    RAIZ / "data" / "raw" / "mapbiomas"
    / f"uso_{ANO_BASE}_{ANO_FINAL}.tif"
)

ARQUIVO_TARGET: Path = (
    RAIZ / "data" / "processed"
    / f"target_{ANO_BASE}_{ANO_FINAL}.tif"
)

ARQUIVO_RELATORIO: Path = (
    RAIZ / "data" / "processed"
    / f"auditoria_transicoes_{ANO_BASE}_{ANO_FINAL}.csv"
)


# 2. Contar pixels que atendem a uma condição
def contar(mascara: NDArray[np.bool_]) -> int:
    """Retorna a quantidade de posições True em uma matriz booleana."""
    return int(np.count_nonzero(mascara))


# 3. Conferir metadados e alinhamento espacial
def validar_grade(
    origem: DatasetReader,
    alvo: DatasetReader,
) -> None:
    """Confere se os dois arquivos usam exatamente a mesma grade."""
    if origem.count != 2 or alvo.count != 1:
        raise ValueError(
            "Esperávamos 2 bandas na origem e 1 no target."
        )

    if origem.crs != CRS_ESPERADO or alvo.crs != CRS_ESPERADO:
        raise ValueError(
            "Um dos arquivos está com CRS diferente do esperado."
        )

    if origem.shape != alvo.shape:
        raise ValueError(
            "Os arquivos têm diferentes números de linhas/colunas."
        )

    if origem.transform != alvo.transform:
        raise ValueError(
            "Origem e target não estão na mesma grade espacial."
        )

    if not np.allclose(
        origem.res,
        (RESOLUCAO, RESOLUCAO),
        rtol=0,
        atol=1e-6,
    ):
        raise ValueError(
            f"Resolução inesperada: {origem.res}."
        )

    if not np.allclose(
        (origem.transform.b, origem.transform.d),
        (0.0, 0.0),
        rtol=0,
        atol=1e-9,
    ):
        raise ValueError(
            "Este projeto espera uma grade sem rotação."
        )

    if (
        origem.dtypes != ("uint8", "uint8")
        or alvo.dtypes != ("uint8",)
    ):
        raise ValueError(
            "Os arquivos deveriam armazenar valores uint8."
        )

    if alvo.nodata != SEM_DADOS:
        raise ValueError(
            "O NoData do target deveria ser 255."
        )


# 4. Verificar os códigos encontrados nas classificações
def validar_codigos(
    uso: NDArray[np.uint8],
    ano: int,
) -> None:
    """Interrompe a execução ao encontrar códigos não previstos."""
    codigos = np.unique(uso)

    permitidos = (
        *CODIGOS_COBERTURA,
        0,
        SEM_DADOS,
    )

    desconhecidos = codigos[
        ~np.isin(codigos, permitidos)
    ]

    print(
        f"Códigos encontrados em {ano}: "
        f"{codigos.tolist()}"
    )

    if desconhecidos.size > 0:
        raise ValueError(
            f"Códigos inesperados em {ano}: "
            f"{desconhecidos.tolist()}. "
            "Confira a legenda e a versão do arquivo "
            "antes de continuar."
        )


# 5. Reconstituir os rótulos e contar as quatro transições
def analisar_transicoes(
    uso_base: NDArray[np.uint8],
    uso_final: NDArray[np.uint8],
    target: NDArray[np.uint8],
    validos: NDArray[np.bool_],
) -> dict[str, int]:
    """Confere o target e retorna as contagens na área válida comum."""
    if not validos.any():
        raise ValueError(
            "Nenhum pixel válido nos dois anos."
        )

    urbano_base = uso_base == CLASSE_URBANA
    urbano_final = uso_final == CLASSE_URBANA

    candidatos = validos & ~urbano_base

    esperado = np.full(
        target.shape,
        SEM_DADOS,
        dtype=np.uint8,
    )

    esperado[candidatos] = (
        urbano_final[candidatos]
        .astype(np.uint8)
    )

    divergencias = contar(target != esperado)

    print(
        f"Pixels com target divergente: "
        f"{divergencias:,}"
    )

    if divergencias > 0:
        raise ValueError(
            "O target salvo não corresponde às classificações."
        )

    return {
        "nao_urbano_para_nao_urbano": contar(
            validos & ~urbano_base & ~urbano_final
        ),
        "nao_urbano_para_urbano": contar(
            validos & ~urbano_base & urbano_final
        ),
        "urbano_para_urbano": contar(
            validos & urbano_base & urbano_final
        ),
        "urbano_para_nao_urbano": contar(
            validos & urbano_base & ~urbano_final
        ),
    }


# 6. Salvar o relatório sem modificar os rasters
def salvar_relatorio(
    transicoes: dict[str, int],
    area_pixel_m2: float,
    destino: Path,
) -> None:
    """Salva contagens e áreas planas das células em um CSV."""
    destino.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with destino.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as arquivo:
        escritor = csv.writer(arquivo)

        escritor.writerow([
            "ano_base",
            "ano_final",
            "transicao",
            "pixels",
            "area_km2",
        ])

        for nome, quantidade in transicoes.items():
            area_km2 = (
                quantidade
                * area_pixel_m2
                / 1_000_000
            )

            escritor.writerow([
                ANO_BASE,
                ANO_FINAL,
                nome,
                quantidade,
                area_km2,
            ])

    print(f"\nRelatório salvo em: {destino}")


# 7. Fluxo principal
def main() -> None:
    for caminho in (
        ARQUIVO_ORIGEM,
        ARQUIVO_TARGET,
    ):
        if not caminho.is_file():
            raise FileNotFoundError(
                f"Arquivo não encontrado: {caminho}"
            )

    print("AUDITORIA LOCAL DO TARGET")

    with rasterio.open(ARQUIVO_ORIGEM) as origem:
        with rasterio.open(ARQUIVO_TARGET) as alvo:
            validar_grade(origem, alvo)

            uso_base: NDArray[np.uint8] = origem.read(1)
            uso_final: NDArray[np.uint8] = origem.read(2)
            target: NDArray[np.uint8] = alvo.read(1)

            validos: NDArray[np.bool_] = (
                (origem.read_masks(1) > 0)
                & (origem.read_masks(2) > 0)
                & np.isin(
                    uso_base,
                    CODIGOS_COBERTURA,
                )
                & np.isin(
                    uso_final,
                    CODIGOS_COBERTURA,
                )
            )

            if not np.array_equal(
                alvo.read_masks(1) > 0,
                target != SEM_DADOS,
            ):
                raise ValueError(
                    "Máscara de validade do target inconsistente."
                )

            transformacao = origem.transform

            area_pixel_m2 = float(abs(
                transformacao.a * transformacao.e
                - transformacao.b * transformacao.d
            ))

            print(f"CRS: {origem.crs}")

            print(
                "Dimensões (linhas, colunas): "
                f"{origem.shape}"
            )

            print(f"Resolução: {origem.res}")

            print(
                "Área de cada célula: "
                f"{area_pixel_m2:.6f} m²"
            )

            print("Grades espaciais coincidentes.")

    validar_codigos(
        uso_base,
        ANO_BASE,
    )

    validar_codigos(
        uso_final,
        ANO_FINAL,
    )

    transicoes = analisar_transicoes(
        uso_base,
        uso_final,
        target,
        validos,
    )

    negativos = transicoes[
        "nao_urbano_para_nao_urbano"
    ]

    ganhos = transicoes[
        "nao_urbano_para_urbano"
    ]

    permanencias = transicoes[
        "urbano_para_urbano"
    ]

    perdas = transicoes[
        "urbano_para_nao_urbano"
    ]

    elegiveis = negativos + ganhos

    if elegiveis == 0:
        raise ValueError(
            "Não há pixels elegíveis para o problema de ML."
        )

    fator_km2 = area_pixel_m2 / 1_000_000

    area_base = (
        permanencias + perdas
    ) * fator_km2

    area_final = (
        permanencias + ganhos
    ) * fator_km2

    print("\nTRANSIÇÕES NA MESMA GRADE")

    for nome, quantidade in transicoes.items():
        print(
            f"{nome}: "
            f"{quantidade:,} pixels"
        )

    print("\nÁREAS NA INTERSEÇÃO VÁLIDA DOS DOIS ANOS")

    print(
        f"Área urbana em {ANO_BASE}: "
        f"{area_base:.4f} km²"
    )

    print(
        f"Área urbana em {ANO_FINAL}: "
        f"{area_final:.4f} km²"
    )

    print(
        "Conversões para urbano: "
        f"{ganhos * fator_km2:.4f} km²"
    )

    print(
        "Saídas da classe urbana: "
        f"{perdas * fator_km2:.4f} km²"
    )

    print(
        "Variação líquida: "
        f"{area_final - area_base:+.4f} km²"
    )

    print("\nDISTRIBUIÇÃO DOS CANDIDATOS")

    print(
        f"Pixels elegíveis: "
        f"{elegiveis:,}"
    )

    print(
        "Taxa de positivos: "
        f"{100 * ganhos / elegiveis:.3f}%"
    )

    print(
        "Acurácia de prever sempre 0: "
        f"{100 * negativos / elegiveis:.3f}%"
    )

    salvar_relatorio(
        transicoes,
        area_pixel_m2,
        ARQUIVO_RELATORIO,
    )

    print("\nAuditoria estrutural concluída!")


if __name__ == "__main__":
    main()