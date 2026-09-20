#!/usr/bin/env python3
"""
Converte uma nuvem de pontos densa (LAS/LAZ/PLY/XYZ) em um arquivo
3D Gaussian Splatting (.ply no formato do Inria/3DGS).

A conversao NAO cria informacao visual nova: nao ha treino contra imagens,
entao nao existe cor dependente do angulo de visao (SH grau 0 apenas).
O ganho e de continuidade de superficie: cada ponto vira um "surfel" -
um gaussiano achatado, orientado pela normal local e dimensionado pelo
espacamento local dos vizinhos - de modo que os discos se sobrepoem e a
superficie aparece solida em vez de granulada.

Uso tipico:
    python3 tools/nuvem_para_splat.py voo_t16.laz -o projetos/T_16/talude_t16.ply

Depois, no visualizador, use o offset gravado no .json ao lado da saida.
"""

import argparse
import json
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

# Coeficiente do harmonico esferico de grau 0 (usado pelo 3DGS do Inria).
SH_C0 = 0.28209479177387814

# Processa os vizinhos em blocos para nao estourar a memoria em nuvens grandes.
TAMANHO_BLOCO = 200_000


# ---------------------------------------------------------------- leitura ---

def ler_las(caminho):
    import laspy
    with laspy.open(caminho) as f:
        las = f.read()
    xyz = np.column_stack([las.x, las.y, las.z]).astype(np.float64)

    dims = {d.name.lower() for d in las.point_format.dimensions}
    if {"red", "green", "blue"} <= dims:
        rgb = np.column_stack([las.red, las.green, las.blue]).astype(np.float64)
        # LAS guarda cor em 16 bits; algumas exportacoes ja gravam em 8.
        if rgb.max() > 255:
            rgb /= 257.0
    elif "intensity" in dims:
        inten = np.asarray(las.intensity, dtype=np.float64)
        faixa = np.percentile(inten, [2, 98])
        cinza = np.clip((inten - faixa[0]) / max(faixa[1] - faixa[0], 1e-9), 0, 1) * 255.0
        rgb = np.repeat(cinza[:, None], 3, axis=1)
        print("  aviso: nuvem sem RGB, usando intensidade em escala de cinza")
    else:
        rgb = np.full_like(xyz, 180.0)
        print("  aviso: nuvem sem RGB e sem intensidade, usando cinza uniforme")

    return xyz, np.clip(rgb, 0, 255)


def voxel_para_alvo(mins, maxs, alvo):
    """Aresta de voxel que rende ~alvo pontos. A nuvem e uma superficie
    drapeada, entao a densidade e governada pela area dos dois maiores eixos."""
    extensao = np.sort(np.asarray(maxs) - np.asarray(mins))[::-1]
    area = extensao[0] * extensao[1]
    if area <= 0 or alvo <= 0:
        return 0.0
    return max(float(np.sqrt(area / alvo)), 1e-4)


def _fundir(estado, chaves, soma_xyz, soma_rgb, contagem):
    """Funde um lote reduzido no acumulador, somando voxels repetidos."""
    if estado is not None:
        chaves = np.concatenate([estado[0], chaves])
        soma_xyz = np.concatenate([estado[1], soma_xyz])
        soma_rgb = np.concatenate([estado[2], soma_rgb])
        contagem = np.concatenate([estado[3], contagem])

    unicas, inverso = np.unique(chaves, return_inverse=True)
    n = len(unicas)
    acc_xyz, acc_rgb, acc_n = np.zeros((n, 3)), np.zeros((n, 3)), np.zeros(n)
    np.add.at(acc_xyz, inverso, soma_xyz)
    np.add.at(acc_rgb, inverso, soma_rgb)
    np.add.at(acc_n, inverso, contagem)
    return unicas, acc_xyz, acc_rgb, acc_n


def _reduzir_lote(xyz, rgb, origem, voxel, ny, nz):
    """Agrupa um lote de pontos por voxel e devolve as somas por voxel."""
    ijk = np.floor((xyz - origem) / voxel).astype(np.int64)
    chaves = (ijk[:, 0] * ny + ijk[:, 1]) * nz + ijk[:, 2]
    unicas, inverso, contagem = np.unique(chaves, return_inverse=True, return_counts=True)
    n = len(unicas)
    soma_xyz, soma_rgb = np.zeros((n, 3)), np.zeros((n, 3))
    np.add.at(soma_xyz, inverso, xyz)
    np.add.at(soma_rgb, inverso, rgb)
    return unicas, soma_xyz, soma_rgb, contagem.astype(np.float64)


def ler_las_reduzido(caminho, voxel, alvo, pontos_por_lote=4_000_000):
    """Le LAS/LAZ em lotes, reamostrando em voxels durante a leitura.

    E o unico caminho viavel para nuvens de centenas de milhoes de pontos:
    a memoria fica limitada ao numero de voxels, nao ao numero de pontos.
    """
    import laspy

    with laspy.open(caminho) as f:
        hdr = f.header
        mins, maxs = np.asarray(hdr.mins), np.asarray(hdr.maxs)
        total = hdr.point_count

        if voxel <= 0:
            voxel = voxel_para_alvo(mins, maxs, alvo)
            print(f"  alvo de {alvo:,} pontos -> voxel de {voxel:.3f} m")

        ny = int((maxs[1] - mins[1]) / voxel) + 2
        nz = int((maxs[2] - mins[2]) / voxel) + 2
        nx = int((maxs[0] - mins[0]) / voxel) + 2
        if nx * ny * nz >= 2 ** 62:
            raise ValueError(f"voxel de {voxel:.4f} m e pequeno demais para a extensao da nuvem")

        dims = {d.name.lower() for d in hdr.point_format.dimensions}
        tem_rgb = {"red", "green", "blue"} <= dims
        if not tem_rgb:
            print("  aviso: nuvem sem RGB, usando cinza uniforme")

        estado = None
        lidos = 0
        for lote in f.chunk_iterator(pontos_por_lote):
            xyz = np.column_stack([lote.x, lote.y, lote.z]).astype(np.float64)
            if tem_rgb:
                rgb = np.column_stack([lote.red, lote.green, lote.blue]).astype(np.float64)
                if rgb.max() > 255:
                    rgb /= 257.0
            else:
                rgb = np.full_like(xyz, 180.0)
            estado = _fundir(estado, *_reduzir_lote(xyz, np.clip(rgb, 0, 255), mins, voxel, ny, nz))
            lidos += len(xyz)
            print(f"\r  lidos {lidos:,}/{total:,} pontos -> {len(estado[0]):,} voxels", end="", flush=True)

    print()
    _, soma_xyz, soma_rgb, contagem = estado
    c = contagem[:, None]
    return soma_xyz / c, soma_rgb / c, voxel


def ler_ply(caminho):
    from plyfile import PlyData
    ply = PlyData.read(caminho)
    v = ply["vertex"].data
    nomes = {n.lower(): n for n in v.dtype.names}

    xyz = np.column_stack([v[nomes["x"]], v[nomes["y"]], v[nomes["z"]]]).astype(np.float64)

    for trio in (("red", "green", "blue"), ("r", "g", "b"), ("diffuse_red", "diffuse_green", "diffuse_blue")):
        if all(c in nomes for c in trio):
            rgb = np.column_stack([v[nomes[c]] for c in trio]).astype(np.float64)
            if rgb.max() <= 1.0:  # alguns exportadores gravam cor normalizada
                rgb *= 255.0
            return xyz, np.clip(rgb, 0, 255)

    print("  aviso: PLY sem RGB, usando cinza uniforme")
    return xyz, np.full_like(xyz, 180.0)


def ler_texto(caminho):
    dados = np.loadtxt(caminho, delimiter=None if caminho.endswith(".xyz") else ",")
    if dados.shape[1] < 3:
        raise ValueError("arquivo de texto precisa de ao menos 3 colunas (X Y Z)")
    xyz = dados[:, :3].astype(np.float64)
    if dados.shape[1] >= 6:
        rgb = np.clip(dados[:, 3:6].astype(np.float64), 0, 255)
    else:
        print("  aviso: arquivo sem RGB, usando cinza uniforme")
        rgb = np.full_like(xyz, 180.0)
    return xyz, rgb


def ler_nuvem(caminho):
    """Devolve (xyz, rgb, ordem_padrao). LAS/LAZ e sempre Z-up (E, N, cota),
    entao precisa de 'xzy' para virar a ordem (E, cota, N) do visualizador."""
    ext = os.path.splitext(caminho)[1].lower()
    if ext in (".las", ".laz"):
        return (*ler_las(caminho), "xzy")
    if ext == ".ply":
        return (*ler_ply(caminho), "xyz")
    if ext in (".xyz", ".txt", ".csv", ".pts"):
        return (*ler_texto(caminho), "xyz")
    raise ValueError(f"extensao nao suportada: {ext} (use .las/.laz/.ply/.xyz/.txt/.csv)")


def reordenar_eixos(xyz, ordem):
    """Permuta as colunas de XYZ; 'xzy' troca cota e norte (Z-up -> Y-up)."""
    if ordem == "xyz":
        return xyz
    if sorted(ordem) != ["x", "y", "z"]:
        raise ValueError(f"ordem de eixos invalida: {ordem} (use uma permutacao de x, y, z)")
    return xyz[:, ["xyz".index(c) for c in ordem]]


# ------------------------------------------------------------ preparacao ---

def reamostrar_em_voxels(xyz, rgb, tamanho):
    """Media dos pontos dentro de cada voxel - uniformiza a densidade."""
    chaves = np.floor((xyz - xyz.min(axis=0)) / tamanho).astype(np.int64)
    _, inverso, contagem = np.unique(chaves, axis=0, return_inverse=True, return_counts=True)
    n = len(contagem)
    soma_xyz = np.zeros((n, 3))
    soma_rgb = np.zeros((n, 3))
    np.add.at(soma_xyz, inverso, xyz)
    np.add.at(soma_rgb, inverso, rgb)
    c = contagem[:, None]
    return soma_xyz / c, soma_rgb / c


def vizinhos_em_blocos(arvore, xyz, k):
    """Distancias e indices dos k vizinhos, processados por blocos."""
    n = len(xyz)
    dist = np.empty((n, k), dtype=np.float64)
    idx = np.empty((n, k), dtype=np.int64)
    for ini in range(0, n, TAMANHO_BLOCO):
        fim = min(ini + TAMANHO_BLOCO, n)
        # k+1 porque o primeiro vizinho e o proprio ponto
        d, i = arvore.query(xyz[ini:fim], k=k + 1, workers=-1)
        dist[ini:fim] = d[:, 1:]
        idx[ini:fim] = i[:, 1:]
    return dist, idx


def remover_outliers(xyz, rgb, dist, desvios):
    """Filtro estatistico (SOR): descarta pontos isolados / 'flyers'."""
    media_local = dist.mean(axis=1)
    limite = media_local.mean() + desvios * media_local.std()
    manter = media_local <= limite
    removidos = int((~manter).sum())
    if removidos:
        print(f"  SOR: {removidos} pontos isolados removidos ({removidos / len(xyz) * 100:.2f}%)")
    return xyz[manter], rgb[manter], manter


# ------------------------------------------------------------- geometria ---

def eixos_locais(xyz, idx):
    """PCA da vizinhanca: retorna (tangente1, tangente2, normal) por ponto."""
    n = len(xyz)
    t1 = np.empty((n, 3))
    t2 = np.empty((n, 3))
    nor = np.empty((n, 3))

    for ini in range(0, n, TAMANHO_BLOCO):
        fim = min(ini + TAMANHO_BLOCO, n)
        viz = xyz[idx[ini:fim]]                       # (b, k, 3)
        centrado = viz - viz.mean(axis=1, keepdims=True)
        cov = np.einsum("bki,bkj->bij", centrado, centrado) / centrado.shape[1]
        # eigh devolve autovalores em ordem crescente; a menor variancia e a normal
        _, vecs = np.linalg.eigh(cov)
        nor[ini:fim] = vecs[:, :, 0]
        t2[ini:fim] = vecs[:, :, 1]
        t1[ini:fim] = vecs[:, :, 2]

    return t1, t2, nor


def matriz_para_quaternion(R):
    """Converte rotacoes (N,3,3) em quaternions (N,4) na ordem (w, x, y, z)."""
    m00, m01, m02 = R[:, 0, 0], R[:, 0, 1], R[:, 0, 2]
    m10, m11, m12 = R[:, 1, 0], R[:, 1, 1], R[:, 1, 2]
    m20, m21, m22 = R[:, 2, 0], R[:, 2, 1], R[:, 2, 2]

    traco = m00 + m11 + m22
    q = np.empty((len(R), 4))

    # Quatro ramos para evitar divisao por valor proximo de zero.
    a = traco > 0
    b = (~a) & (m00 >= m11) & (m00 >= m22)
    c = (~a) & (~b) & (m11 >= m22)
    d = (~a) & (~b) & (~c)

    s = np.sqrt(np.maximum(traco[a] + 1.0, 1e-12)) * 2
    q[a] = np.column_stack([0.25 * s, (m21[a] - m12[a]) / s, (m02[a] - m20[a]) / s, (m10[a] - m01[a]) / s])

    s = np.sqrt(np.maximum(1.0 + m00[b] - m11[b] - m22[b], 1e-12)) * 2
    q[b] = np.column_stack([(m21[b] - m12[b]) / s, 0.25 * s, (m01[b] + m10[b]) / s, (m02[b] + m20[b]) / s])

    s = np.sqrt(np.maximum(1.0 + m11[c] - m00[c] - m22[c], 1e-12)) * 2
    q[c] = np.column_stack([(m02[c] - m20[c]) / s, (m01[c] + m10[c]) / s, 0.25 * s, (m12[c] + m21[c]) / s])

    s = np.sqrt(np.maximum(1.0 + m22[d] - m00[d] - m11[d], 1e-12)) * 2
    q[d] = np.column_stack([(m10[d] - m01[d]) / s, (m02[d] + m20[d]) / s, (m12[d] + m21[d]) / s, 0.25 * s])

    return q / np.linalg.norm(q, axis=1, keepdims=True)


# ----------------------------------------------------------------- escrita ---

def escrever_ply_3dgs(caminho, xyz, normais, f_dc, opacidade, escalas, quat, com_sh_rest):
    n = len(xyz)
    campos = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
    if com_sh_rest:
        campos += [f"f_rest_{i}" for i in range(45)]
    campos += ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]

    dtype = np.dtype([(c, "<f4") for c in campos])
    saida = np.empty(n, dtype=dtype)
    for i, c in enumerate(("x", "y", "z")):
        saida[c] = xyz[:, i]
    for i, c in enumerate(("nx", "ny", "nz")):
        saida[c] = normais[:, i]
    for i in range(3):
        saida[f"f_dc_{i}"] = f_dc[:, i]
    if com_sh_rest:
        for i in range(45):
            saida[f"f_rest_{i}"] = 0.0
    saida["opacity"] = opacidade
    for i in range(3):
        saida[f"scale_{i}"] = escalas[:, i]
    for i in range(4):
        saida[f"rot_{i}"] = quat[:, i]

    cabecalho = ["ply", "format binary_little_endian 1.0", f"element vertex {n}"]
    cabecalho += [f"property float {c}" for c in campos]
    cabecalho += ["end_header", ""]

    with open(caminho, "wb") as f:
        f.write("\n".join(cabecalho).encode("ascii"))
        f.write(saida.tobytes())


def escrever_splat(caminho, xyz, f_dc, opacidade, escalas, quat):
    """Formato .splat (antimatter15): 32 bytes por gaussiano, ~2x menor que o
    PLY. E o que os viewers web carregam mais rapido."""
    n = len(xyz)
    buf = np.zeros((n, 32), dtype=np.uint8)

    buf[:, 0:12] = xyz.astype("<f4").view(np.uint8).reshape(n, 12)
    buf[:, 12:24] = np.exp(escalas).astype("<f4").view(np.uint8).reshape(n, 12)

    rgb = np.clip((0.5 + SH_C0 * f_dc) * 255, 0, 255)
    alfa = np.clip(255 / (1 + np.exp(-opacidade)), 0, 255)
    buf[:, 24:27] = rgb.astype(np.uint8)
    buf[:, 27] = alfa.astype(np.uint8)

    q = quat / np.linalg.norm(quat, axis=1, keepdims=True)
    buf[:, 28:32] = np.clip(q * 128 + 128, 0, 255).astype(np.uint8)

    # Os viewers desenham na ordem do arquivo, entao os maiores/mais opacos
    # primeiro reduz artefato enquanto o resto ainda esta carregando.
    peso = np.exp(escalas).sum(axis=1) / (1 + np.exp(-opacidade))
    buf[np.argsort(-peso)].tofile(caminho)


# -------------------------------------------------------------------- main ---

def main():
    p = argparse.ArgumentParser(
        description="Converte nuvem de pontos densa em gaussian splat (.ply 3DGS).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("entrada", help="nuvem de pontos (.las/.laz/.ply/.xyz/.txt/.csv)")
    p.add_argument("-o", "--saida", help="arquivo .ply de saida (padrao: <entrada>_splat.ply)")
    p.add_argument("--alvo", type=int, default=0,
                   help="numero desejado de gaussianos; escolhe o voxel sozinho. "
                        "Use ~1500000 para web, ~4000000 para desktop")
    p.add_argument("--formato", choices=("ply", "splat", "ambos"), default="ply",
                   help="ply = 68 bytes/gaussiano (edicao); splat = 32 bytes (web)")
    p.add_argument("--voxel", type=float, default=0.0,
                   help="aresta do voxel em metros para uniformizar a densidade (0 = nao reamostrar)")
    p.add_argument("--max-pontos", type=int, default=0,
                   help="teto de pontos; acima disso faz amostragem aleatoria (0 = sem teto)")
    p.add_argument("-k", "--vizinhos", type=int, default=16,
                   help="vizinhos usados para estimar normal e espacamento local")
    p.add_argument("--escala", type=float, default=0.6,
                   help="sigma tangencial = escala x espacamento entre pontos; raio visivel ~2 sigma")
    p.add_argument("--espessura", type=float, default=0.15,
                   help="razao entre sigma normal e sigma tangencial; baixo = disco achatado")
    p.add_argument("--opacidade", type=float, default=0.9, help="alfa de cada gaussiano (0-1)")
    p.add_argument("--sor", type=float, default=2.5,
                   help="remocao de outliers em desvios-padrao (0 = desligado)")
    p.add_argument("--origem", type=float, nargs=3, metavar=("E", "COTA", "N"),
                   help="origem local a subtrair; padrao: centroide da nuvem")
    p.add_argument("--ordem-eixos", default="auto",
                   help="permutacao das colunas XYZ; 'auto' usa xzy para LAS/LAZ (Z-up) e xyz "
                        "para o resto. A saida fica na ordem (E, cota, N) do visualizador.html")
    p.add_argument("--sem-recentrar", action="store_true",
                   help="mantem as coordenadas absolutas (NAO use com UTM: float32 perde ~0.5 m)")
    p.add_argument("--sh-rest", action="store_true",
                   help="grava os 45 coeficientes f_rest zerados (arquivo 5x maior; so para viewers antigos)")
    args = p.parse_args()

    saida = args.saida or os.path.splitext(args.entrada)[0] + "_splat.ply"

    # Cria a pasta de destino antes de processar: descobrir que ela nao existe
    # so na hora de gravar jogaria fora todo o trabalho de conversao.
    pasta = os.path.dirname(os.path.abspath(saida))
    os.makedirs(pasta, exist_ok=True)

    print(f"Lendo {args.entrada} ...")
    ext = os.path.splitext(args.entrada)[1].lower()
    reduziu_na_leitura = False

    if ext in (".las", ".laz") and (args.voxel > 0 or args.alvo > 0):
        # Nuvem grande: reamostra durante a leitura, sem nunca materializar
        # todos os pontos na memoria.
        xyz, rgb, voxel_usado = ler_las_reduzido(args.entrada, args.voxel, args.alvo)
        ordem_padrao = "xzy"
        reduziu_na_leitura = True
        print(f"  {len(xyz):,} pontos apos reamostragem")
    else:
        xyz, rgb, ordem_padrao = ler_nuvem(args.entrada)
        voxel_usado = 0.0
        print(f"  {len(xyz):,} pontos")

    ordem = ordem_padrao if args.ordem_eixos == "auto" else args.ordem_eixos.lower()
    if ordem != "xyz":
        xyz = reordenar_eixos(xyz, ordem)
        print(f"  eixos reordenados para '{ordem}' -> (E, cota, N)")

    if not reduziu_na_leitura:
        voxel = args.voxel or voxel_para_alvo(xyz.min(axis=0), xyz.max(axis=0), args.alvo)
        if voxel > 0:
            if args.alvo and not args.voxel:
                print(f"  alvo de {args.alvo:,} pontos -> voxel de {voxel:.3f} m")
            xyz, rgb = reamostrar_em_voxels(xyz, rgb, voxel)
            voxel_usado = voxel
            print(f"  voxel {voxel:.3f} m -> {len(xyz):,} pontos")

    if args.max_pontos and len(xyz) > args.max_pontos:
        sel = np.random.default_rng(0).choice(len(xyz), args.max_pontos, replace=False)
        xyz, rgb = xyz[sel], rgb[sel]
        print(f"  amostragem -> {len(xyz):,} pontos")

    if len(xyz) <= args.vizinhos:
        sys.exit(f"erro: {len(xyz)} pontos e pouco para k={args.vizinhos} vizinhos")

    # As coordenadas ficam em float64 ate o fim; o recentramento acontece antes
    # da escrita, porque o PLY do 3DGS e float32 e nao aguenta UTM absoluto.
    print("Estimando vizinhanca ...")
    arvore = cKDTree(xyz)
    dist, idx = vizinhos_em_blocos(arvore, xyz, args.vizinhos)

    if args.sor > 0:
        xyz, rgb, manter = remover_outliers(xyz, rgb, dist, args.sor)
        if not manter.all():
            arvore = cKDTree(xyz)
            dist, idx = vizinhos_em_blocos(arvore, xyz, args.vizinhos)

    print("Orientando os surfels ...")
    t1, t2, nor = eixos_locais(xyz, idx)

    # Colunas de R na mesma ordem de scale_0/1/2: duas tangentes e a normal.
    R = np.stack([t1, t2, nor], axis=2)
    invertida = np.linalg.det(R) < 0          # garante base destra
    R[invertida, :, 2] *= -1
    nor = R[:, :, 2]
    quat = matriz_para_quaternion(R)

    # Numa superficie localmente 2D cabem pi*r^2/d^2 vizinhos dentro do raio r,
    # entao a distancia ao k-esimo vizinho e d*sqrt(k/pi) e o espacamento medio
    # sai de d = r_k*sqrt(pi/k). Usar o k-esimo (e nao o primeiro) torna a
    # estimativa robusta a pontos duplicados.
    espacamento = dist[:, -1] * np.sqrt(np.pi / args.vizinhos)
    sigma_t = np.maximum(espacamento * args.escala, 1e-6)
    sigma_n = np.maximum(sigma_t * args.espessura, 1e-7)
    escalas = np.log(np.column_stack([sigma_t, sigma_t, sigma_n]))
    print(f"  espacamento mediano {np.median(espacamento):.4f} m "
          f"-> sigma tangencial {np.median(sigma_t):.4f} m")

    # Cor: o 3DGS guarda o termo DC do SH, nao o RGB direto.
    f_dc = (rgb / 255.0 - 0.5) / SH_C0

    alfa = float(np.clip(args.opacidade, 1e-4, 1 - 1e-4))
    opacidade = np.full(len(xyz), np.log(alfa / (1 - alfa)))   # logit

    if args.sem_recentrar:
        origem = np.zeros(3)
    elif args.origem:
        origem = np.asarray(args.origem, dtype=np.float64)
    else:
        origem = xyz.mean(axis=0)
    xyz_local = xyz - origem

    extensao = np.abs(xyz_local).max()
    if extensao > 1e5:
        print(f"  AVISO: coordenadas ate {extensao:.0f} m da origem; float32 vai perder precisao")

    xyz32 = xyz_local.astype(np.float32)
    f_dc32, op32 = f_dc.astype(np.float32), opacidade.astype(np.float32)
    esc32, quat32 = escalas.astype(np.float32), quat.astype(np.float32)

    gerados = []
    if args.formato in ("ply", "ambos"):
        print(f"Gravando {saida} ...")
        escrever_ply_3dgs(saida, xyz32, nor.astype(np.float32), f_dc32, op32,
                          esc32, quat32, args.sh_rest)
        gerados.append(saida)
    if args.formato in ("splat", "ambos"):
        caminho_splat = os.path.splitext(saida)[0] + ".splat"
        print(f"Gravando {caminho_splat} ...")
        escrever_splat(caminho_splat, xyz32, f_dc32, op32, esc32, quat32)
        gerados.append(caminho_splat)

    meta = {
        "origem_local": {"E": origem[0], "cota": origem[1], "N": origem[2]},
        "observacao": "some estes valores as coordenadas do splat para voltar ao sistema original",
        "ordem_eixos_aplicada": ordem,
        "num_gaussianos": int(len(xyz)),
        "sigma_tangencial_mediano_m": float(np.median(sigma_t)),
        "sh_degree": 0,
        "parametros": {
            "voxel": args.voxel, "vizinhos": args.vizinhos, "escala": args.escala,
            "espessura": args.espessura, "opacidade": args.opacidade, "sor": args.sor,
        },
    }
    caminho_meta = os.path.splitext(saida)[0] + ".json"
    meta["parametros"]["alvo"] = args.alvo
    with open(caminho_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"OK: {len(xyz):,} gaussianos")
    if args.alvo:
        desvio = len(xyz) / args.alvo
        if not 0.8 <= desvio <= 1.25:
            # O alvo sai da area em planta; terreno inclinado tem mais superficie
            # que planta, entao o resultado costuma passar do pedido.
            sugerido = voxel_usado * np.sqrt(desvio)
            print(f"    ficou {desvio:.2f}x o alvo; para chegar perto, "
                  f"repita com --voxel {sugerido:.3f}")
    for g in gerados:
        print(f"    {g}  ({os.path.getsize(g) / 1e6:.1f} MB)")
    print(f"    offset da origem gravado em {caminho_meta}")


if __name__ == "__main__":
    main()
