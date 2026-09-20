# tools/

## `nuvem_para_splat.py` — nuvem de pontos → gaussian splat

Converte uma nuvem densa (`.las`, `.laz`, `.ply`, `.xyz`, `.txt`, `.csv`) num arquivo
**3D Gaussian Splatting** (`.ply` no formato do Inria), para uso visual em
apresentações e relatórios.

### O que a conversão faz — e o que não faz

**Não faz:** não há treino contra imagens, então não existe cor dependente do ângulo
de visão. A saída usa harmônicos esféricos de grau 0 — a cor de cada gaussiano é fixa,
copiada do ponto original. Nenhuma informação visual nova é criada, nenhum buraco da
nuvem é preenchido, nenhum detalhe abaixo da resolução da nuvem aparece.

**Faz:** cada ponto vira um *surfel* — um gaussiano achatado, **orientado pela normal
local** da superfície e **dimensionado pelo espaçamento local** dos vizinhos. Os discos
se sobrepõem e o talude é renderizado como superfície contínua em vez de poeira
granulada. É esse o ganho visual, e é o que os conversores genéricos (que usam escala
isotrópica fixa e rotação identidade) não fazem.

Se você ainda tem as fotos do voo, **não use este script**: treine um 3DGS de verdade
a partir das imagens (Postshot, Brush, gsplat/Nerfstudio). O resultado é incomparavelmente
melhor.

### Instalação

```bash
pip install numpy scipy plyfile "laspy[lazrs]"
```

### Uso

```bash
# caso comum: LAZ do voo de drone
python3 tools/nuvem_para_splat.py voo_t16.laz -o projetos/T_16/talude_t16.ply

# nuvem muito densa: uniformiza em voxels de 5 cm antes de converter
python3 tools/nuvem_para_splat.py voo_t16.laz -o talude.ply --voxel 0.05

# alinhar com o visualizador: mesma origem usada nos campos originE/originElev/originN
python3 tools/nuvem_para_splat.py voo_t16.laz -o talude.ply --origem 231675 138 6714216
```

### Parâmetros que importam

| Flag | Padrão | Efeito |
|---|---|---|
| `--escala` | `0.6` | Tamanho do disco em múltiplos do espaçamento entre pontos. O raio visível de um gaussiano é ~2σ, então `0.6` cobre ~1,2× o espaçamento. Abaixo de `0.5` abrem buracos; acima de `1.0` o talude borra. |
| `--espessura` | `0.15` | Razão σ_normal/σ_tangencial. Baixo = disco achatado colado na superfície. Aumentar deixa o splat "fofo". |
| `--voxel` | `0` | Reamostra em grade. Use quando a densidade varia muito entre o pé e a crista do talude. |
| `--sor` | `2.5` | Remove pontos isolados (vegetação solta, ruído) em desvios-padrão. `0` desliga. |
| `-k` | `16` | Vizinhos para estimar normal e espaçamento. Aumente em nuvem ruidosa, diminua em nuvem limpa e esparsa. |

### Duas armadilhas que o script já trata

**Coordenadas UTM.** O PLY do 3DGS guarda posições em `float32`. Numa coordenada norte
como `6714216.222` o passo do `float32` é de **0,5 m** — o talude inteiro viraria uma
escada. O script recentra a nuvem na origem local por padrão e grava o offset num `.json`
ao lado da saída. Só use `--sem-recentrar` se souber exatamente por quê.

**Convenção de eixos.** LAS/LAZ é Z-up (`E, N, cota`); o `visualizador.html` é Y-up
(`E, cota, N`, os mesmos campos `originE`/`originElev`/`originN`). Com `--ordem-eixos auto`
(padrão) o script aplica `xzy` em LAS/LAZ e deixa os outros formatos intactos. Se o
talude sair deitado, é aqui que se corrige.

### Saída

- `saida.ply` — o gaussian splat, abre em SuperSplat, PlayCanvas, Postshot,
  `antimatter15/splat` e no `@mkkellogg/gaussian-splats-3d`.
- `saida.json` — offset da origem, ordem de eixos aplicada, número de gaussianos e os
  parâmetros usados. Some os valores de `origem_local` às coordenadas do splat para
  voltar ao sistema UTM original.
