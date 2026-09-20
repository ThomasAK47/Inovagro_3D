# tools/

## `nuvem_para_splat.py` — nuvem de pontos → gaussian splat

Converte uma nuvem densa (`.las`, `.laz`, `.ply`, `.xyz`, `.txt`, `.csv`) num
**3D Gaussian Splatting** (`.ply` do Inria ou `.splat`), para uso visual em
apresentações e relatórios.

### O que a conversão faz — e o que não faz

**Não faz:** não há treino contra imagens, então não existe cor dependente do ângulo
de visão. A saída usa harmônicos esféricos de grau 0 — a cor de cada gaussiano é fixa,
copiada do ponto original. Nenhuma informação visual nova é criada, nenhum buraco da
nuvem é preenchido, nenhum detalhe abaixo da resolução da nuvem aparece.

**Faz:** cada ponto vira um *surfel* — um gaussiano achatado, **orientado pela normal
local** e **dimensionado pelo espaçamento local** dos vizinhos. Os discos se sobrepõem
e o talude é renderizado como superfície contínua em vez de poeira granulada. É esse o
ganho, e é o que os conversores genéricos (escala isotrópica fixa, rotação identidade)
não fazem.

Se você ainda tem as fotos do voo, **não use este script**: treine um 3DGS de verdade
a partir das imagens (Postshot, Brush, gsplat/Nerfstudio).

### Instalação

```bash
pip install numpy scipy plyfile "laspy[lazrs]"
```

### Uso

```bash
# nuvem grande, saída para web: escolhe o voxel sozinho e grava .splat
python3 tools/nuvem_para_splat.py voo_t16.laz -o projetos/T_16/talude_t16.ply \
    --alvo 700000 --formato splat

# ajuste fino: o script imprime o --voxel que acerta o alvo em uma tentativa
python3 tools/nuvem_para_splat.py voo_t16.laz -o talude.ply --voxel 0.282 --formato splat

# alinhar com o visualizador: mesma origem dos campos originE/originElev/originN
python3 tools/nuvem_para_splat.py voo_t16.laz -o talude.ply --origem 231675 138 6714216
```

Nuvens em `.las`/`.laz` com `--alvo` ou `--voxel` são lidas **em lotes**, reamostrando
durante a leitura: o pico de memória depende do número de gaussianos pedido, não do
tamanho do arquivo. Medido: 20 milhões de pontos (230 MB de LAZ) → 2,2 M de gaussianos
em 45 s com 1,8 GB de pico.

### Parâmetros que importam

| Flag | Padrão | Efeito |
|---|---|---|
| `--alvo` | `0` | Número desejado de gaussianos; deriva o voxel da área em planta. **Costuma passar do alvo** (~1,5× em terreno inclinado, porque a superfície real é maior que a planta). O script imprime o `--voxel` que corrige, e uma segunda passada acerta. |
| `--formato` | `ply` | `splat` = 32 bytes/gaussiano, `ply` = 68. Para web, use `splat`. |
| `--escala` | `0.6` | Tamanho do disco em múltiplos do espaçamento entre pontos. Raio visível ≈ 2σ, então `0.6` cobre ~1,25× o espaçamento. Abaixo de `0.5` abrem buracos; acima de `1.0` borra o detalhe fino. |
| `--espessura` | `0.15` | Razão σ_normal/σ_tangencial. Baixo = disco colado na superfície. |
| `--voxel` | `0` | Aresta do voxel em metros, quando você quer controlar direto. |
| `--sor` | `2.5` | Remove pontos isolados (vegetação solta, ruído) em desvios-padrão. `0` desliga. |
| `-k` | `16` | Vizinhos para estimar normal e espaçamento. Aumente em nuvem ruidosa. |

### Três armadilhas que o script já trata

**Coordenadas UTM.** O PLY do 3DGS guarda posições em `float32`. Numa coordenada norte
como `6714216.222` o passo do `float32` é de **0,50 m** — o talude viraria uma escada.
O script recentra na origem local por padrão e grava o offset num `.json` ao lado da
saída. Só use `--sem-recentrar` se souber exatamente por quê.

**Convenção de eixos.** LAS/LAZ é Z-up (`E, N, cota`); o `visualizador.html` é Y-up
(`E, cota, N`, os mesmos campos `originE`/`originElev`/`originN`). Com
`--ordem-eixos auto` (padrão) o script aplica `xzy` em LAS/LAZ e deixa os outros
formatos intactos. Se o talude sair deitado, é aqui que se corrige.

**Espaçamento local.** Numa superfície localmente 2D cabem `π·r²/d²` vizinhos dentro do
raio `r`, então o espaçamento médio sai de `d = r_k·√(π/k)` — não de `r_k/√k`. Sem o
`√π` os discos ficam 44% pequenos demais e a superfície renderiza com padrão de tela.

### Limite que nenhum parâmetro resolve

Gaussian splats não têm LOD. Quando um gaussiano projeta menor que um pixel, o shader
simplesmente não o desenha — não há tamanho mínimo. Num talude de ~270 m visto inteiro
na tela, os splats ficam sub-pixel e o modelo parece esmaecido; aproximando, a
superfície fica sólida. Aumentar `--escala` **não** compensa (medido: 0,6 → 1,4 muda a
cobertura a 400 m de 1,5% para 1,7%). A saída é enquadrar mais perto, o que
`visualizador_splat.html` já faz.

### Saída

- `saida.ply` / `saida.splat` — abrem em SuperSplat, PlayCanvas, Postshot,
  `antimatter15/splat` e no `visualizador_splat.html` deste repositório.
- `saida.json` — offset da origem, ordem de eixos aplicada, número de gaussianos e os
  parâmetros usados. Some os valores de `origem_local` às coordenadas do splat para
  voltar ao sistema UTM original.

---

## `../visualizador_splat.html` — visualizador web

Abre um `.splat`/`.ply` seguindo o mesmo padrão de URL do `visualizador.html`:

```
visualizador_splat.html?splat=projetos/T_16/talude_t16.splat&nome=Talude T-16
```

Enquadra a câmera sozinho a partir da caixa envolvente do modelo, então serve para
qualquer talude sem ajuste manual.

**`sharedMemoryForWorkers: false` é obrigatório** e já está no código: memória
compartilhada exige os cabeçalhos COOP/COEP, que o GitHub Pages não deixa configurar.
Com o padrão da biblioteca (`true`) a página quebra em produção.
