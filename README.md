# Análise Comparativa de Técnicas de Melhoramento de Imagem em Exames de VFSS

Avalia o impacto de técnicas de realce e redução de ruído na qualidade visual
de exames de videofluoroscopia da deglutição e na segmentação das vértebras
C2–C4. O plano completo está em [plano_pesquisa_vfss.pdf](plano_pesquisa_vfss.pdf).

## Instalação

Requer [uv](https://docs.astral.sh/uv/) (`pip install uv`).

```powershell
uv sync                 # ambiente idêntico ao uv.lock (sem PyTorch)
uv sync --extra dl      # + PyTorch com CUDA 12.8, a partir da Fase 1
```

## Executar um experimento

```powershell
uv run python -m vfss experiment=fase0_extracao_quadros
```

Qualquer valor da configuração pode ser trocado na linha de comando, e a
troca fica registrada:

```powershell
uv run python -m vfss experiment=fase0_extracao_quadros extract.max_frames_per_video=50
uv run python -m vfss experiment=<nome> seed=0,1,2,3,4 --multirun
```

Resultados:

- `outputs/<experimento>/<data_hora>/`: log, configuração e linha de comando da execução
- `mlruns/`: métricas, parâmetros e proveniência de todas as execuções. Para ver:
  `uv run mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db` e abrir http://127.0.0.1:5000

## Infraestrutura de reprodutibilidade (Fase 0)

| Exigência do plano | Como está atendida |
|---|---|
| Repositório com controle de versão | git; dados, saídas e vídeos bloqueados no `.gitignore` |
| Configuração por experimento | `configs/experiment/<nome>.yaml` (Hydra). O código não tem parâmetros experimentais |
| Registro automático de métricas | toda execução passa por `vfss.tracking.tracked_run` → MLflow local |
| Sementes fixas | `seed` na configuração; `vfss.seed.seed_everything` fixa `random`, NumPy e PyTorch |
| Versões travadas | `uv.lock` com hash de cada pacote; o hash do lock é gravado em cada execução |

Cada execução registra automaticamente: configuração resolvida, commit do git,
se havia alterações não commitadas (com o diff), hash do `uv.lock`, Python,
sistema, GPU, tempo de execução e status. Execuções que falham também ficam
registradas (status `FAILED`).

**Regra prática:** faça commit antes de rodar um experimento que vai para o
texto. Uma execução com `git_dirty=True` ainda é reproduzível pelo diff salvo,
mas fica mais difícil de auditar.

## Novo experimento

1. Copie um arquivo em `configs/experiment/` e mude `experiment_name` para o
   nome do arquivo (um teste verifica isso).
2. Se for um pipeline novo, escreva `run(cfg, run_dir)` em `src/vfss/pipelines/`
   e registre-o em `PIPELINES`.
3. `uv run pytest`.

## Formato padronizado da extração de quadros

Saída de `fase0_extracao_quadros` em `data/interim/frames/<experimento>/`:

| Arquivo | Conteúdo |
|---|---|
| `<video_id>/<quadro>.png` | quadro sem nenhuma alteração, PNG sem perda (8 ou 16 bits) |
| `frames.csv` | uma linha por quadro: `video_id, patient_id, source, frame_index, time_s, path, height, width, channels, dtype` |
| `videos.csv` | uma linha por exame: origem, paciente, SHA-256 do arquivo original, leitor, quadros declarados/decodificados/gravados, fps, resolução, `bits_stored`, `photometric`, fração de quadros com canais de cor idênticos |
| `failures.csv` | exames ignorados e o motivo (fora do padrão de caminho, corrompido...) |
| `provenance.json` | execução do MLflow, commit e configuração que geraram a pasta |

Leia com `vfss.pipelines.extract_frames.read_extraction`, que mantém os
identificadores como texto (paciente `007` não vira `7`).

Observação: vídeos comuns (MP4/AVI) são decodificados pelo OpenCV em 8 bits.
DICOM preserva a profundidade original (ex.: 12 bits em PNG de 16 bits).

## Testes

```powershell
uv run pytest
```

Os testes geram exames sintéticos (AVI e DICOM de 12 bits) e verificam o
pipeline de ponta a ponta, a reprodutibilidade das sementes e o registro no MLflow.

## Estrutura

```
configs/            configuração Hydra (raiz, caminhos, tracking, experimentos)
src/vfss/           pacote: config, seed, tracking, video, pipelines/
tests/              testes com dados sintéticos
data/               dados clínicos (não versionados), ver data/README.md
outputs/, mlruns/   resultados de execução (não versionados)
```
