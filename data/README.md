# Dados

Nada nesta pasta é versionado (exceto este arquivo). Exames são dados clínicos.

## Layout esperado

```
data/
  raw/                      exames originais, somente leitura
    <origem>/               instituição ou base (ex.: hc_puc, base_publica)
      <paciente>/           identificador PSEUDONIMIZADO do paciente
        <vídeo>.<ext>       .mp4, .avi, .dcm, DICOM sem extensão...
  interim/
    frames/<experimento>/   saída do pipeline da Fase 0
```

O identificador do paciente vem **só** da pasta, via `extract.path_regex`
em `configs/experiment/fase0_extracao_quadros.yaml`. Se a base chegar em
outro layout, ajuste a expressão regular na configuração, não o código. Arquivo
que não casar com o padrão vai para `failures.csv` e nunca é associado a um
paciente por palpite, porque a divisão treino/teste depende disso.

Nunca use nome, prontuário ou qualquer dado identificável como nome de pasta:
o identificador aparece nos manifestos e no MLflow.

## Dados fora do repositório

Para manter os exames em outro disco, defina a variável de ambiente antes de rodar:

```powershell
$env:VFSS_DATA_DIR = "D:\vfss\data"
```
