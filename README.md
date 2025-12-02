# py-data-medical-embeddings

**Objetivo**

Criar uma base vetorial no Pinecone a partir de arquivos CSV envolve algumas etapas essenciais: configuração, leitura de dados, geração de embeddings e o upsert (inserção) no índice.

O código a seguir utiliza o Pandas para ler o CSV e a OpenAI para gerar os vetores, uma combinação padrão e eficiente.

⚠️ Observação sobre o Plano Gratuito: O plano gratuito da Pinecone permite apenas um índice e tem um limite de 50.000 vetores. Certifique-se de que sua base de dados de exames não exceda esse limite, e use um nome de índice que não esteja em uso se você já tiver um.

**Pré-requisitos e Instalação**

Você precisará instalar as seguintes bibliotecas:

```console
pip install pandas pinecone-client openai
```

**Código Python para Criação do Índice**

Este script fará a leitura do arquivo fleury-db.csv, gerará os embeddings e fará o upsert no Pinecone.

**Como Executar o Script**

Configure as Chaves: Crie as variáveis de ambiente PINECONE_API_KEY e OPENAI_API_KEY no seu terminal ou preencha as strings SUA_CHAVE... no topo do código.

Ajuste as Colunas: Revise as linhas que definem ID_COLUMN e, principalmente, a criação da coluna df['TEXT_TO_EMBED']. Você deve ajustar os nomes das colunas (METODO, TIPO_COLETA, etc.) para corresponderem exatamente aos cabeçalhos do seu arquivo fleury-db.csv.

```console
python seu_script_de_ingestao.py
```