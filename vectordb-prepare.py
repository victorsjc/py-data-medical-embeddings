import pandas as pd
import os
import time
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI

# ----------------------------------------------------------------------
# 1. CONFIGURAÇÃO DE AMBIENTE
# ----------------------------------------------------------------------

# As chaves de API e variáveis de ambiente DEVEM ser configuradas
# no ambiente onde este script for executado.
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# Parâmetros do Índice Pinecone
INDEX_NAME = "medical-exams"
# O modelo 'text-embedding-3-small' da OpenAI gera 1536 dimensões
DIMENSION = 1536 
MODEL_NAME = "text-embedding-3-small"

# Nome do arquivo de entrada
INPUT_FILE = "fleury-db.csv"

# Inicialização dos clientes
# Inicialização do Cliente OpenAI
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Inicialização do Cliente Pinecone
# Usa a sintaxe mais recente do 'pinecone-client'
try:
    pinecone_client = Pinecone(api_key=PINECONE_API_KEY)
except Exception as e:
    print(f"Erro ao inicializar Pinecone: {e}")
    exit()

# ----------------------------------------------------------------------
# 2. FUNÇÕES AUXILIARES
# ----------------------------------------------------------------------

def generate_embeddings(texts: list[str]) -> list[list[float]]:
    """Gera embeddings para uma lista de textos usando a API da OpenAI."""
    if not texts:
        return []
    
    response = openai_client.embeddings.create(
        input=texts,
        model=MODEL_NAME
    )
    # Retorna apenas os vetores (data.embedding)
    return [record.embedding for record in response.data]

def create_pinecone_index_if_not_exists():
    """Cria o índice Pinecone se ele ainda não existir."""
    if INDEX_NAME not in pinecone_client.list_indexes().names():
        print(f"Criando índice '{INDEX_NAME}'...")
        
        # Usando a configuração Serverless (plano gratuito compatível)
        pinecone_client.create_index(
            name=INDEX_NAME,
            dimension=DIMENSION,
            metric='cosine', # Métrica ideal para buscas por similaridade
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )
        # Espera o índice ficar pronto
        while not pinecone_client.describe_index(INDEX_NAME).status['ready']:
            time.sleep(1)
        print("Índice criado e pronto.")
    else:
        print(f"Índice '{INDEX_NAME}' já existe. Usando índice existente.")

# ----------------------------------------------------------------------
# 3. PROCESSO PRINCIPAL DE INGESTÃO (UPSERT)
# ----------------------------------------------------------------------

def ingest_data_to_pinecone():
    """Função principal para ler, gerar embeddings e fazer o upsert."""
    
    create_pinecone_index_if_not_exists()
    
    # Conecta ao índice
    index = pinecone_client.Index(INDEX_NAME)

    try:
        # Carrega o CSV
        df = pd.read_csv(INPUT_FILE)
        print(f"Arquivo lido. Total de registros: {len(df)}")
    except FileNotFoundError:
        print(f"Erro: Arquivo '{INPUT_FILE}' não encontrado. Certifique-se de que ele está na mesma pasta.")
        return

    # A coluna usada para o Embedding deve ser a que você vai pesquisar.
    # Criamos uma 'Query String' rica combinando informações relevantes.
    # Exemplo (ajuste os nomes das colunas conforme seu CSV):
    
    # ASSUNÇÃO: Seus arquivos têm colunas como 'ID_PROCEDIMENTO', 'DESCRICAO', 'METODO', 'TIPO_COLETA'
    
# --- ATENÇÃO: AJUSTE OS NOMES DAS COLUNAS CONFORME SEU CSV FINAL ---
    ID_COLUMN = 'id'
    DESC_COLUMN = 'name'
    METHOD_COLUMN = 'analytical_method'
    COLECTION_COLUMN = 'specimen_type'
    SYNONYMS_COLUMN = 'synonyms' # Nova coluna
    SEARCHABLE_TERMS_COLUMN = 'searchable_terms' # Nova coluna
    
    # ------------------------------------------------------------------

    # Preenche colunas ausentes ou vazias com strings vazias para evitar erros
    # Nota: O método .get() é seguro para colunas que podem não existir no DF.
    df[DESC_COLUMN] = df.get(DESC_COLUMN, pd.Series([''] * len(df))).fillna('')
    df[METHOD_COLUMN] = df.get(METHOD_COLUMN, pd.Series([''] * len(df))).fillna('')
    df[COLECTION_COLUMN] = df.get(COLECTION_COLUMN, pd.Series([''] * len(df))).fillna('')
    df[SYNONYMS_COLUMN] = df.get(SYNONYMS_COLUMN, pd.Series([''] * len(df))).fillna('')
    df[SEARCHABLE_TERMS_COLUMN] = df.get(SEARCHABLE_TERMS_COLUMN, pd.Series([''] * len(df))).fillna('')
    
    # --- LÓGICA DE ADAPTAÇÃO ---
    
    # 1. Trata a coluna de Coleta: Substitui campos vazios por 'UNKNOWN'
    df[COLECTION_COLUMN] = df[COLECTION_COLUMN].apply(
        lambda x: 'UNKNOWN' if pd.isna(x) or x.strip() == '' else x
    )
    
    # 2. Constrói a String de Contexto RICA para o Embedding
    df['TEXT_TO_EMBED'] = (
        df[DESC_COLUMN] + " MÉTODO: " + df[METHOD_COLUMN].apply(lambda x: 'UNKNOWN' if x.strip() == '' else x) + 
        " COLETA: " + df[COLECTION_COLUMN].apply(lambda x: 'UNKNOWN' if x.strip() == '' else x) +
        ". SINÔNIMOS: " + df[SYNONYMS_COLUMN] +
        ". TERMOS DE BUSCA: " + df[SEARCHABLE_TERMS_COLUMN]
    )
    
    # --- FIM DA LÓGICA DE ADAPTAÇÃO ---
    
    # Processa em batches para respeitar limites da API e da memória (Pinecone aceita até 100 vetores por upsert)
    BATCH_SIZE = 50 
    total_vectors_uploaded = 0
    
    for i in range(0, len(df), BATCH_SIZE):
        batch = df.iloc[i:i + BATCH_SIZE]
        ids = batch[ID_COLUMN].astype(str).tolist()
        texts = batch['TEXT_TO_EMBED'].tolist()
        
        # 1. Gerar embeddings
        vectors = generate_embeddings(texts)
        
        # 2. Preparar metadados
        # Salvamos todas as colunas de dados do procedimento (TUSS, LOINC, Descrição)
        # exceto a que usamos para gerar o texto, que pode ser descartada.
        metadata_columns = [col for col in df.columns if col not in ['TEXT_TO_EMBED']]
        
        # Cria a lista de dicionários de metadados
        metadatas = batch[metadata_columns].to_dict('records')

        # 3. Preparar o payload para o upsert (ID, Vetor, Metadado)
        to_upsert = list(zip(ids, vectors, metadatas))
        
        # 4. Upsert
        index.upsert(vectors=to_upsert)
        
        total_vectors_uploaded += len(to_upsert)
        print(f"Batch {i//BATCH_SIZE + 1} enviado. Total de vetores: {total_vectors_uploaded}")
        
    print(f"\n--- Ingestão Completa. {total_vectors_uploaded} vetores em {INDEX_NAME} ---")

if __name__ == "__main__":
    ingest_data_to_pinecone()