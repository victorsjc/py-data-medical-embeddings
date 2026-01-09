import pandas as pd
import medspacy
import hashlib
import re
import json
import spacy
import unicodedata
from spacy.tokens import Span
from typing import Dict, Any, List
from medspacy.ner import TargetMatcher, TargetRule

# ==============================================================================
# 0. CONFIGURAÇÃO E FUNÇÕES DE PRÉ-PROCESSAMENTO (REUTILIZADAS)
# ==============================================================================

# Definições de Domínio para Semantic Chunking (Solução 1)
PALAVRAS_STOPWORDS_CLINICAS = {
    'teste', 'exame', 'procedimento', 'dosagem', 'dosar', 'quantificacao', 
    'determinacao', 'analise', 'completo', 'total', 'parcial', 'serico', 
    'plasmatico', 'qualitativo', 'quantitativo', 'padrao', 'automatizado',
    'por', 'de', 'do', 'da', 'no', 'na', 'e', 'ou' 
}

MAPA_SINONIMOS_DOMINIO = {
    'glicemia': 'glicose',
    'acucar': 'glicose',
    'soro': 'plasma',
    'total': 'completo',
    'quimioluminescencia': 'imunoensaio',
    'enzimatico': 'metodo-enzima'
}

nlp = spacy.load("pt_core_news_sm", disable={"ner", "parser"})
nlp = medspacy.load(nlp)

# Registrar extensões customizadas para os atributos das TargetRules
# Isso é necessário para que o MedSpaCy possa atribuir valores a esses atributos
custom_extensions = ['loinc_code', 'property', 'system', 'qualifiers']
for ext in custom_extensions:
    if not Span.has_extension(ext):
        Span.set_extension(ext, default=None)


def load_target_rules_from_jsonl(jsonl_path: str) -> List[TargetRule]:
    """
    Carrega TargetRules a partir de um arquivo JSONL (uma regra JSON por linha).
    
    Args:
        jsonl_path: Caminho para o arquivo JSONL
        
    Returns:
        Lista de objetos TargetRule
    """
    target_rules = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rule_data = json.loads(line)
                target_rules.append(
                    TargetRule(
                        literal=rule_data.get('literal'),
                        category=rule_data.get('category', 'PROC'),
                        pattern=rule_data.get('pattern'),
                        attributes=rule_data.get('attributes', {})
                    )
                )
    return target_rules


def normalizar_termo(termo: str) -> str:
    """Aplica o mapeamento de sinônimos e padronização básica."""
    termo = str(termo).lower().strip()
    termo = re.sub(r'[^a-z0-9\s]', '', termo) # Remove caracteres especiais
    for sin, padrao in MAPA_SINONIMOS_DOMINIO.items():
        termo = termo.replace(sin, padrao)
    return termo

def preparar_string_para_embedding(componente: str, sistema: str, propriedade: str, metodo: str = None) -> str:
    """
    Limpa e concatena os campos essenciais para otimizar o Embedding (Semantic Chunking).
    Usamos o Componente (Nome), o Sistema (Coleta) e a Propriedade (o que você inferia da unidade).
    """
    partes_essenciais = []
    
    # 1. Componente (Nome) - O mais importante
    componente_limpo = normalizar_termo(componente)
    palavras = componente_limpo.split()
    #palavras_filtradas = [
    #    p for p in palavras 
    #    if p not in PALAVRAS_STOPWORDS_CLINICAS and len(p) > 2
    #]
    #if palavras_filtradas:
    #    partes_essenciais.append(" ".join(palavras_filtradas))
    if palavras:
        partes_essenciais.append(" ".join(palavras[0]))
    
    # 2. Sistema (Coleta)
    #sistema_limpo = normalizar_termo(sistema)
    #if sistema_limpo and sistema_limpo != 'paciente': # Ignorar 'paciente' que é genérico
    #    partes_essenciais.append(sistema_limpo)
        
    # 3. Propriedade (O que diferencia a unidade)
    #propriedade_limpa = normalizar_termo(propriedade)
    #if propriedade_limpa:
    #    partes_essenciais.append(propriedade_limpa)
    
    # 4. Método (Opcional, se conseguirmos extrair dos RELATEDNAMES2)
    #if metodo:
    #     metodo_limpo = normalizar_termo(metodo)
    #     if metodo_limpo and metodo_limpo not in PALAVRAS_STOPWORDS_CLINICAS:
    #         partes_essenciais.append(metodo_limpo)

    # Concatenação final
    return ", ".join(partes_essenciais)

def normalizar_para_hash(texto: str) -> str:
    """
    Remove acentos e converte para minusculo.
    Também normaliza quebras de linha e outros caracteres de espaçamento.
    """
    if not texto: return ""
    
    # Primeiro, substitui quebras de linha e tabs por espaços
    texto = re.sub(r'[\n\r\t]+', ' ', str(texto))
    
    # Remove acentos
    norm = unicodedata.normalize('NFKD', texto).encode('ASCII', 'ignore').decode('ASCII')
    
    # Normaliza espaços múltiplos para um único espaço
    texto_limpo = re.sub(r'\s+', ' ', norm)
    
    return texto_limpo.lower().strip()

def gerar_hash_master(componente: str, sistema: str, propriedade: str) -> str:
    """
    Gera o Hash Determinístico (Fingerprint) para o Cache Rápido.
    Usamos apenas Componente, Sistema e Propriedade (substituindo a unidade).
    Agora normalizado (sem acento, minusculo).
    """
    # É fundamental que esta lógica seja o mais limpa e consistente possível.
    #string_base = f"{normalizar_termo(componente)}|{normalizar_termo(sistema)}|{normalizar_termo(propriedade)}"
    string_base = componente
    string_norm = normalizar_para_hash(string_base)
    
    hash_obj = hashlib.md5(string_norm.encode('utf-8'))
    return hash_obj.hexdigest()

# ==============================================================================
# 1. EXECUÇÃO PRINCIPAL
# ==============================================================================

def criar_base_mestra_inicial(caminho_arquivo: str) -> pd.DataFrame:
    """
    Carrega o LOINC, processa e gera a Base Mestra com MK, Hash e String de Embedding.
    """
    print(f"Lendo arquivo: {caminho_arquivo}")
    try:
        # Carrega o CSV. Assumindo que o delimitador é vírgula (padrão)
        df = pd.read_csv(caminho_arquivo, dtype=str, keep_default_na=False)
    except FileNotFoundError:
        print(f"ERRO: Arquivo não encontrado em {caminho_arquivo}")
        return pd.DataFrame()

    # Carregar regras do arquivo JSONL
    print("Carregando regras do arquivo JSONL...")
    target_rules = load_target_rules_from_jsonl("./target_rules_loinc.jsonl")
    print(f"✓ {len(target_rules)} regras carregadas")
    
    target_matcher = nlp.get_pipe("medspacy_target_matcher")
    target_matcher.add(target_rules)

    # Renomeia colunas para facilitar o uso
    df.columns = [col.upper() for col in df.columns]
    
    # Garantir colunas essenciais (em maiúsculas após transformação na linha anterior)
    cols_necessarias = ['EXAME', 'LOINC', 'SISTEMA', 'SINONIMOS', 'TEMPO/PREPARO (AJUSTADO)', 
                        'UNIDADE DE MEDIDA', 'MÉTODO', 'AJUSTE TÉCNICO & VISÃO SAÚDE PÚBLICA', 
                        'CATEGORIA', 'STATUS']
    if not all(col in df.columns for col in cols_necessarias):
        print(f"ERRO: O arquivo não contém todas as colunas necessárias: {cols_necessarias}")
        return pd.DataFrame()

    registros_processados = []
    cache_componentes: Dict[str, Dict[str, Any]] = {}  # Cache em memória: componente_normalizado -> dados
    
    total_rows = len(df)
    print(f"Processando {total_rows} registros...")
    
    for index, row in df.iterrows():
        if index % 5000 == 0:
            print(f"  Processado: {index}/{total_rows}")
        
        loinc_num = row['LOINC']
        componente_original = row['EXAME']  # Componente SEM tratamento
        
        # 1. Processar com NLP para identificar entidades PROC
        doc = nlp(componente_original)
        
        # Filtrar apenas entidades com label PROC
        entidades_proc = [ent.text for ent in doc.ents if ent.label_ == 'PROC']
        
        # 2. Determinar o componente normalizado baseado nas entidades encontradas
        if len(entidades_proc) > 1:
            # Múltiplas entidades PROC: unificar separadas por espaço
            componente = ' '.join(entidades_proc)
        elif len(entidades_proc) == 1:
            # Uma única entidade PROC: usar apenas ela
            componente = entidades_proc[0]
        else:
            # Nenhuma entidade PROC encontrada: usar o componente original inteiro
            componente = componente_original
        
        # Verifica se o componente normalizado já existe no cache
        # Se existir, pula para o próximo registro (não inclui duplicatas no cadastro final)
        if componente in cache_componentes:
            continue
        
        # Processamento normal para componentes novos
        # IMPORTANTE: SISTEMA contém o material (Sangue, Urina, etc)
        # UNIDADE DE MEDIDA contém a propriedade/unidade (mg/dL, U/L, etc)
        sistema = row['SISTEMA']  # Material de coleta -> LOINC_SYSTEM
        propriedade = row['UNIDADE DE MEDIDA']  # Unidade -> LOINC_PROPERTY
        sinonimos = row['SINONIMOS']
        
        # 1. Geração da Master Key (MK)
        # Aqui, cada código LOINC é tratado como uma Master Key única.
        master_key = f"MK-LOINC-{loinc_num}"
        
        # 2. Geração do Hash Determinístico (Fingerprint para o Cache Rápido)
        # Usando o componente normalizado (baseado nas entidades PROC)
        hash_determinista = gerar_hash_master(componente_original, sistema, propriedade)
        
        # 3. Geração da String de Embedding (Semantic Chunking)
        string_embedding = preparar_string_para_embedding(componente, sistema, propriedade)

        novo_registro = {
            'MASTER_KEY_ID': master_key,
            'LOINC_NUM': loinc_num,
            'NOME_PREFERENCIAL_LOINC': componente,
            'COMPONENTE_ORIGINAL': componente_original,  # Manter original para referência
            'SINONIMOS': sinonimos,
            'ENTIDADES_PROC': ';'.join(entidades_proc) if entidades_proc else '',
            'QTD_ENTIDADES_PROC': len(entidades_proc),
            'LOINC_PROPERTY': propriedade,
            'LOINC_SYSTEM': sistema,
            'HASH_CACHE': hash_determinista,
            'STRING_EMBEDDING': string_embedding,
            'STATUS': 'GOLD_STANDARD'
        }
        
        # Adiciona ao cache para reutilização futura
        cache_componentes[componente] = novo_registro
        registros_processados.append(novo_registro)

    df_base_mestra = pd.DataFrame(registros_processados)
    
    # Remove duplicatas de Hash (ocorre se dois LOINCs têm o mesmo hash)
    duplicatas_hash = df_base_mestra['HASH_CACHE'].duplicated(keep=False)
    if duplicatas_hash.any():
        print(f"⚠️ Atenção: {duplicatas_hash.sum() // 2} Hashes duplicados encontrados (Ex: LOINCs com mesma Componente/Sistema/Propriedade).")
    
    print(f"Base Mestra inicial criada com sucesso. Total de {len(df_base_mestra)} registros.")
    return df_base_mestra

# ==============================================================================
# 2. EXECUÇÃO E SALVAMENTO
# ==============================================================================

if __name__ == '__main__':
    NOME_ARQUIVO_LOINC = 'base_medical_exam.csv' # Seu arquivo de entrada
    NOME_ARQUIVO_SAIDA = 'base_mestra_loinc_inicial.csv'
    
    df_base = criar_base_mestra_inicial(NOME_ARQUIVO_LOINC)
    
    if not df_base.empty:
        df_base.to_csv(NOME_ARQUIVO_SAIDA, index=False, encoding='utf-8')
        print("-" * 50)
        print(f"✅ Base Mestra Inicial salva como: '{NOME_ARQUIVO_SAIDA}'")
        print("Agora você pode usar a coluna 'STRING_EMBEDDING' para indexar no Pinecone")
        print("e a coluna 'HASH_CACHE' para popular seu banco de dados de Via Rápida.")