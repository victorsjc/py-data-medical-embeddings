import json
import time
import hashlib
import pandas as pd
import re
import unicodedata
from typing import List, Dict, Any, Tuple

# ==============================================================================
# 0. CONFIGURAÇÕES E DADOS
# ==============================================================================

# Arquivos de Entrada/Saída
#CAMINHO_JSON = 'unimed-exame-medico-casue.pdf_20251213_021301_codified.json'
#CAMINHO_JSON = '7508836_66269.31244.843496.50316204.pdf_20251209_023139_codified.json'
CAMINHO_JSON = 'exames_drajuliapitombo_anacarolina_codified.json'
CAMINHO_BASE_MESTRA = 'base_mestra_loinc_inicial.csv' # Arquivo com os Hashs mestres

# Simulação dos tempos de processamento
TEMPO_CACHE_HIT_MS = 5      # Via Rápida (Acerto Determinístico)
TEMPO_PINE_CONE_MISS_MS = 350 # Via Lenta (Consulta Vetorial)

# Dicionário que simula a tabela de Cache/Base Mestra em memória
CACHE_PRODUCAO_HASH_MK: Dict[str, str] = {} 
CACHE_SINONIMOS_HASH_MK: Dict[str, str] = {}

# Log detalhado de cada exame processado
LOG_EXAMES_DETALHADO: List[Dict[str, Any]] = []

# ==============================================================================
# 1. FUNÇÕES DE CARREGAMENTO E HASH
# ==============================================================================

def carregar_base_mestra_cache() -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Carrega a base mestra do LOINC e popula o dicionário de cache.
    Usa HASH_CACHE como chave e MASTER_KEY_ID como valor.
    Tambem popula o CACHE_SINONIMOS_HASH_MK.
    """
    try:
        df_base = pd.read_csv(CAMINHO_BASE_MESTRA, dtype=str, keep_default_na=False)
        
        if 'HASH_CACHE' not in df_base.columns or 'MASTER_KEY_ID' not in df_base.columns:
            print(f"ERRO: O arquivo {CAMINHO_BASE_MESTRA} deve conter as colunas 'HASH_CACHE' e 'MASTER_KEY_ID'.")
            return {}, {}
            
        # cache = dict(zip(df_base['HASH_CACHE'], df_base['MASTER_KEY_ID']))
        
        # Populando Cache Principal com Variacoes (Original e Normalizado)
        cache = {}
        # Garante que as colunas essenciais existem
        col_comp = 'COMPONENTE_ORIGINAL' if 'COMPONENTE_ORIGINAL' in df_base.columns else 'NOME_PREFERENCIAL_LOINC'
        
        for _, row in df_base.iterrows():
            mk_id = row['MASTER_KEY_ID']
            hash_oficial = row['HASH_CACHE']
            
            # Adiciona o hash oficial (vindo do CSV)
            cache[hash_oficial] = mk_id
            
            # Adiciona variacoes do componente original (Normalizado, Splits, etc)
            # Isso permite que se o input for "Ureia" e o original "Uréia", ambos gerem hashes conhecidos
            texto_base = str(row[col_comp])
            variacoes = gerar_variacoes_pesquisa(texto_base)
            
            for var in variacoes:
                h_var = gerar_hash_string(var)
                if h_var not in cache:
                    cache[h_var] = mk_id

        # Processamento de Sinonimos
        synonym_cache = {}
        if 'SINONIMOS' in df_base.columns:
            for _, row in df_base.iterrows():
                sinonimos_str = str(row['SINONIMOS'])
                master_key = row['MASTER_KEY_ID']
                
                if sinonimos_str and sinonimos_str.lower() != 'nan':
                    matchs = sinonimos_str.split(',')
                    for termo in matchs:
                        termo_limpo = termo.strip()
                        if termo_limpo:
                            # Gera variacoes do sinonimo (Original, Normalizado/Lower, Splits)
                            # Isso garante que match case-insensitive funcione para sinonimos tambem
                            variacoes_sin = gerar_variacoes_pesquisa(termo_limpo)
                            
                            for var_sin in variacoes_sin:
                                h_sin = gerar_hash_string(var_sin)
                                if h_sin not in synonym_cache:
                                    synonym_cache[h_sin] = master_key

        print(f"✅ Cache de Master Keys populado com {len(cache)} registros mestres.")
        print(f"✅ Cache de Sinonimos populado com {len(synonym_cache)} registros derivados.")
        
        return cache, synonym_cache
        
    except FileNotFoundError:
        print(f"ERRO: Arquivo da Base Mestra não encontrado em {CAMINHO_BASE_MESTRA}. O Cache de Produção estará vazio.")
        return {}, {}
    except Exception as e:
        print(f"ERRO ao carregar a Base Mestra: {e}")
        return {}, {}

def carregar_dados_exames(caminho_arquivo: str) -> List[Dict[str, Any]]:
    """Carrega o arquivo JSON e retorna a lista de exames."""
    try:
        with open(caminho_arquivo, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        analysis_data = json.loads(data['analysis']) if isinstance(data['analysis'], str) else data['analysis']
        return analysis_data.get('exames', [])
    except Exception as e:
        print(f"ERRO ao carregar ou decodificar JSON: {e}")
        return []

def normalizar_termo(termo: str) -> str:
    """Normaliza o termo para garantir consistência no Hash (Legado)."""
    return str(termo).lower().strip().replace('slide - química seca', 'quimica_seca')

def normalizar_texto_completo(texto: str) -> str:
    """
    Remove acentuação, converte para minúsculas e remove espaços extras.
    """
    if not texto: return ""
    # Remove acentos
    texto_sem_acento = unicodedata.normalize('NFKD', str(texto)).encode('ASCII', 'ignore').decode('ASCII')
    return texto_sem_acento.lower().strip()

def gerar_hash_string(texto: str) -> str:
    """Gera o MD5 de uma string simples."""
    if not texto: return ""
    return hashlib.md5(texto.encode('utf-8')).hexdigest()

def gerar_ngrams(tokens: List[str], n: int) -> List[str]:
    """Gera n-grams de uma lista de tokens."""
    if len(tokens) < n:
        return []
    return [' '.join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

def gerar_variacoes_pesquisa(texto: str) -> List[str]:
    """
    Gera variações do termo para tentativa de match.
    Ordem:
    1. Texto Original (limpo)
    2. Texto Normalizado (sem acentos/lower)
    3. Texto sem modificadores conhecidos (Alta Sensibilidade -> Metodo)
    4. Acronimos em parenteses (TGO)
    5. N-Grams (Bigramas, Trigramas)
    6. Tokens individuais
    """
    candidatos = []
    
    # 0. Base cleanup
    if not texto: return []
    texto_original = texto.strip()
    texto_lower = texto_original.lower()
    
    # 1. Original
    candidatos.append(texto_lower)
    
    # 2. Normalizado Completo (sem acentos)
    norm = normalizar_texto_completo(texto_lower)
    if norm and norm not in candidatos:
        candidatos.append(norm)

    # 3. Tratamento de Parenteses (Extrair TGO, TGP)
    match_parenteses = re.search(r'\((.*?)\)', texto_original)
    if match_parenteses:
        conteudo = match_parenteses.group(1)
        candidatos.append(conteudo.lower().strip())
        var_norm = normalizar_texto_completo(conteudo)
        if var_norm not in candidatos:
            candidatos.append(var_norm)
        
        texto_sem_parenteses = texto_original.replace(f"({conteudo})", "").strip()
        candidatos.append(texto_sem_parenteses.lower())
        candidatos.append(normalizar_texto_completo(texto_sem_parenteses))

    # 4. Tratamento de Modificadores
    termos_metodo = [
        'alta sensibilidade', 'ultra sensivel', 'ultra-sensivel', 'automatizado', 
        'dosagem', 'quantificacao', 'determinacao', 'analise', 'confirmacao',
        'h.p.l.c', 'hplc', 'quimioluminescencia', 'eletroquimioluminescencia', 
        'imunoturbidimetria', 'nefelometria', 'colorimetrico', 'cinetico',
        'elisa', 'enzimatico', 'por', 'soro', 'plasma', 'basal', 'total', 
        'livre', 'indireta', 'direta', 'reverso', 'fracoes',
        'hormonio', 'vitamina', 'de', 'e', 'para' # Stopwords comuns em nomes de exames
    ]
    
    # Remove modificadores do texto normalizado
    texto_limpo_metodos = norm
    alterado = False
    
    # Ordena modificadores por tamanho para remover os maiores primeiro
    termos_metodo.sort(key=len, reverse=True)
    
    for termo in termos_metodo:
        # Usa regex para remover palavra exata evitando remover partes de palavras (ex: 'de' em 'desidrogenase')
        pattern = r'\b' + re.escape(termo) + r'\b'
        if re.search(pattern, texto_limpo_metodos):
            texto_limpo_metodos = re.sub(pattern, " ", texto_limpo_metodos).strip()
            alterado = True
            
    if alterado and texto_limpo_metodos:
        # Remove espaços duplos
        texto_limpo_metodos = re.sub(r'\s+', ' ', texto_limpo_metodos).strip()
        candidatos.append(texto_limpo_metodos)

    # 5. Splits e N-Grams
    # Tokeniza por qualquer separador comum
    tokens = re.split(r'[;+/\n\r\s-]+', norm)
    tokens = [t.strip() for t in tokens if t.strip()]
    
    if len(tokens) > 1:
        # Trigramas
        trigrams = gerar_ngrams(tokens, 3)
        candidatos.extend(trigrams)
        
        # Bigramas
        bigrams = gerar_ngrams(tokens, 2)
        candidatos.extend(bigrams)
        
        # Unigramas (Tokens individuais)
        # Prioriza tokens maiores (> 1 char) e remove stopwords 'de', 'e' se ficarem sozinhas
        for t in tokens:
            if len(t) > 1 and t not in ['de', 'da', 'do', 'em']:
                candidatos.append(t)
                
    # 6. Específicos para os casos relatados (Heurística de fallback)
    # Se contiver 'eletroforese', tenta 'eletroforese' isolado se não estiver
    if 'eletroforese' in norm:
        candidatos.append('eletroforese')
        
    # Remove duplicatas preservando ordem
    final_candidatos = []
    seen = set()
    for c in candidatos:
        c_clean = c.strip()
        if c_clean and c_clean not in seen:
            final_candidatos.append(c_clean)
            seen.add(c_clean)
            
    return final_candidatos

def gerar_hash_determinista(exame: Dict[str, Any]) -> str:
    """
    Mantido para retrocompatibilidade, mas agora usa apenas o procedimento.
    """
    componente_original = exame.get('procedimento', '')
    return gerar_hash_string(componente_original)

# ==============================================================================
# 2. SIMULAÇÃO DO WORKFLOW E GERAÇÃO DE LOGS
# ==============================================================================

def simular_processamento_hibrido(exames_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Simula o workflow Cache -> Pinecone (Feedback Loop) para cada exame, gerando logs.
    """
    global CACHE_PRODUCAO_HASH_MK
    global CACHE_SINONIMOS_HASH_MK
    global LOG_EXAMES_DETALHADO
    
    resultados = []
    
    for i, exame in enumerate(exames_data):
        
        # Hash "Principal" (apenas para registro inicial, pode mudar se encontrarmos match em variação)
        hash_inicial = gerar_hash_determinista(exame)
        exame['hash_id'] = hash_inicial 
        
        procedimento_original = exame.get('procedimento', '')
        
        # Log inicial
        log_entry = {
            'id_exame': i + 1,
            'procedimento': procedimento_original,
            'material': exame.get('material'),
            'hash_id_curto': hash_inicial[:8],
            'decisao': 'PENDING',
            'mk_atribuida': 'N/A',
            'tempo_simulado_ms': 0
        }
        
        match_encontrado = False
        mk_encontrada = ""
        motivo_match = ""
        
        # Gera lista de tentativas
        candidatos = gerar_variacoes_pesquisa(procedimento_original)
        
        for candidato in candidatos:
            hash_candidato = gerar_hash_string(candidato)
            
            # 1. Check Base Mestra
            if hash_candidato in CACHE_PRODUCAO_HASH_MK:
                mk_encontrada = CACHE_PRODUCAO_HASH_MK[hash_candidato]
                motivo_match = f"MATCH DETERMINISTICO ({'ORIGINAL' if candidato == procedimento_original.strip() else 'NORMALIZADO/SPLIT'})"
                match_encontrado = True
                exame['hash_id'] = hash_candidato # Atualiza para o hash que deu match
                break
                
            # 2. Check Sinonimos
            if hash_candidato in CACHE_SINONIMOS_HASH_MK:
                mk_encontrada = CACHE_SINONIMOS_HASH_MK[hash_candidato]
                motivo_match = f"MATCH SINONIMO ({'ORIGINAL' if candidato == procedimento_original.strip() else 'NORMALIZADO/SPLIT'})"
                match_encontrado = True
                exame['hash_id'] = hash_candidato
                break
        
        if match_encontrado:
             # CACHE HIT
            exame['tempo_simulado_ms'] = TEMPO_CACHE_HIT_MS
            exame['decisao'] = motivo_match
            exame['mk_atribuida'] = mk_encontrada
            
            log_entry.update({
                'decisao': motivo_match,
                'mk_atribuida': mk_encontrada,
                'tempo_simulado_ms': TEMPO_CACHE_HIT_MS,
                'hash_id_curto': exame['hash_id'][:8]
            })
            
        else:
            # CACHE MISS (Via Lenta: Consulta ao Pinecone)
            
            # Simula a atribuição da MK pelo Pinecone + Lógica de filtro + Aprendizado
            # Atribuímos uma MK temporária para simular o resultado do aprendizado
            temp_mk_id = f"MK-NOVA-{hash_inicial[:8]}" 
            
            exame['tempo_simulado_ms'] = TEMPO_PINE_CONE_MISS_MS
            exame['decisao'] = "PINE_CONE_MISS" 
            exame['mk_atribuida'] = temp_mk_id
            
            log_entry.update({
                'decisao': 'PINE CONE MISS',
                'mk_atribuida': temp_mk_id,
                'tempo_simulado_ms': TEMPO_PINE_CONE_MISS_MS
            })
            
        resultados.append(exame)
        LOG_EXAMES_DETALHADO.append(log_entry)
    
    return resultados

# ==============================================================================
# 3. CÁLCULO DAS MÉTRICAS DE PERFORMANCE
# ==============================================================================

def analisar_performance(resultados: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calcula e retorna as métricas de performance baseadas no Hash Hit Rate."""
    
    total_exames = len(resultados)
    
    tempos = [r['tempo_simulado_ms'] for r in resultados]
    
    acertos_hash = sum(1 for r in resultados if "MATCH DETERMINISTICO" in r['decisao'])
    acertos_sinonimo = sum(1 for r in resultados if "MATCH SINONIMO" in r['decisao'])
    cache_misses = total_exames - (acertos_hash + acertos_sinonimo)
    
    assertividade_percentual = ((acertos_hash + acertos_sinonimo) / total_exames) * 100
    falsos_positivos = 0 
    
    tempo_total_ms = sum(tempos)
    tempo_medio_ms = tempo_total_ms / total_exames
    
    relatorio = {
        "configuracao_teste": {
            "total_exames_analisados": total_exames,
            "tamanho_cache_base_mestra": len(CACHE_PRODUCAO_HASH_MK),
            "tamanho_cache_sinonimos": len(CACHE_SINONIMOS_HASH_MK)
        },
        "metricas_performance": {
            "assertividade_percentual": f"{assertividade_percentual:.2f}%",
            "quantidade_acertos_deterministicos": acertos_hash,
            "quantidade_acertos_sinonimos": acertos_sinonimo,
            "quantidade_revisoes_e_novos_casos": cache_misses,
            "quantidade_falsos_positivos_simulado": falsos_positivos,
            "tempo_processamento_medio_ms": f"{tempo_medio_ms:.2f} ms",
            "tempo_processamento_total_segundos": f"{tempo_total_ms / 1000:.2f} s"
        }
    }
    return relatorio

# ==============================================================================
# 4. EXECUÇÃO DA APLICAÇÃO
# ==============================================================================

if __name__ == '__main__':
    
    # 1. Carrega a base mestra (Simula o DB de Produção)
    CACHE_PRODUCAO_HASH_MK, CACHE_SINONIMOS_HASH_MK = carregar_base_mestra_cache()

    # 2. Carrega os dados para análise
    exames_data = carregar_dados_exames(CAMINHO_JSON)
    
    if not exames_data:
        print("Finalizando execução devido a erro no carregamento dos dados.")
    else:
        print(f"\nIniciando simulação com {len(exames_data)} requisições de exames.")
        
        # 3. Simula o processamento e gera os logs
        resultados_processados = simular_processamento_hibrido(exames_data)
        
        # 4. Executa a análise
        relatorio_final = analisar_performance(resultados_processados)
        
        # 5. Exibe os logs detalhados
        print("\n" + "="*80)
        print("📝 LOG DETALHADO POR EXAME")
        print("="*80)
        
        # Formatação do Log
        log_df = pd.DataFrame(LOG_EXAMES_DETALHADO)
        # Reordenar colunas para melhor visualização
        log_df = log_df[['id_exame', 'procedimento', 'material', 'hash_id_curto', 'decisao', 'mk_atribuida', 'tempo_simulado_ms']]
        print(log_df.to_string(index=False, max_rows=100))
        print("="*80)
        
        # 6. Exibe o relatório de performance
        print("\n" + "="*60)
        print("📊 RELATÓRIO DE ANÁLISE DE PERFORMANCE (HASH/CACHE BASED)")
        print("="*60)
        print(json.dumps(relatorio_final, indent=4))
        print("="*60)