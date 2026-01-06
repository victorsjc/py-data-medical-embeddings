"""
Sistema de Processamento de Exames Médicos
Gerador de TargetRules JSON a partir de CSV com sinônimos
"""

import pandas as pd
import json
from typing import List, Dict, Set
import re
import nltk

# Garantir que o corpus de stopwords está disponível
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)


class TargetRulesGenerator:
    """Gera arquivo JSON de TargetRules para MedSpaCy a partir de CSV"""
    
    def __init__(self, csv_path: str):
        """
        Inicializa o gerador com arquivo CSV
        
        Args:
            csv_path: Caminho para o arquivo CSV com componentes e sinônimos
        """
        self.df = pd.read_csv(csv_path)
        self.target_rules = []


class TargetRulesGeneratorLOINC:
    """
    Gera arquivo JSONL de TargetRules para MedSpaCy a partir do CSV loinc-processed-ptBR.csv
    
    O CSV esperado deve ter as colunas:
    - LOINC_NUM: Código LOINC
    - COMPONENT: Nome do componente/procedimento
    - PROPERTY: Propriedade do exame
    - SYSTEM: Sistema/amostra biológica
    - RELATEDNAMES2: Nomes relacionados/sinônimos separados por ';'
    """
    
    # Stopwords - termos descritivos que não são nomes de exames
    STOPWORDS = {
        # Termos temporais/descritivos em inglês (do RELATEDNAMES2)
        'point in time', 'random', 'quantitative', 'qualitative',
        'qnt', 'quant', 'quan', 'ql', 'ord', 'ordinal',
        'nominal', 'narrative', 'finding', 'findings',
        'chemistry', 'drug doses', 'drug/tox', 'drug',
        'observation', 'survey', 'document', 'attachment',
        # Termos em português
        'amostra', 'após', 'antes', 'durante', 'min', 'hora', 'horas',
        'dia', 'dias', 'semana', 'semanas', 'mês', 'meses',
        'administração', 'dose', 'ev', 'im', 'vo', 'sc',
        'primeira', 'segunda', 'terceira', 'quarta', 'quinta',
        '1ª', '2ª', '3ª', '4ª', '5ª', '6ª', '7ª', '8ª', '9ª', '10ª',
        # Números ordinais extensos
        'ª amostra', 'amostra às',
    }
    
    # Qualificadores importantes que devem ser preservados (relacionados a exames)
    IMPORTANT_QUALIFIERS = {
        'ige', 'igg', 'iga', 'igm', 'igd',  # Imunoglobulinas
        'ac', 'ab', 'ag',  # Anticorpo/Antígeno
        'anti', 'auto',  # Prefixos de anticorpos
        'total', 'livre', 'direto', 'indireto',  # Qualificadores de dosagem
        'fração', 'fracao', 'subclasse',
        'cadeia', 'cadeias',
        'reagente', 'específico', 'especifico',
        'recombinante', 'nativo',
    }
    
    def __init__(self, csv_path: str):
        """
        Inicializa o gerador com arquivo CSV do LOINC
        
        Args:
            csv_path: Caminho para o arquivo CSV loinc-processed-ptBR.csv
        """
        self.df = pd.read_csv(csv_path)
        self.target_rules = []
        self.processed_literals = set()  # Para evitar duplicatas
        
        # Carregar stopwords do NLTK para português (exceto 'não')
        try:
            portuguese_stopwords = nltk.corpus.stopwords.words("portuguese")
            self._nltk_stopwords = set(filter(lambda sw: sw != 'não', portuguese_stopwords))
        except Exception:
            self._nltk_stopwords = set()
    
    def _clean_string(self, text: str, remove_stopwords: bool = True) -> str:
        """
        Limpa e normaliza uma string seguindo o padrão de normalização.
        
        Operações realizadas:
        - Converte para minúsculas
        - Remove quebras de linha
        - Remove espaços múltiplos
        - Remove underscores
        - Remove stopwords em português (exceto 'não')
        
        Args:
            text: Texto a ser limpo
            remove_stopwords: Se True, remove stopwords do NLTK
            
        Returns:
            Texto limpo e normalizado
        """
        if not text:
            return ""
        
        # Converter para minúsculas
        text = text.lower()
        
        # Remover quebras de linha
        text = re.sub(r'\n', ' ', text)
        
        # Remover espaços múltiplos
        text = re.sub(r'\s+', ' ', text)
        
        # Remover underscores
        text = re.sub(r'\_', '', text)
        
        # Remover stopwords em português (opcional)
        if remove_stopwords and self._nltk_stopwords:
            words = text.split()
            text_filtered = [word for word in words if word not in self._nltk_stopwords]
            text = ' '.join(text_filtered)
        
        return text.strip()
    
    def _extract_main_component(self, component: str) -> tuple:
        """
        Extrai o componente principal e qualificadores importantes
        
        Separadores: ^, ., ; (nesta ordem de prioridade)
        
        Args:
            component: Nome completo do componente
            
        Returns:
            tuple: (componente_principal, lista_de_qualificadores_importantes)
        """
        # Primeiro, separar por ^ (qualificadores temporais/condicionais)
        parts = re.split(r'\^', component)
        main_part = parts[0].strip()
        qualifiers_after_caret = parts[1:] if len(parts) > 1 else []
        
        # Depois, verificar se há qualificadores após . no nome principal
        dot_parts = re.split(r'\.', main_part)
        main_name = dot_parts[0].strip()
        qualifiers_after_dot = dot_parts[1:] if len(dot_parts) > 1 else []
        
        # Coletar qualificadores importantes
        important_quals = []
        
        # Verificar qualificadores após o ponto
        for qual in qualifiers_after_dot:
            qual_lower = qual.lower().strip()
            # Verificar se contém algum qualificador importante
            for imp_qual in self.IMPORTANT_QUALIFIERS:
                if imp_qual in qual_lower:
                    important_quals.append(qual.strip())
                    break
        
        # Se houver qualificadores importantes após o ponto, incluí-los no nome
        if important_quals:
            main_name = main_name + '.' + '.'.join(important_quals)
        
        return main_name, qualifiers_after_caret
    
    def _normalize_text(self, text: str, apply_stopwords: bool = False) -> str:
        """
        Normaliza o texto removendo caracteres especiais desnecessários.
        
        Para o literal (nome principal): NÃO remove stopwords (preserva o nome original)
        Para validação/comparação: pode remover stopwords
        
        Args:
            text: Texto a ser normalizado
            apply_stopwords: Se True, aplica remoção de stopwords NLTK
            
        Returns:
            Texto normalizado
        """
        if not text:
            return ""
        
        # Remover quebras de linha
        text = re.sub(r'\n', ' ', text)
        
        # Remover espaços múltiplos
        text = re.sub(r'\s+', ' ', text)
        
        # Remover underscores
        text = re.sub(r'\_', '', text)
        
        # Trim
        text = text.strip()
        
        return text
    
    def _is_valid_literal(self, text: str) -> bool:
        """
        Verifica se o texto é um literal válido para o mapeamento
        
        Args:
            text: Texto a ser validado
            
        Returns:
            True se válido, False caso contrário
        """
        if not text or text == 'nan':
            return False
        
        text_lower = text.lower().strip()
        
        # Verificar se é uma stopword completa (lista customizada)
        if text_lower in self.STOPWORDS:
            return False
        
        # Verificar se é apenas uma stopword do NLTK
        if text_lower in self._nltk_stopwords:
            return False
        
        # Verificar se começa com número ordinal seguido de "amostra"
        if re.match(r'^\d+[ªº]?\s*(amostra|sample)', text_lower):
            return False
        
        # Verificar se é apenas números/datas/horas
        if re.match(r'^[\d\s\-:\/]+$', text):
            return False
        
        # Muito curto (menos de 2 caracteres, exceto siglas conhecidas)
        if len(text) < 2:
            return False
        
        # Verificar se contém apenas palavras de stopwords (customizadas)
        words = text_lower.split()
        if all(w in self.STOPWORDS or w.isdigit() for w in words):
            return False
        
        # Verificar se após remover todas as stopwords (NLTK + custom), sobra conteúdo válido
        cleaned = self._clean_string(text, remove_stopwords=True)
        if not cleaned or len(cleaned) < 2:
            return False
        
        return True
    
    def _is_valid_synonym(self, text: str) -> bool:
        """
        Verifica se o sinônimo é válido (mais restritivo que literais)
        
        Args:
            text: Texto a ser validado
            
        Returns:
            True se válido, False caso contrário
        """
        if not self._is_valid_literal(text):
            return False
        
        text_lower = text.lower().strip()
        
        # Lista de padrões de sinônimos descritivos a serem excluídos
        invalid_patterns = [
            r'^point\s+in\s+time',
            r'^random$',
            r'^quantitative$',
            r'^qualitative$',
            r'^\d+\s*(min|hour|day|week|month)',
            r'^(first|second|third|fourth|fifth)',
            r'^sample\s+at',
            r'^post\s+',
            r'^pre\s+',
        ]
        
        for pattern in invalid_patterns:
            if re.match(pattern, text_lower):
                return False
        
        return True
    
    def generate_rules(self) -> List[Dict]:
        """
        Gera lista de regras a partir do CSV LOINC com normalização
        
        Returns:
            Lista de dicionários com regras formatadas
        """
        self.target_rules = []
        self.processed_literals = set()
        
        for idx, row in self.df.iterrows():
            loinc_code = str(row.get('LOINC_NUM', '')).strip()
            component = str(row.get('COMPONENT', '')).strip()
            property_val = str(row.get('PROPERTY', '')).strip() if pd.notna(row.get('PROPERTY')) else ''
            system = str(row.get('SYSTEM', '')).strip() if pd.notna(row.get('SYSTEM')) else ''
            
            if not component or component == 'nan':
                continue
            
            # Extrair componente principal e qualificadores
            main_component, qualifiers = self._extract_main_component(component)
            main_component = self._normalize_text(main_component)
            
            if not self._is_valid_literal(main_component):
                continue
            
            # Atributos do exame
            attributes = {
                "loinc_code": loinc_code
            }
            if property_val and property_val != 'nan':
                attributes['property'] = property_val
            if system and system != 'nan':
                attributes['system'] = system
            if qualifiers:
                attributes['qualifiers'] = qualifiers
            
            # Chave única para evitar duplicatas (componente + loinc)
            literal_key = (main_component.lower(), loinc_code)
            
            if literal_key not in self.processed_literals:
                self.processed_literals.add(literal_key)
                self.target_rules.append({
                    "literal": main_component,
                    "category": "PROC",
                    "pattern": None,
                    "attributes": attributes.copy()
                })
            
            # Processar sinônimos de RELATEDNAMES2
            if 'RELATEDNAMES2' in row and pd.notna(row['RELATEDNAMES2']):
                related_names = str(row['RELATEDNAMES2'])
                synonyms = [s.strip() for s in re.split(r'[;]', related_names) if s.strip()]
                
                for synonym in synonyms:
                    synonym = self._normalize_text(synonym)
                    
                    if not self._is_valid_synonym(synonym):
                        continue
                    
                    if synonym.lower() == main_component.lower():
                        continue
                    
                    synonym_key = (synonym.lower(), loinc_code)
                    
                    if synonym_key not in self.processed_literals:
                        self.processed_literals.add(synonym_key)
                        self.target_rules.append({
                            "literal": synonym,
                            "category": "PROC",
                            "pattern": None,
                            "attributes": attributes.copy()
                        })
        
        return self.target_rules
    
    def save_jsonl(self, output_path: str):
        """
        Salva regras em arquivo JSONL (uma regra JSON por linha)
        
        Args:
            output_path: Caminho para salvar o arquivo JSONL
        """
        if not self.target_rules:
            self.generate_rules()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for rule in self.target_rules:
                f.write(json.dumps(rule, ensure_ascii=False) + '\n')
        
        print(f"✓ Arquivo JSONL salvo: {output_path}")
        print(f"✓ Total de regras: {len(self.target_rules)}")
    
    def save_json(self, output_path: str, indent: int = 2):
        """
        Salva regras em arquivo JSON (array único)
        
        Args:
            output_path: Caminho para salvar o arquivo JSON
            indent: Indentação do JSON
        """
        if not self.target_rules:
            self.generate_rules()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.target_rules, f, ensure_ascii=False, indent=indent)
        
        print(f"✓ Arquivo JSON salvo: {output_path}")
        print(f"✓ Total de regras: {len(self.target_rules)}")
    
    def get_statistics(self) -> Dict:
        """Retorna estatísticas sobre as regras geradas"""
        if not self.target_rules:
            self.generate_rules()
        
        stats = {
            "total_rules": len(self.target_rules),
            "unique_components": len(self.df),
            "systems": {},
            "properties": {}
        }
        
        for rule in self.target_rules:
            if 'system' in rule['attributes']:
                sys = rule['attributes']['system']
                stats['systems'][sys] = stats['systems'].get(sys, 0) + 1
            if 'property' in rule['attributes']:
                prop = rule['attributes']['property']
                stats['properties'][prop] = stats['properties'].get(prop, 0) + 1
        
        return stats


class TargetRulesGeneratorOriginal(TargetRulesGenerator):
    """Extensão da classe base para CSV com formato original (component, synonyms, loinc_code, category)"""
    
    def generate_rules(self) -> List[Dict]:
        """
        Gera lista de regras a partir do CSV
        
        CSV deve ter as colunas:
        - component: Nome do componente/procedimento
        - synonyms: Sinônimos separados por ';' ou '|'
        - loinc_code (opcional): Código LOINC
        - category (opcional): Categoria do exame
        
        Returns:
            Lista de dicionários com regras formatadas
        """
        self.target_rules = []
        
        for idx, row in self.df.iterrows():
            component = str(row['component']).strip()
            
            # Processar sinônimos
            synonyms = []
            if 'synonyms' in row and pd.notna(row['synonyms']):
                synonyms_str = str(row['synonyms'])
                # Separar por ; ou |
                synonyms = [s.strip() for s in re.split(r'[;|]', synonyms_str) if s.strip()]
            
            # Atributos adicionais
            attributes = {}
            if 'loinc_code' in row and pd.notna(row['loinc_code']):
                attributes['loinc_code'] = str(row['loinc_code'])
            if 'category' in row and pd.notna(row['category']):
                attributes['category'] = str(row['category'])
            
            # Adicionar regra para o componente principal
            self.target_rules.append({
                "literal": component,
                "category": "PROC",
                "pattern": None,
                "attributes": attributes.copy()
            })
            
            # Adicionar regras para cada sinônimo
            for synonym in synonyms:
                if synonym and synonym != component:
                    self.target_rules.append({
                        "literal": synonym,
                        "category": "PROC",
                        "pattern": None,
                        "attributes": attributes.copy()
                    })
        
        return self.target_rules
    
    def save_json(self, output_path: str, indent: int = 2):
        """
        Salva regras em arquivo JSON
        
        Args:
            output_path: Caminho para salvar o arquivo JSON
            indent: Indentação do JSON
        """
        if not self.target_rules:
            self.generate_rules()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.target_rules, f, ensure_ascii=False, indent=indent)
        
        print(f"✓ Arquivo JSON salvo: {output_path}")
        print(f"✓ Total de regras: {len(self.target_rules)}")
    
    def get_statistics(self) -> Dict:
        """Retorna estatísticas sobre as regras geradas"""
        if not self.target_rules:
            self.generate_rules()
        
        stats = {
            "total_rules": len(self.target_rules),
            "unique_components": len(self.df),
            "categories": {}
        }
        
        for rule in self.target_rules:
            if 'category' in rule['attributes']:
                cat = rule['attributes']['category']
                stats['categories'][cat] = stats['categories'].get(cat, 0) + 1
        
        return stats


def create_sample_csv(filename: str = "exames_componentes.csv"):
    """Cria um CSV de exemplo com componentes e sinônimos"""
    
    data = {
        'component': [
            'Hemoglobina',
            'Glicose',
            'Creatinina',
            'Colesterol Total',
            'Triglicerídeos',
            'TSH',
            'Leucócitos',
            'Plaquetas',
            'Ureia',
            'ALT',
            'AST',
            'HDL Colesterol',
            'LDL Colesterol',
            'T4 Livre',
            'Ácido Úrico',
            'Bilirrubina Total',
            'Proteínas Totais',
            'Albumina',
            'Fosfatase Alcalina',
            'Gama GT',
            'Ferritina',
            'Vitamina D',
            'Vitamina B12',
            'Ácido Fólico',
            'PCR',
            'VHS',
            'Potássio',
            'Sódio',
            'Cálcio',
            'Magnésio'
        ],
        'synonyms': [
            'Hb;Hemoglobina sérica',
            'Glicemia;Glicose em jejum;Glicemia de jejum;Glucose',
            'Creatinina sérica;Creat',
            'Colesterol;Col total',
            'Triglicerides;TG;Triglicérides',
            'Hormônio tireoestimulante;Tireotrofina',
            'Glóbulos brancos;Leucograma;WBC',
            'Contagem de plaquetas;PLT',
            'Ureia sérica;BUN',
            'TGP;Alanina aminotransferase;ALAT',
            'TGO;Aspartato aminotransferase;ASAT',
            'HDL;Colesterol HDL;HDL-C',
            'LDL;Colesterol LDL;LDL-C',
            'Tiroxina livre;FT4',
            'Urato;Ácido úrico sérico',
            'Bilirrubina;BT',
            'Proteínas totais séricas;PT',
            'Albumina sérica;ALB',
            'Fosfatase alcalina sérica;FA;ALP',
            'Gama glutamil transferase;GGT;Gama-GT',
            'Ferritina sérica',
            'Vitamina D total;25-OH vitamina D;Calcidiol',
            'Cobalamina;Vit B12',
            'Folato;Ácido fólico sérico',
            'Proteína C reativa;PCR ultrassensível;CRP',
            'Velocidade de hemossedimentação;VS',
            'Potássio sérico;K',
            'Sódio sérico;Na',
            'Cálcio sérico;Ca',
            'Magnésio sérico;Mg'
        ],
        'loinc_code': [
            '718-7',
            '2345-7',
            '2160-0',
            '2093-3',
            '2571-8',
            '3016-3',
            '6690-2',
            '777-3',
            '3094-0',
            '1742-6',
            '1920-8',
            '2085-9',
            '13457-7',
            '3026-2',
            '3084-1',
            '1975-2',
            '2885-2',
            '1751-7',
            '6768-6',
            '2324-2',
            '2276-4',
            '1989-3',
            '2498-4',
            '2284-8',
            '1988-5',
            '30341-2',
            '2823-3',
            '2951-2',
            '17861-6',
            '2601-3'
        ],
        'category': [
            'Hematologia',
            'Bioquímica',
            'Bioquímica',
            'Lipidograma',
            'Lipidograma',
            'Endocrinologia',
            'Hematologia',
            'Hematologia',
            'Bioquímica',
            'Bioquímica',
            'Bioquímica',
            'Lipidograma',
            'Lipidograma',
            'Endocrinologia',
            'Bioquímica',
            'Bioquímica',
            'Bioquímica',
            'Bioquímica',
            'Bioquímica',
            'Bioquímica',
            'Hematologia',
            'Vitaminas',
            'Vitaminas',
            'Vitaminas',
            'Inflamação',
            'Inflamação',
            'Eletrólitos',
            'Eletrólitos',
            'Eletrólitos',
            'Eletrólitos'
        ]
    }
    
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False, encoding='utf-8')
    print(f"✓ CSV de exemplo criado: {filename}")
    return filename


# Exemplo de uso com MedSpaCy
def load_rules_in_medspacy(json_path: str):
    """
    Exemplo de como carregar as regras JSON no MedSpaCy
    
    Args:
        json_path: Caminho do arquivo JSON com regras
    """
    import medspacy
    from medspacy.ner import TargetRule
    
    # Carregar MedSpaCy
    nlp = medspacy.load()
    
    # Carregar regras do JSON
    with open(json_path, 'r', encoding='utf-8') as f:
        rules_data = json.load(f)
    
    # Converter para TargetRules
    target_rules = []
    for rule_data in rules_data:
        target_rules.append(
            TargetRule(
                literal=rule_data['literal'],
                category=rule_data['category'],
                pattern=rule_data['pattern'],
                attributes=rule_data['attributes']
            )
        )
    
    # Adicionar ao matcher
    target_matcher = nlp.get_pipe("medspacy_target_matcher")
    target_matcher.add(target_rules)
    
    print(f"✓ {len(target_rules)} regras carregadas no MedSpaCy")
    
    return nlp


def test_processor(nlp):
    """Testa o processador com texto de exemplo"""
    
    texto_teste = """
    EXAMES LABORATORIAIS
    
    Hemograma:
    - Hemoglobina: 14.2 g/dL
    - Leucócitos: 7500/mm³
    - PLT: 250.000/mm³
    
    Bioquímica:
    - Glicemia de jejum: 95 mg/dL
    - Creat: 1.0 mg/dL
    - TGP: 28 U/L
    - TGO: 24 U/L
    
    Lipidograma:
    - Colesterol total: 180 mg/dL
    - HDL-C: 55 mg/dL
    - LDL: 100 mg/dL
    - TG: 125 mg/dL
    
    Tireoide:
    - TSH: 2.1 µUI/mL
    - T4 livre: 1.2 ng/dL
    """
    
    doc = nlp(texto_teste)
    
    print("\n" + "="*60)
    print("ENTIDADES IDENTIFICADAS (PROC):")
    print("="*60)
    
    for ent in doc.ents:
        if ent.label_ == "PROC":
            attrs = ent._.get("attributes", {})
            print(f"\n📋 Texto: {ent.text}")
            print(f"   Posição: {ent.start_char}-{ent.end_char}")
            if 'loinc_code' in attrs:
                print(f"   LOINC: {attrs['loinc_code']}")
            if 'category' in attrs:
                print(f"   Categoria: {attrs['category']}")


# Script principal
if __name__ == "__main__":
    import os
    
    print("="*60)
    print("GERADOR DE TARGET RULES PARA MEDSPACY")
    print("="*60)
    
    # Verifica se o arquivo loinc-processed-ptBR.csv existe
    loinc_csv = "loinc-processed-ptBR.csv"
    
    if os.path.exists(loinc_csv):
        print(f"\n1. Usando arquivo LOINC existente: {loinc_csv}")
        
        # 2. Gerar JSONL de regras a partir do CSV LOINC
        print("\n2. Gerando regras a partir do CSV LOINC...")
        generator = TargetRulesGeneratorLOINC(loinc_csv)
        generator.generate_rules()
        
        # 3. Salvar JSONL
        jsonl_file = "target_rules_loinc.jsonl"
        generator.save_jsonl(jsonl_file)
        
        # 4. Estatísticas
        print("\n3. Estatísticas das regras:")
        stats = generator.get_statistics()
        print(f"   Total de regras: {stats['total_rules']}")
        print(f"   Componentes únicos no CSV: {stats['unique_components']}")
        print(f"   Top 10 sistemas:")
        sorted_systems = sorted(stats['systems'].items(), key=lambda x: x[1], reverse=True)[:10]
        for sys, count in sorted_systems:
            print(f"     - {sys}: {count} regras")
        print(f"   Top 10 propriedades:")
        sorted_props = sorted(stats['properties'].items(), key=lambda x: x[1], reverse=True)[:10]
        for prop, count in sorted_props:
            print(f"     - {prop}: {count} regras")
        
        print("\n" + "="*60)
        print("✓ PROCESSO CONCLUÍDO!")
        print("="*60)
        print(f"\nArquivo gerado:")
        print(f"  - {jsonl_file} (JSONL com TargetRules)")
        
    else:
        print(f"\n⚠ Arquivo {loinc_csv} não encontrado.")
        print("Usando o fluxo alternativo com CSV de exemplo...")
        
        # 1. Criar CSV de exemplo
        print("\n1. Criando CSV de exemplo...")
        csv_file = create_sample_csv("exames_componentes.csv")
        
        # 2. Gerar JSON de regras
        print("\n2. Gerando regras a partir do CSV...")
        generator = TargetRulesGeneratorOriginal(csv_file)
        generator.generate_rules()
        
        # 3. Salvar JSON
        json_file = "target_rules_exames.json"
        generator.save_json(json_file)
        
        # 4. Estatísticas
        print("\n3. Estatísticas das regras:")
        stats = generator.get_statistics()
        print(f"   Total de regras: {stats['total_rules']}")
        print(f"   Componentes únicos: {stats['unique_components']}")
        print(f"   Categorias:")
        for cat, count in stats['categories'].items():
            print(f"     - {cat}: {count} regras")
        
        # 5. Testar com MedSpaCy
        print("\n4. Testando integração com MedSpaCy...")
        try:
            nlp = load_rules_in_medspacy(json_file)
            test_processor(nlp)
        except ImportError:
            print("   ⚠ MedSpaCy não instalado. Instale com: pip install medspacy")
            print("   As regras JSON foram geradas e podem ser usadas quando instalar MedSpaCy")
        
        print("\n" + "="*60)
        print("✓ PROCESSO CONCLUÍDO!")
        print("="*60)
        print(f"\nArquivos gerados:")
        print(f"  - {csv_file} (CSV com componentes e sinônimos)")
        print(f"  - {json_file} (JSON com TargetRules)")
    
    print(f"\nPara usar no MedSpaCy:")
    print(f"  from medspacy.ner import TargetRule")
    print(f"  import json")
    print(f"  ")
    print(f"  # Para JSONL:")
    print(f"  rules = []")
    print(f"  with open('target_rules_loinc.jsonl', 'r') as f:")
    print(f"      for line in f:")
    print(f"          rules.append(json.loads(line))")