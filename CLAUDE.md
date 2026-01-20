# Discord Conversation Synthesizer

Bot que transforma conversas orgânicas de comunidades no Discord em conhecimento estruturado e indexado.

## Visão Geral

O Synthesizer processa o histórico de canais do Discord, segmenta mensagens em conversas coerentes, e usa Claude Opus para sintetizar cada conversa em explicações claras e intuitivas no estilo 3Blue1Brown.

### Filosofia Central

Comunidades discutindo tópicos complexos geram valor intelectual que se perde no fluxo efêmero do chat. Este projeto captura esse valor através de:

1. **Preservar atribuição** — Ideias permanecem conectadas às pessoas que as contribuíram
2. **Construir intuição** — Sínteses explicam *por que* as ideias importam, não apenas *o que* foi dito
3. **Criar estrutura** — Conversas viram notas atômicas em vault Obsidian, linkadas por tópico
4. **Permitir descoberta** — Base de conhecimento navegável emerge do caos

## Comandos

```bash
# Rodar com mocks (desenvolvimento/teste)
uv run python -m src

# Preview sem executar (dry run)
uv run python -m src --dry-run

# Rodar com Discord + Claude reais
uv run python -m src --real

# Gerar apenas estatísticas (sem chamar APIs)
uv run python -m src --stats-only

# Usar config customizada
uv run python -m src --config config/my-config.yaml

# Verbose mode
uv run python -m src -v
```

## Estrutura do Output

```
output/
├── conversations/       # Notas sintetizadas (uma por conversa)
│   └── _versions/      # Versões arquivadas de notas re-processadas
├── topics/             # Índices por tag (auto-gerados)
├── participants/       # Perfis de participantes
│   └── _index.md      # Ranking geral da comunidade
├── _meta/             # Metadados e estado
│   ├── processing-state.json  # Estado do processamento incremental
│   ├── run-history.json       # Histórico de execuções
│   ├── stats.json             # Estatísticas da comunidade
│   └── links.json             # Links extraídos
└── _transcripts/      # Transcrições originais das conversas
```

## Arquitetura

```
Discord Server
      │
      ▼
┌─────────────────┐
│  FETCHER        │  Discord.py, rate-limited
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  SEGMENTER      │  Gap temporal >24h, threads, reply chains
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  SYNTHESIZER    │  Claude Opus, estilo 3Blue1Brown
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  EXPORTER       │  Markdown + YAML frontmatter + wikilinks
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  OBSIDIAN VAULT │  /conversations, /topics, /participants
└─────────────────┘
```

## Módulos

| Módulo | Responsabilidade |
|--------|------------------|
| `src/__main__.py` | CLI e orquestração da pipeline |
| `src/models.py` | Data classes (Message, Conversation, SynthesizedNote) |
| `src/config.py` | Carregamento de configuração YAML |
| `src/fetcher.py` | Busca mensagens do Discord API |
| `src/segmenter.py` | Agrupa mensagens em conversas |
| `src/synthesizer.py` | Chama Claude para síntese |
| `src/exporter.py` | Exporta para markdown + gerencia estado |
| `src/stats.py` | Calcula estatísticas sem LLM |
| `src/profiles.py` | Gera perfis de participantes |
| `src/links.py` | Extrai e cataloga links compartilhados |

## Configuração

Editar `config/config.yaml`:

```yaml
discord:
  token: ${DISCORD_TOKEN}  # Variável de ambiente

server:
  id: "123456789"
  name: "Minha Comunidade"
  channels:
    - id: "987654321"
      name: "general"
      enabled: true

segmentation:
  temporal_gap_hours: 24  # Gap > 24h = nova conversa
  min_messages_per_conversation: 3

synthesis:
  model: "claude-opus-4-5-20250514"
  max_tokens: 4096
  temperature: 0.3

export:
  vault_path: "./output"
  archive_versions: true
  generate_topic_indexes: true

privacy:
  excluded_user_ids: []  # IDs de usuários para não sintetizar
  redaction_placeholder: "[REDACTED]"
```

## Variáveis de Ambiente

```bash
export DISCORD_TOKEN="seu-token-do-discord"
export ANTHROPIC_API_KEY="sua-api-key-anthropic"
```

## Regras de Segmentação

1. **Thread = 1 Conversa**: Qualquer thread do Discord é tratada como uma conversa única
2. **Gap Temporal**: Gap > 24h entre mensagens = nova conversa
3. **Reply Chains**: Mensagens conectadas por replies permanecem juntas
4. **Mínimo de Mensagens**: Conversas com < 3 mensagens são ignoradas

## Formato da Síntese

Cada nota gerada segue este formato:

```markdown
---
title: "Título Descritivo"
date: "2026-01-15"
timespan: "10:30 - 14:45 UTC"
participants: ["alice", "bob", "charlie"]
channel: "#trading"
tags: ["machine-learning", "portfolio-allocation"]
related: ["[[Factor Models]]", "[[Risk Management]]"]
---

# Título Descritivo

## Summary
2-3 frases sobre o que a conversa alcançou.

## The Core Idea
Explicação principal no estilo 3Blue1Brown...

## Key Contributions
- **alice**: Contribuição específica
- **bob**: Outra contribuição

## Points of Tension
Onde houve desacordo ou questões não resolvidas.

## Connections
Como isso se relaciona com outros conceitos.

---

<details>
<summary>Source Messages (click to expand)</summary>

**Conversation ID:** `channel_123_2026-01-15T10:30:00`
**Message count:** 15

**Discord Links:**
- [`1234567890`](https://discord.com/channels/server/channel/1234567890)
- [`1234567891`](https://discord.com/channels/server/channel/1234567891)
- ...

</details>
```

## Rastreabilidade e Reconstrução

Cada síntese mantém referências completas para as mensagens originais:

1. **Message IDs**: Todos os IDs de mensagens do Discord são preservados
2. **Discord Deep Links**: Links clicáveis para cada mensagem (formato `discord.com/channels/{server}/{channel}/{message}`)
3. **Transcripts**: Arquivo `.txt` em `_transcripts/` com o transcript original incluindo IDs
4. **Conversation ID**: Identificador único para reconstruir a conversa

### Formato do Transcript

```
CHANNEL: #trading
THREAD: main channel
TIMESPAN: 2026-01-15 10:30 to 2026-01-15 14:45
PARTICIPANTS: alice, bob, charlie
MESSAGE COUNT: 15
MESSAGE IDS: 1234567890, 1234567891, 1234567892, ...

[1234567890] [10:30:00] alice: Hello, this is the first message
[1234567891] [10:31:15] bob: Here's my response
[1234567892] [10:32:00] charlie: And my contribution
```

Isso permite:
- **Verificar fontes**: Clicar no link do Discord para ver a mensagem original
- **Reconstruir input**: Usar os IDs para re-processar a conversa
- **Auditar sínteses**: Comparar o output com o input original

## Custos Estimados

- **Claude Opus 4.5**: ~$0.05-0.15 por conversa
- Input: $15/M tokens
- Output: $75/M tokens

Use `--dry-run` para estimar custos antes de processar.

## Limitações Conhecidas

1. **Imagens**: Atualmente apenas URLs são preservados (não há download local)
2. **Links**: Extração básica por regex (sem fetch de metadados)
3. **Threads Longas**: Threads com 500+ mensagens podem precisar de chunking manual
4. **Rate Limits**: Respeita rate limits do Discord, pode ser lento para históricos grandes

## Desenvolvimento

```bash
# Instalar dependências
uv sync

# Rodar testes
uv run pytest

# Rodar com mocks para desenvolvimento
uv run python -m src
```

## Próximos Passos

- [ ] Download e cache local de imagens
- [ ] Fetch de metadados de links (OpenGraph)
- [ ] Interface web para navegação (Quartz)
- [ ] Webhooks para processamento em tempo real
