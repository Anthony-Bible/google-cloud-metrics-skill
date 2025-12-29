# Cloud Metrics Skill for Claude Code

A Claude Code skill that enables querying Google Cloud Monitoring metrics directly from your terminal. Query Kubernetes metrics, GCP resource usage, and export monitoring data with natural language.

## Features

- **Query metrics** - Fetch time series data with filtering, aggregation, and grouping
- **Describe metrics** - Discover metric metadata, available labels, and filter examples
- **List metrics** - Browse available metrics in your GCP project
- **Multiple output formats** - Table, JSON, or CSV output
- **Statistical analysis** - Built-in min, max, avg, p50, p95, p99 percentiles
- **Flexible time ranges** - Duration strings (`1h`, `30m`, `7d`) or ISO timestamps

## Prerequisites

1. **Python 3.10+** installed
2. **Google Cloud SDK** (`gcloud`) installed
3. **GCP Project** with Cloud Monitoring API enabled
4. **IAM Permissions** to read Cloud Monitoring metrics

### Authentication

Configure application default credentials:

```bash
gcloud auth application-default login
```

## Installation

### Option 1: Project-Level Installation

Install the skill for a specific project:

```bash
# From your project root
cp -r /path/to/cloud-metrics/.claude/skills/cloud-metrics .claude/skills/
```

Or manually create the structure:

```bash
mkdir -p .claude/skills/cloud-metrics/scripts
cp /path/to/cloud_metrics.py .claude/skills/cloud-metrics/scripts/
```

Then create `.claude/skills/cloud-metrics/SKILL.md` (see [SKILL.md template](#skillmd-template) below).

### Option 2: User-Level Installation (All Projects)

Install the skill globally so it's available in all your projects:

```bash
cp -r /path/to/cloud-metrics/.claude/skills/cloud-metrics ~/.claude/skills/
```

Or manually:

```bash
mkdir -p ~/.claude/skills/cloud-metrics/scripts
cp /path/to/cloud_metrics.py ~/.claude/skills/cloud-metrics/scripts/
# Then create SKILL.md as shown below
```

### SKILL.md Template

Create this file at `.claude/skills/cloud-metrics/SKILL.md` or `~/.claude/skills/cloud-metrics/SKILL.md`:

```markdown
---
name: cloud-metrics
description: Query Google Cloud Monitoring metrics using the cloud_metrics.py tool. Use when users ask about GCP metrics, Cloud Monitoring, Kubernetes metrics (CPU, memory, network), container resource usage, or need to export monitoring data. Triggers on requests like "show me CPU usage", "list available metrics", "describe this metric", "top memory consumers", or any Google Cloud Monitoring queries.
---

# Cloud Metrics

Query GCP Monitoring API using the bundled cloud_metrics.py script.

## Prerequisites

Requires GCP authentication:
\`\`\`bash
gcloud auth application-default login
\`\`\`

## Commands

\`\`\`bash
# Run with uv (handles dependencies automatically)
uv run scripts/cloud_metrics.py <command> [options]

# Commands
query     # Query metric data
describe  # Show metric labels and filter examples
list      # List available metrics
\`\`\`
```

## Usage

Once installed, just ask Claude Code about GCP metrics:

```
> Show me CPU usage for my Kubernetes containers

> List available metrics in project my-gcp-project

> Describe the kubernetes.io/container/memory/used_bytes metric

> Top 10 memory consumers in namespace production
```

## Command Reference

### `query` - Query metric data

```bash
cloud_metrics.py query --project PROJECT --metric METRIC [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `--project` | GCP project ID (required) |
| `--metric` | Metric type, e.g., `kubernetes.io/container/cpu/core_usage_time` |
| `--duration` | Time range, e.g., `1h`, `30m`, `7d` (default: `1h`) |
| `--start`, `--end` | ISO timestamps for custom range |
| `--filter` | Label filters (repeatable), e.g., `namespace_name=production` |
| `--align` | Per-series alignment: `ALIGN_RATE`, `ALIGN_MEAN`, `ALIGN_SUM`, etc. |
| `--reduce` | Cross-series reducer: `REDUCE_SUM`, `REDUCE_MEAN`, `REDUCE_MAX`, etc. |
| `--group-by` | Group results by labels (repeatable) |
| `--top` | Show top N results by value |
| `--format` | Output format: `table`, `json`, `csv` |
| `--stats` | Include percentile statistics |
| `--latest` | Show only the latest value per series |

### `describe` - Get metric metadata

```bash
cloud_metrics.py describe --project PROJECT --metric METRIC
```

Shows:
- Metric display name, description, kind, value type, unit
- Available metric labels
- Resource labels with sample values
- System metadata labels
- Filter examples

### `list` - List available metrics

```bash
cloud_metrics.py list --project PROJECT [--filter PREFIX]
```

| Option | Description |
|--------|-------------|
| `--project` | GCP project ID (required) |
| `--filter` | Filter by metric type prefix, e.g., `kubernetes.io` |

## Examples

### Query Kubernetes CPU usage

```bash
uv run --with 'google-cloud-monitoring>=2.0.0' cloud_metrics.py query \
  --project my-project \
  --metric kubernetes.io/container/cpu/core_usage_time \
  --duration 1h \
  --align ALIGN_RATE \
  --reduce REDUCE_SUM \
  --group-by namespace_name \
  --format table
```

### Find top memory consumers

```bash
uv run --with 'google-cloud-monitoring>=2.0.0' cloud_metrics.py query \
  --project my-project \
  --metric kubernetes.io/container/memory/used_bytes \
  --duration 30m \
  --latest \
  --top 10 \
  --format table
```

### Export metrics to CSV

```bash
uv run --with 'google-cloud-monitoring>=2.0.0' cloud_metrics.py query \
  --project my-project \
  --metric compute.googleapis.com/instance/cpu/utilization \
  --duration 24h \
  --format csv > metrics.csv
```

## Dependencies

The script uses `uv` to automatically install dependencies:

- `google-cloud-monitoring>=2.0.0`

No manual `pip install` required when using `uv run`.

## License

MIT
