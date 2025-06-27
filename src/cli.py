"""
Command-line interface for the Semantic Codebase Graph Engine.

Provides commands for analyzing repositories and computing
similarity scores.
"""

import sys
from pathlib import Path
from typing import Optional

import click

from src.utils.logging_config import setup_logging
from src.utils.validation import validate_source


@click.group()
@click.version_option(version="1.0.0")
@click.option(
    "--verbose", "-v",
    is_flag=True,
    help="Enable verbose output"
)
@click.option(
    "--log-file",
    type=click.Path(),
    help="Path to log file"
)
@click.pass_context
def cli(ctx, verbose, log_file):
    """
    Semantic Codebase Graph Engine
    
    Analyze repositories and compute similarity scores using
    structural and semantic analysis.
    """
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose

    log_level = "DEBUG" if verbose else "INFO"
    setup_logging(level=log_level, log_file=Path(log_file) if log_file else None)


@cli.command()
@click.argument("source")
@click.argument("target")
@click.option(
    "--output", "-o",
    type=click.Path(),
    help="Output base path for reports (both .txt and .json will be generated)"
)
@click.option(
    "--format", "-f",
    type=click.Choice(["text", "json", "both"]),
    default="both",
    help="Output format (default: both)"
)
@click.option(
    "--structural-weight",
    type=float,
    default=0.4,
    help="Weight for structural similarity (0-1)"
)
@click.option(
    "--semantic-weight",
    type=float,
    default=0.6,
    help="Weight for semantic similarity (0-1)"
)
@click.pass_context
def compare(ctx, source, target, output, format, structural_weight, semantic_weight):
    """
    Compare two repositories for similarity.
    
    SOURCE and TARGET can be local paths or GitHub URLs.
    
    Examples:
    
        scge compare ./repo1 ./repo2
        
        scge compare https://github.com/user/repo1 https://github.com/user/repo2
        
        scge compare ./repo1 https://github.com/user/repo2 -o report
    """
    source_type, is_valid, error = validate_source(source)
    if not is_valid:
        click.echo(f"Error: {error}", err=True)
        sys.exit(1)

    target_type, is_valid, error = validate_source(target)
    if not is_valid:
        click.echo(f"Error: {error}", err=True)
        sys.exit(1)

    click.echo(f"Source: {source} ({source_type})")
    click.echo(f"Target: {target} ({target_type})")
    click.echo()

    from src.core.config import Config

    config = Config.get()
    config.similarity.structural_weight = structural_weight
    config.similarity.semantic_weight = semantic_weight

    total = structural_weight + semantic_weight
    config.similarity.structural_weight /= total
    config.similarity.semantic_weight /= total

    from src.engine import SemanticCodebaseEngine

    try:
        click.echo("Initializing analysis engine...")
        engine = SemanticCodebaseEngine(config)

        click.echo("Running analysis pipeline...")
        click.echo()

        output_path = Path(output) if output else None
        
        if format == "both" and output_path:
            from src.reporting.formatter import JSONFormatter, TextFormatter
            
            txt_path = Path(str(output_path) + ".txt")
            result = engine.analyze(source, target, "text", txt_path)
            
            json_formatter = JSONFormatter()
            json_path = Path(str(output_path) + ".json")
            json_formatter.save(result["report_obj"], json_path)
            
            click.echo(f"\nReports saved to:")
            click.echo(f"  Text: {txt_path}")
            click.echo(f"  JSON: {json_path}")
        else:
            actual_format = "text" if format == "both" else format
            result = engine.analyze(source, target, actual_format, output_path)

        if "similarity" in result:
            sim = result["similarity"]
            click.echo("=" * 60)
            click.echo("SIMILARITY RESULTS")
            click.echo("=" * 60)
            click.echo(f"Overall Similarity:    {sim['overall']:.2%}")
            click.echo(f"Structural Similarity: {sim['structural']:.2%}")
            click.echo(f"Semantic Similarity:   {sim['semantic']:.2%}")
            click.echo("=" * 60)

        if output and format != "both":
            click.echo(f"\nReport saved to: {output}")
        elif "report" in result and format == "text":
            click.echo("\n" + result["report"])

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        if ctx.obj.get("verbose"):
            import traceback
            traceback.print_exc()
        sys.exit(1)


@cli.command()
@click.argument("source")
@click.option(
    "--output", "-o",
    type=click.Path(),
    help="Output file for report"
)
@click.option(
    "--format", "-f",
    type=click.Choice(["text", "json"]),
    default="text",
    help="Output format"
)
@click.pass_context
def analyze(ctx, source, output, format):
    """
    Analyze a single repository.
    
    SOURCE can be a local path or GitHub URL.
    
    Examples:
    
        scge analyze ./my-project
        
        scge analyze https://github.com/user/repo -o analysis.json -f json
    """
    source_type, is_valid, error = validate_source(source)
    if not is_valid:
        click.echo(f"Error: {error}", err=True)
        sys.exit(1)

    click.echo(f"Source: {source} ({source_type})")
    click.echo()

    from src.engine import SemanticCodebaseEngine

    try:
        click.echo("Initializing analysis engine...")
        engine = SemanticCodebaseEngine()

        click.echo("Running analysis pipeline...")
        click.echo()

        output_path = Path(output) if output else None
        result = engine.analyze(source, output_format=format, output_path=output_path)

        click.echo("=" * 60)
        click.echo("ANALYSIS COMPLETE")
        click.echo("=" * 60)
        click.echo(f"Pipeline ID: {result['pipeline_id']}")
        click.echo(f"Status: {result['status']}")
        click.echo("=" * 60)

        if output:
            click.echo(f"\nReport saved to: {output}")
        elif "report" in result and format == "text":
            click.echo("\n" + result["report"])

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        if ctx.obj.get("verbose"):
            import traceback
            traceback.print_exc()
        sys.exit(1)


@cli.command()
@click.option(
    "--output", "-o",
    type=click.Path(),
    default="config.json",
    help="Output path for configuration file"
)
def init(output):
    """
    Initialize configuration file.
    
    Creates a default configuration file that can be customized.
    """
    from src.core.config import Config

    Config.save_to_file(output)
    click.echo(f"Configuration saved to: {output}")


@cli.command()
def list_languages():
    """List supported programming languages."""
    from src.analysis.detector import LanguageDetector

    detector = LanguageDetector()
    languages = detector.get_supported_languages()

    click.echo("Supported Languages:")
    click.echo("-" * 40)
    for lang in sorted(languages):
        extensions = detector.get_extensions_for_language(lang)
        click.echo(f"  {lang}: {', '.join(extensions)}")


@cli.command()
def storage_stats():
    """Show storage statistics."""
    from src.storage.backend import JSONStorageBackend

    backend = JSONStorageBackend()
    stats = backend.get_storage_stats()

    click.echo("Storage Statistics:")
    click.echo("-" * 40)
    click.echo(f"  Stored graphs: {stats['graph_count']}")
    click.echo(f"  Total size: {stats['total_size_mb']:.2f} MB")
    click.echo(f"  Storage dir: {stats['storage_dir']}")


@cli.command()
@click.option(
    "--days",
    type=int,
    default=30,
    help="Remove graphs older than this many days"
)
@click.option(
    "--yes", "-y",
    is_flag=True,
    help="Skip confirmation prompt"
)
def cleanup(days, yes):
    """Clean up old stored graphs."""
    from src.storage.manager import StorageManager
    from src.core.config import Config

    config = Config.get()
    manager = StorageManager(config)

    graphs = manager.list_graphs()
    click.echo(f"Found {len(graphs)} stored graphs")

    if not yes:
        if not click.confirm(f"Remove graphs older than {days} days?"):
            click.echo("Cancelled")
            return

    deleted = manager.cleanup_old_graphs(days)
    click.echo(f"Deleted {deleted} old graphs")


def main():
    """Entry point for the CLI."""
    cli(obj={})


if __name__ == "__main__":
    main()

