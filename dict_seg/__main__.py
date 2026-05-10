import click

from .config import DEFAULT_MEM_MB, WORKERS, MIN_FREQ
from .pipeline import run_pipeline, merge_wordfreq_files


@click.command(context_settings={"show_default": True})
@click.argument("input_path", type=click.Path(exists=True, readable=True))
@click.option("--output", "-o", default=None,
              help="Output file path (auto-generated if omitted).")
@click.option("--mem", "-m", default=DEFAULT_MEM_MB,
              help="Memory budget in MB for external sort.")
@click.option("--workers", "-w", default=WORKERS,
              help="Number of worker processes.")
@click.option("--min-freq", default=MIN_FREQ,
              help="Minimum frequency threshold for output.")
@click.option("--pos", is_flag=True, default=False,
              help="Enable POS tagging (output word\\tpos\\tfreq).")
@click.option("--strip-html", is_flag=True, default=False,
              help="Strip HTML tags with BeautifulSoup before segmentation.")
@click.option("--force", is_flag=True, default=False,
              help="Overwrite output file if it already exists.")
def main(
    input_path: str,
    output: str | None,
    mem: int,
    workers: int,
    min_freq: int,
    pos: bool,
    strip_html: bool,
    force: bool,
) -> None:
    """Batch Chinese word segmentation with jieba and word frequency counting.

    INPUT_PATH can be a .txt file or a directory containing text files.

    Output format (tab-separated, sorted by frequency descending):

    \b
    Default: word    freq
    --pos:    word    pos    freq
    """
    run_pipeline(
        input_path=input_path,
        output_path=output,
        mem_mb=mem,
        workers=workers,
        min_freq=min_freq,
        no_pos=not pos,
        strip_html=strip_html,
        force=force,
    )


@click.command(context_settings={"show_default": True})
@click.argument("input_path", type=click.Path(exists=True, readable=True))
@click.option("--output", "-o", default=None,
              help="Output file path (auto-generated if omitted).")
@click.option("--mem", "-m", default=DEFAULT_MEM_MB,
              help="Memory budget in MB for external sort.")
@click.option("--workers", "-w", default=WORKERS,
              help="Number of worker processes.")
@click.option("--min-freq", default=MIN_FREQ,
              help="Minimum frequency threshold for output.")
@click.option("--pos", is_flag=True, default=False,
              help="Input files have POS column (word\\tpos\\tfreq).")
@click.option("--force", is_flag=True, default=False,
              help="Overwrite output file if it already exists.")
def merge(
    input_path: str,
    output: str | None,
    mem: int,
    workers: int,
    min_freq: int,
    pos: bool,
    force: bool,
) -> None:
    """Merge multiple _wordfreq.txt files from a directory into one.

    Collects all files containing 'wordfreq' in the name from INPUT_PATH,
    concatenates, sorts, merges same-word counts, filters, and outputs
    a single combined frequency file.
    """
    merge_wordfreq_files(
        input_path=input_path,
        output_path=output,
        mem_mb=mem,
        workers=workers,
        min_freq=min_freq,
        no_pos=not pos,
        force=force,
    )
