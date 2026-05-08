import click

from .config import DEFAULT_MEM_MB, WORKERS, MIN_FREQ
from .pipeline import run_pipeline


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
              help="Enable POS tagging (output word\\tfreq\\tpos, default: word\\tfreq).")
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

    Default: word    freq
    --pos:    word    freq    pos
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


if __name__ == "__main__":
    main()
