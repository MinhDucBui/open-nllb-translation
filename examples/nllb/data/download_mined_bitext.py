from datasets import load_dataset
import argparse
import os
from nllb_lang_pairs import NLLB_PAIRS, CCMATRIX_PAIRS

LANGUAGES = [("kor_Hang", "vie_Latn")]


def download_lang_pair_dataset(target_directory: str, src_lang: str, tgt_lang: str, cache: str = None) -> None:
    lang_directory = "/".join([target_directory, f"{src_lang}-{tgt_lang}"])
    if not os.path.exists(lang_directory):
        os.mkdir(lang_directory)
    if cache:
        dataset = load_dataset("allenai/nllb", f"{src_lang}-{tgt_lang}", 
                               ignore_verifications=True, cache_dir=cache)
    else:
        dataset = load_dataset("allenai/nllb", f"{src_lang}-{tgt_lang}", ignore_verifications=True)
    
    f_src = open(f"{lang_directory}/allenai.nllb.{src_lang}", 'w', encoding='utf-8')
    f_tgt = open(f"{lang_directory}/allenai.nllb.{tgt_lang}", 'w', encoding='utf-8')
    for d in dataset["train"]["translation"]:
        f_src.write(f"{d[src_lang]}\n")
        f_tgt.write(f"{d[tgt_lang]}\n")
    f_src.close()
    f_tgt.close()
    print(f"Wrote {src_lang}-{tgt_lang}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Script to download allenai nllb data from HF hub."
    )
    parser.add_argument(
        "--directory",
        "-d",
        type=str,
        required=True,
        help="directory to save downloaded data",
    )
    parser.add_argument(
        "--cache",
        "-c",
        type=str,
        required=True,
        help="huggingface cache folde to save",
    )
    parser.add_argument(
        "--src_lang",
        "-s",
        type=str,
        default="kor_Hang",
        help="source language",
    )
    parser.add_argument(
        "--trg_lang",
        "-t",
        type=str,
        default="vie_Latn",
        help="target language"
    )
    parser.add_argument(
        "--minimal",
        "-m",
        # type=bool,
        help="launch a minimal test to see if everything is ok (two lang pairs)",
        action="store_true"
    )
    args = parser.parse_args()
    target_directory = "/".join([args.directory, "nllb"])
    if not os.path.exists(target_directory):
        os.mkdir(target_directory)

    if args.minimal:
        for src_lang, tgt_lang in [("ace_Latn", "ban_Latn"), ("amh_Ethi", "nus_Latn")]:
            download_lang_pair_dataset(target_directory, src_lang, tgt_lang)
    else:
        download_lang_pair_dataset(target_directory, args.src_lang, args.trg_lang, args.cache)
    print("done")
