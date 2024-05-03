from pathlib import Path


def get_all_files(
    folder:Path,
    exts=[
        ".png",
        ".jpg",
        ".jpeg",
        ".PNG",
        ".JPG",
        ".JPEG",
    ],
):
    return [f for f in folder.iterdir() if f.suffix.lower() in exts]
