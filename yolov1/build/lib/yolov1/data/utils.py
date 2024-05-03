from pathlib import Path


def get_all_files(
    folder,
    exts=[
        ".png",
        ".jpg",
        ".jpeg",
        ".PNG",
        ".JPG",
        ".JPEG",
    ],
):
    return [f for f in Path(folder).iterdir() if f.suffix.lower() in exts]
