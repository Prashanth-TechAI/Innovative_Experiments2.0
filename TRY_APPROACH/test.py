import os

# List of extensions you consider “frontend code” (adjust as needed)
FRONTEND_EXTS = {
    ".ts", ".tsx", ".js", ".jsx",
    ".json", ".md",
    ".css", ".scss", ".html",
    ".svg", ".png", ".jpg", ".jpeg"
}

def dump_frontend_files(root_dir: str, output_file: str):
    """
    Recursively walks `root_dir` (wednesai-agent-forge)
    and writes each “frontend” file’s path + contents into output_file.
    """
    with open(output_file, "w", encoding="utf-8") as out_f:
        for dirpath, dirnames, filenames in os.walk(root_dir):
            # Write folder heading
            out_f.write(f"\nDirectory: {dirpath}\n")
            out_f.write("─" * 80 + "\n")

            for name in filenames:
                ext = os.path.splitext(name)[1].lower()
                if ext not in FRONTEND_EXTS:
                    continue  # skip non‐frontend extensions

                file_path = os.path.join(dirpath, name)
                out_f.write(f"\nFile: {file_path}\n")
                out_f.write("─" * 40 + "\n")

                try:
                    with open(file_path, "r", encoding="utf-8") as in_f:
                        contents = in_f.read()
                        out_f.write(contents + "\n")
                except Exception as e:
                    out_f.write(f"⚠ Could not read file ({e})\n")

                out_f.write("─" * 80 + "\n")

if __name__ == "__main__":
    # Point this to your local clone of the frontend folder
    dump_frontend_files("ragizen-pipeline-builder-be60b036", "frontend_dump.txt")
