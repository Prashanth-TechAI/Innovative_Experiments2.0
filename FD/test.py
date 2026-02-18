import os

REPO_PATH = 'textile'
OUTPUT_FILE = 'mongo+react.txt'
INCLUDE_EXTS = {'.py', '.js', '.ts', '.md', '.json', '.txt'}

def is_text_file(fname):
    return os.path.splitext(fname)[1] in INCLUDE_EXTS

def dump_repo(repo_path, output_file):
    repo_path = os.path.abspath(repo_path)
    print("📁 Repository path:", repo_path)
    if not os.path.isdir(repo_path):
        raise FileNotFoundError(f"Directory not found: {repo_path}")

    with open(output_file, 'w', encoding='utf-8') as out:
        out.write(f"# Repository dump: {repo_path}\n\n")
        for root, dirs, files in os.walk(repo_path):
            # debug
            print(f"Entering dir: {root} ({len(files)} files)")
            if '.git' in root.split(os.sep):
                continue
            files = sorted(files)
            for fname in files:
                if not is_text_file(fname):
                    continue
                full = os.path.join(root, fname)
                rel = os.path.relpath(full, repo_path)
                out.write(f"## File: {rel}\n\n")
                try:
                    with open(full, 'r', encoding='utf-8') as f:
                        content = f.read()
                    out.write(content + "\n\n")
                except Exception as e:
                    out.write(f"[Error reading file: {e}]\n\n")
        out.write("\n--- End of repository dump ---\n")

if __name__ == "__main__":
    dump_repo(REPO_PATH, OUTPUT_FILE)
    print("✅ Dump complete:", OUTPUT_FILE)
