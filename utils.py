from dataclasses import dataclass
import hashlib

@dataclass
class FileInfo:
    name: str
    hash: str

def compute_file_hash(file_path: str) -> str:
    with open(file_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()