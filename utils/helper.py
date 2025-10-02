import sys
import os

def import_backend():
    # Add backend folder to path
    backend_path = os.path.join(os.path.dirname(__file__), "..", "backend")
    sys.path.append(backend_path)
    import backend
    return backend
