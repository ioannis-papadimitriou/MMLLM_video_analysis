import uuid
import os

# Check for a Docker-specific environment variables
host=os.getenv('DATABASE_HOST', 'localhost')
port=os.getenv('DATABASE_PORT', '5432')
dbname=os.getenv('DATABASE_NAME', 'memory_agent')
user=os.getenv('DATABASE_USER', 'ollama_agent')
password=os.getenv('DATABASE_PASSWORD', 'AGRAGMEMNON')

SAVE_FOLDER = os.environ.get('SAVE_FOLDER', './data')

# Postgres
DB_PARAMS = {
            'dbname': dbname,
            'user': user,
            'password': password,
            'host' : host,
            'port': port
        }

def file_name(name, extension):
    """
    Simple function to enable saving items in designated folder
    """
    unique_id = uuid.uuid4().hex  # Generate a unique identifier
    file_name = f"{name}_{unique_id}.{extension}"
    return os.path.join(SAVE_FOLDER, file_name)